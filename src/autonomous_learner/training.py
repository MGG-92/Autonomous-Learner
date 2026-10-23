"""Gold world access is restricted to expert supervision and external rewards."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .agent import Agent, run_episode
from .memory import Memory
from .policy import Controller, FEATURES
from .world import SyntheticTeacher, World


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    imitation_worlds: int = 160
    epochs: int = 12
    rl_episodes: int = 160
    learning_rate: float = 0.003
    max_steps: int = 8
    train_start: int = 0
    validation_start: int = 100_000
    validation_worlds: int = 40
    test_start: int = 200_000
    test_worlds: int = 100

    def validate(self):
        if min(self.imitation_worlds, self.epochs, self.validation_worlds, self.test_worlds, self.max_steps) < 1:
            raise ValueError("world counts, epochs, and budgets must be positive")
        if self.rl_episodes < 0 or self.learning_rate <= 0:
            raise ValueError("invalid optimization settings")
        ranges = [(self.train_start, self.train_start + self.imitation_worlds + self.rl_episodes),
                  (self.validation_start, self.validation_start + self.validation_worlds),
                  (self.test_start, self.test_start + self.test_worlds)]
        for i, (a, b) in enumerate(ranges):
            for c, d in ranges[i + 1:]:
                if max(a, c) < min(b, d):
                    raise ValueError("train, validation, and test seeds must be disjoint")


def expert_index(agent: Agent, world: World) -> int:
    if agent.memory.answer(agent.goal).value is not None:
        return next(i for i, a in enumerate(agent.actions) if a.kind == "ANSWER")
    # Gold access for training labels ONLY. Walk the proof and ask for its first gap.
    known = {f for _, f in agent.memory.facts()}
    subject = agent.goal.subject
    while True:
        fact = next((f for f in world.facts if f.subject == subject and f.relation == "is_a"), None)
        if fact is None:
            return next(i for i, a in enumerate(agent.actions)
                        if a.query and a.query.subject == subject and a.query.relation == "can"
                        and a.query.object == agent.goal.object)
        if fact not in known:
            return next(i for i, a in enumerate(agent.actions)
                        if a.query and a.query.subject == subject and a.query.relation == "is_a")
        subject = fact.object


def demonstrations(config: TrainConfig):
    examples = []
    for seed in range(config.train_start, config.train_start + config.imitation_worlds):
        world = World.generate(seed)
        memory = Memory()
        agent = Agent(memory, world.catalog(), world.goal(), config.max_steps)
        teacher = SyntheticTeacher(world)
        for step in range(config.max_steps):
            index = expert_index(agent, world)
            examples.append((agent.features(step), index))
            action = agent.actions[index]
            if action.kind == "ANSWER":
                break
            agent.execute(action, teacher)
        memory.close()
    return examples


def save_checkpoint(path: Path, policy: Controller, config: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "config": asdict(config), "features": FEATURES}, path)


def load_checkpoint(path: Path) -> tuple[Controller, TrainConfig]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(saved["features"]) != FEATURES:
        raise ValueError("Checkpoint feature schema differs from this code")
    policy = Controller()
    policy.load_state_dict(saved["state_dict"])
    policy.eval()
    return policy, TrainConfig(**saved["config"])


def train(config: TrainConfig, progress=None):
    from .evaluation import evaluate

    config.validate()
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    policy = Controller()
    initial = Controller()
    initial.load_state_dict(policy.state_dict())
    examples = demonstrations(config)
    x = torch.stack([features for features, _ in examples])
    targets = torch.tensor([index for _, index in examples])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    curve = []
    best_reward = float("-inf")
    best_state = None

    def measure(stage, epoch, loss=None):
        nonlocal best_reward, best_state
        metrics = evaluate(policy, config.validation_start, config.validation_worlds, config.max_steps)
        row = {"stage": stage, "iteration": epoch, "loss": loss, **metrics}
        curve.append(row)
        if metrics["mean_reward"] > best_reward:
            best_reward = metrics["mean_reward"]
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        if progress:
            progress(row)

    measure("initial", 0)
    for epoch in range(config.epochs):
        order = torch.randperm(len(x))
        losses = []
        for batch in order.split(64):
            loss = F.cross_entropy(policy(x[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        measure("imitation", epoch + 1, sum(losses) / len(losses))

    baseline = 0.0
    for episode in range(config.rl_episodes):
        world = World.generate(config.train_start + config.imitation_worlds + episode)
        memory = Memory()
        agent = Agent(memory, world.catalog(), world.goal(), config.max_steps)
        result, log_probs = run_episode(agent, SyntheticTeacher(world), policy, world.truth(world.goal()), sample=True)
        loss = -torch.stack(log_probs).sum() * (result.reward - baseline)
        baseline = 0.9 * baseline + 0.1 * result.reward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        memory.close()
        if (episode + 1) % 40 == 0 or episode + 1 == config.rl_episodes:
            measure("reinforce", episode + 1, float(loss.detach()))
    assert best_state is not None
    policy.load_state_dict(best_state)
    policy.eval()
    return policy, initial, curve
