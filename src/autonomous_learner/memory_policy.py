"""Learn retention priorities under a hard memory budget, separately from asking."""
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .memory import Memory
from .types import Fact, Observation, Query
from .world import World

MEMORY_FEATURES = ("goal_subject", "goal_property", "subject_reachable", "object_reachable",
                   "parent_edge", "depth", "recency")


def record_key(record: dict) -> tuple:
    return tuple(record[k] for k in ("subject", "relation", "object", "positive", "source"))


def retention_features(records: list[dict], goal: Query) -> torch.Tensor:
    graph = Memory(capacity=len(records))
    for row in records:
        graph.store(Observation(Query(row["subject"], row["relation"], row["object"]),
                    (Fact(row["subject"], row["relation"], row["object"], bool(row["positive"])),), row["source"]))
    reachable = graph.reachable(goal.subject)
    graph.close()
    return torch.tensor([
        [r["subject"] == goal.subject,
         r["relation"] == goal.relation and r["object"] == goal.object,
         r["subject"] in reachable, r["object"] in reachable,
         r["relation"] == "is_a", len(reachable.get(r["subject"], ())) / 3,
         (i + 1) / len(records)] for i, r in enumerate(records)
    ], dtype=torch.float32)


class RetentionController(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(MEMORY_FEATURES), 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def load_retention(path: Path) -> RetentionController:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(saved["features"]) != MEMORY_FEATURES:
        raise ValueError("Retention checkpoint feature schema differs")
    scorer = RetentionController()
    scorer.load_state_dict(saved["state_dict"])
    return scorer.eval()


class DualMemory:
    """Bounded episodic and belief graphs. Only their contents are agent-accessible.

    Consolidation timing is fixed (each observation). A learned scorer chooses
    retention priorities; this does not yet learn whether or when to consolidate.
    """
    def __init__(self, goal: Query, scorer: RetentionController | str = "fifo", path: Path | None = None,
                 episodic_capacity: int = 4, belief_capacity: int = 3):
        self.goal, self.scorer = goal, scorer
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes = Memory(str(path) + ".episodes.sqlite" if path else ":memory:", episodic_capacity)
        self.beliefs = Memory(str(path) + ".beliefs.sqlite" if path else ":memory:", belief_capacity)

    def close(self):
        self.episodes.close()
        self.beliefs.close()

    def store(self, observation: Observation):
        self.episodes.store(observation)
        unique = {}
        for row in self.beliefs.inspect() + self.episodes.inspect():
            key = record_key(row)
            if key not in unique or row["observed_at"] < unique[key]["observed_at"]:
                unique[key] = row
        records = sorted(unique.values(), key=lambda row: row["observed_at"])
        if not records:
            return
        features = retention_features(records, self.goal)
        with torch.no_grad():
            if isinstance(self.scorer, RetentionController):
                scores = self.scorer(features)
            elif self.scorer == "heuristic":
                scores = (features[:, 0] * 4 + features[:, 1] * features[:, 2] * 4
                          + features[:, 4] * features[:, 2] * 4)
            elif self.scorer == "fifo":
                scores = features[:, -1]
            else:
                raise ValueError("Unknown retention scorer")
        count = min(self.beliefs.capacity, len(records))
        chosen = torch.argsort(scores, descending=True, stable=True)[:count].tolist()
        self.beliefs.retain([records[i] for i in chosen])

    def answer(self, goal: Query):
        # Delayed evaluation reads consolidated beliefs only. Episodic replay cannot
        # recover deleted information from researcher logs.
        return self.beliefs.answer(goal)

    def reachable(self, subject):
        return self.beliefs.reachable(subject)

    def facts(self):
        return self.beliefs.facts()


def retention_stream(seed: int):
    world = World.generate(seed)
    goal = world.goal()
    subject, proof = goal.subject, []
    while True:
        edge = next((f for f in world.facts if f.subject == subject and f.relation == "is_a"), None)
        if edge is None:
            proof.append(next(f for f in world.facts if f.subject == subject
                              and f.relation == goal.relation and f.object == goal.object))
            break
        proof.append(edge)
        subject = edge.object
    distractors = [f for f in world.facts if f not in proof]
    random.Random(seed + 31).shuffle(distractors)
    stream = [Observation(Query(f.subject, f.relation, f.object), (f,), "synthetic-oracle")
              for f in proof + distractors]
    return world, goal, proof, stream


def evaluate_retention(scorer, start: int, worlds: int = 100):
    correct = 0
    for seed in range(start, start + worlds):
        world, goal, _, stream = retention_stream(seed)
        memory = DualMemory(goal, scorer)
        for observation in stream:
            memory.store(observation)
        correct += memory.answer(goal).value == world.truth(goal)
        assert len(memory.beliefs.inspect()) <= 3 and len(memory.episodes.inspect()) <= 4
        memory.close()
    return {"worlds": worlds, "retention_accuracy": correct / worlds,
            "episodic_capacity": 4, "belief_capacity": 3}


def train_retention(seed: int = 42, worlds: int = 80, epochs: int = 20):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    scorer = RetentionController()
    initial = RetentionController()
    initial.load_state_dict(scorer.state_dict())
    examples, targets = [], []
    # Training labels identify proof facts. Feature extraction itself sees observations only.
    for world_seed in range(worlds):
        _, goal, proof, stream = retention_stream(world_seed)
        memory = Memory(capacity=32)
        for observation in stream:
            memory.store(observation)
            records = memory.inspect()
            examples.append(retention_features(records, goal))
            targets.append(torch.tensor([Fact(r["subject"], r["relation"], r["object"], bool(r["positive"])) in proof
                                         for r in records], dtype=torch.float32))
        memory.close()
    x, y = torch.cat(examples), torch.cat(targets)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=0.01)
    losses = []
    best_score, best_state = -1, None
    for epoch in range(epochs):
        batch_losses = []
        for batch in torch.randperm(len(x)).split(256):
            loss = F.binary_cross_entropy_with_logits(scorer(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        losses.append(sum(batch_losses) / len(batch_losses))
        # Selection uses validation worlds, never the final test seeds.
        score = evaluate_retention(scorer, 100_000, 10)["retention_accuracy"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in scorer.state_dict().items()}
    scorer.load_state_dict(best_state)
    return scorer.eval(), initial.eval(), losses


def memory_experiment(output: Path):
    if output.exists():
        raise FileExistsError("Use a new experiment directory")
    output.mkdir(parents=True)
    scorer, initial, losses = train_retention()
    torch.save({"state_dict": scorer.state_dict(), "features": MEMORY_FEATURES}, output / "retention.pt")
    results = {name: evaluate_retention(model, 200_000) for name, model in
               [("learned", scorer), ("untrained", initial), ("fifo", "fifo"), ("heuristic", "heuristic")]}
    report = {"seed": 42, "train_worlds": 80, "epochs": 20, "train_start": 0,
              "validation_start": 100_000, "validation_worlds": 10, "test_start": 200_000,
              "losses": losses, "results": results,
              "limitations": ["Future query is supplied as a task hint", "Proof evidence precedes distractors",
                              "Consolidation timing and memory budgets are fixed",
                              "Supervised proof-retention objective; no learned retrieval or forgetting schedule"]}
    import platform
    import subprocess
    report["environment"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                             "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    import json
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
