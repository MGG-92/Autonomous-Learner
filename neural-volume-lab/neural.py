"""Bounded spatial recurrent learner. No language model, text rules or reply lookup."""
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    side: int = 25
    degree: int = 12
    hidden: int = 96
    slots: int = 16
    vocabulary: int = 128
    seed: int = 42
    replay: int = 8


class SpatialNetwork(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        n = config.side ** 3
        if config.side < 3 or not 1 <= config.degree < n // 2:
            raise ValueError("Invalid grid or connection degree")
        rng = np.random.default_rng(config.seed)
        # Positive cyclic offsets strictly below N/2 prohibit reciprocal edges.
        # Local grid offsets plus long-range offsets; no claim of brain realism.
        offsets = list(dict.fromkeys([1, config.side, config.side ** 2]))
        while len(offsets) < config.degree:
            candidate = int(rng.integers(1, n // 2))
            if candidate not in offsets:
                offsets.append(candidate)
        offsets = offsets[:config.degree]
        sources = (np.arange(n)[:, None] - np.array(offsets)[None, :]) % n
        self.register_buffer("sources", torch.tensor(sources, dtype=torch.long))
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            self.edges = nn.Parameter(torch.randn(n, config.degree) * (.25 / config.degree))
            self.encoder = nn.Linear(n, config.hidden, bias=False)
            self.decoder = nn.Linear(config.hidden, (config.slots + 1) * config.vocabulary)
            nn.init.zeros_(self.decoder.bias)
        self.register_buffer("activity", torch.zeros(n), persistent=False)

    def pulse(self, points):
        points = torch.as_tensor(np.asarray(points), dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all():
            raise ValueError("Expected finite XYZ vertices")
        xyz = ((points.clamp(-1, 1) + 1) * (self.config.side - 1) / 2).round().long()
        ids = (xyz[:, 0] * self.config.side + xyz[:, 1]) * self.config.side + xyz[:, 2]
        pulse = torch.zeros(self.config.side ** 3)
        pulse[ids] = 1
        return pulse

    def tick(self, state, pulse=None):
        drive = (state[self.sources] * self.edges).sum(-1)
        if pulse is not None:
            drive = drive + pulse
        return (.15 * state + .85 * F.relu(drive)).clamp(max=4)

    def forward(self, pulses):
        state = torch.zeros_like(self.activity)
        for pulse in pulses:
            state = self.tick(state, pulse)
            state = self.tick(state)
        return self.readout(state), state

    def readout(self, state):
        logits = self.decoder(F.relu(self.encoder(state)))
        return logits.reshape(self.config.slots + 1, self.config.vocabulary)

    @torch.no_grad()
    def constrain(self):
        # Row L1 bound + leak is a strict contraction without external input.
        total = self.edges.abs().sum(-1, keepdim=True)
        self.edges.mul_((.7 / total.clamp(min=.7)).clamp(max=1))


class Learner:
    def __init__(self, config=Config()):
        self.config = config
        self.network = SpatialNetwork(config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=.003)
        self.vocabulary = []  # Readout support, not a mapping from messages to responses.
        self.examples = []
        self.example_metadata = []
        self.updates = 0
        self.feedback_optimizer = torch.optim.SGD(self.network.parameters(), lr=.05)
        self.sampling = torch.Generator().manual_seed(config.seed)

    def encode(self, sequence):
        tokens = sequence["tokens"]
        if not 1 <= len(tokens) <= self.config.slots:
            raise ValueError(f"Network supports 1–{self.config.slots} input/target tokens")
        return torch.stack([self.network.pulse(t["points"]) for t in tokens])

    def admit(self, message, target, source="user", replace=False, context=None):
        pulses = self.encode(message)
        if not 1 <= len(target["tokens"]) <= self.config.slots:
            raise ValueError("Target exceeds network response slots")
        ids = [t["id"] for t in target["tokens"]]
        key = tuple(t["id"] for t in message["tokens"])
        if len(self.example_metadata) != len(self.examples):
            self.example_metadata = [None] * len(self.examples)
        if source == "professor" and any(meta and meta[0] == key and meta[1] != "professor"
                                         for meta in self.example_metadata):
            return False  # Explicit human examples take priority while retained.
        new = list(dict.fromkeys(self.vocabulary + ids))
        if len(new) >= self.config.vocabulary:
            raise ValueError("Readout vocabulary full; reset learning to start another experiment")
        self.vocabulary = new
        labels = torch.zeros(self.config.slots + 1, dtype=torch.long)
        labels[:len(ids)] = torch.tensor([self.vocabulary.index(i) + 1 for i in ids])
        # Replay contains training examples only; predictions never query it.
        if replace:
            retained = [(example, meta) for example, meta in zip(self.examples, self.example_metadata)
                        if not meta or meta[0] != key]
            self.examples = [example for example, _ in retained]
            self.example_metadata = [meta for _, meta in retained]
        self.examples.append((context if context is not None else pulses, labels))
        self.example_metadata.append((key, source))
        self.examples = self.examples[-self.config.replay:]
        self.example_metadata = self.example_metadata[-self.config.replay:]
        return True

    def train(self, epochs=10):
        if not self.examples:
            raise ValueError("Provide an expected response before training")
        if not 1 <= epochs <= 100:
            raise ValueError("Use 1–100 epochs per request")
        first = None
        for _ in range(epochs):
            self.optimizer.zero_grad(set_to_none=True)
            loss = 0
            for pulses, labels in self.examples:
                logits, _ = self.evaluate(pulses)
                # Ignore unused classes, but include learned EOS (class zero).
                logits = logits[:, :len(self.vocabulary) + 1]
                # Score tokens and first EOS equally; padding does not dominate.
                length = int((labels != 0).sum()) + 1
                loss = loss + F.cross_entropy(logits[:length], labels[:length])
            loss = loss / len(self.examples)
            if first is None:
                first = float(loss.detach())
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.optimizer.step()
            self.network.constrain()
            self.updates += 1
        return {"initial_loss": first, "loss": float(loss.detach()), "updates": self.updates,
                "replay_pairs": len(self.examples), "vocabulary": len(self.vocabulary)}

    @torch.no_grad()
    def predict(self, message):
        logits, state = self.network(self.encode(message))
        self.network.activity.copy_(state)
        if not self.vocabulary:
            return []
        indices = logits[:, :len(self.vocabulary) + 1].argmax(-1).tolist()
        result = []
        for index in indices[:self.config.slots]:
            if index == 0:
                break
            result.append(self.vocabulary[index - 1])
        return result

    @torch.no_grad()
    def sample(self, message=None, context=None):
        """Sample the network, retaining its behavior-policy probabilities for credit."""
        pulses = context if context is not None else self.encode(message)
        logits, state = self.evaluate(pulses)
        self.network.activity.copy_(state)
        distribution = logits[:, :len(self.vocabulary) + 1].log_softmax(-1)
        classes, old_log_probs, ids = [], [], []
        for slot in range(self.config.slots):
            action = int(torch.multinomial(distribution[slot].exp(), 1, generator=self.sampling))
            classes.append(action)
            old_log_probs.append(float(distribution[slot, action]))
            if action == 0:
                break
            ids.append(self.vocabulary[action - 1])
        trajectory = {"pulses": pulses, "classes": classes, "old_log_probs": old_log_probs,
                      "vocabulary": tuple(self.vocabulary), "rated": set()}
        return ids, trajectory

    def evaluate(self, context):
        if isinstance(context, dict):
            # Credit one real transition; earlier activity is a detached observation.
            state = self.network.tick(context["state_before"], context["pulse"])
            return self.network.readout(state), state
        return self.network(context)

    def feedback(self, trajectory, reward, source="human", strength=5):
        """One bounded, clipped token-policy update per source on the actual sampled reply.

        This is immediate-reward contextual-bandit feedback, not a long-horizon RL
        algorithm. Per-token probability ratios permit bounded reuse after teaching.
        """
        if source not in ("human", "professor") or reward not in (-1, 0, 1):
            raise ValueError("Feedback requires a known source and reward -1, 0 or +1")
        if not isinstance(strength, (int, float)) or not 1 <= strength <= 10:
            raise ValueError("Feedback strength must be between 1 and 10")
        if source in trajectory["rated"]:
            raise ValueError("This source already rated this reply")
        old_support = trajectory["vocabulary"]
        if tuple(self.vocabulary[:len(old_support)]) != old_support:
            raise ValueError("Reply belongs to a different vocabulary; send another message")
        weight = 1. if source == "human" else .25
        classes = torch.tensor(trajectory["classes"], dtype=torch.long)
        old = torch.tensor(trajectory["old_log_probs"])
        logits, _ = self.evaluate(trajectory["pulses"])
        log_probs = logits[:len(classes), :len(self.vocabulary) + 1].log_softmax(-1)
        selected = log_probs.gather(1, classes[:, None]).squeeze(1)
        before = float(selected.mean().detach())
        ratio = (selected - old).clamp(-20, 20).exp()
        advantage = float(reward) * weight
        loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
        changed = reward != 0 and bool(self.vocabulary)
        if changed:
            self.feedback_optimizer.param_groups[0]["lr"] = .05 * strength
            self.feedback_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.feedback_optimizer.step()
            self.network.constrain()
            self.updates += 1
        trajectory["rated"].add(source)
        with torch.no_grad():
            logits, _ = self.evaluate(trajectory["pulses"])
            after = float(logits[:len(classes), :len(self.vocabulary)+1].log_softmax(-1)
                          .gather(1, classes[:, None]).mean())
        return {"kind": "reward_feedback", "source": source, "reward": reward,
                "weight": weight, "update_attempted": changed, "updates": self.updates,
                "strength": strength, "step_size": .05 * strength,
                "mean_log_probability_before": before, "mean_log_probability_after": after,
                "loss": float(loss.detach())}

    def rest(self, epochs=10):
        if not self.examples:
            raise ValueError("Rest training requires task replay to check against collapse")
        if not 1 <= epochs <= 100:
            raise ValueError("Use 1–100 rest epochs per request")
        # Perturbed states provide a nontrivial rest gradient. Preserve task fitting.
        rng = torch.Generator().manual_seed(self.config.seed + self.updates)
        before = None
        for _ in range(epochs):
            state = torch.rand(self.network.activity.shape, generator=rng)
            self.optimizer.zero_grad(set_to_none=True)
            rest_loss = self.network.tick(state).square().mean()
            if before is None:
                before = float(rest_loss.detach())
            task_loss = 0
            for pulses, labels in self.examples:
                logits, _ = self.evaluate(pulses)
                length = int((labels != 0).sum()) + 1
                task_loss = task_loss + F.cross_entropy(logits[:length, :len(self.vocabulary)+1], labels[:length])
            loss = rest_loss + task_loss / len(self.examples)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.optimizer.step()
            self.network.constrain()
            self.updates += 1
        return {"initial_rest_loss": before, "rest_loss": float(rest_loss.detach()),
                "updates": self.updates, "task_loss": float(task_loss.detach()) / len(self.examples)}

    def description(self):
        return {"config": asdict(self.config), "neurons": self.config.side ** 3,
                "connections": self.network.sources.numel(), "parameters": sum(p.numel() for p in self.network.parameters()),
                "updates": self.updates, "vocabulary": len(self.vocabulary), "replay_pairs": len(self.examples)}
