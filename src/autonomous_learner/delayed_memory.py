"""Goal-free memory control over bounded, expiring evidence, with supplied inference."""
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
import random

import torch
from torch import nn

from .types import Fact, Query

RANK_FEATURES = ("parent", "capability", "entity_subject", "incoming_edges",
                 "object_outgoing_edges", "subject_capabilities", "reachable_entities",
                 "proof_usage", "connected_pairs", "age", "in_beliefs", "in_episodes")
GATE_FEATURES = ("episodic_fill", "belief_fill", "visible_fill", "pending_fill",
                 "elapsed_fraction", "write_budget", "belief_coverage", "visible_coverage")


@dataclass(frozen=True)
class PublicTask:
    entities: tuple[str, ...]
    abilities: tuple[str, ...]
    horizon: int = 17
    episodic_capacity: int = 2
    belief_capacity: int = 6
    max_consolidations: int = 6

    def __post_init__(self):
        if not self.entities or not self.abilities or len(set(self.entities)) != len(self.entities) or len(set(self.abilities)) != len(self.abilities):
            raise ValueError("Public vocabulary must be nonempty and unique")
        if self.horizon < 1 or self.episodic_capacity < 1 or min(self.belief_capacity, self.max_consolidations) < 0:
            raise ValueError("Invalid public resource budgets")

    def queries(self):
        # A public task distribution, not the specific delayed question.
        return tuple(Query(entity, "can", ability) for entity in self.entities for ability in self.abilities)


@dataclass(frozen=True)
class Record:
    fact: Fact
    observed_tick: int
    source: str = "synthetic-oracle"

    @property
    def key(self):
        return self.fact, self.source


def visible_proofs(records: tuple[Record, ...] | list[Record], task: PublicTask) -> dict:
    """Infer from current evidence ONLY; return query -> (polarity, evidence mask).

    This is a supplied monotonic proof procedure, consistent with Memory on the
    generated task family. It does not learn logic or support general revision.
    """
    edges, capabilities = {}, {}
    for i, row in enumerate(records):
        fact = row.fact
        if fact.relation == "is_a" and fact.positive:
            edges.setdefault(fact.subject, []).append((fact.object, i))
        elif fact.relation == "can":
            capabilities.setdefault((fact.subject, fact.object), []).append((fact.positive, i))
    answers = {}
    for entity in task.entities:
        paths, queue = {entity: 0}, deque([entity])
        while queue:
            subject = queue.popleft()
            for parent, i in edges.get(subject, []):
                if parent not in paths:
                    paths[parent] = paths[subject] | (1 << i)
                    queue.append(parent)
        for ability in task.abilities:
            proofs = {}
            for subject, path in paths.items():
                for positive, i in capabilities.get((subject, ability), []):
                    proofs.setdefault(positive, path | (1 << i))
            if len(proofs) == 1:
                answers[Query(entity, "can", ability)] = next(iter(proofs.items()))
    return answers


class DelayedMemory:
    def __init__(self, task: PublicTask):
        self.task = task
        self.episodes: list[Record] = []
        self.beliefs: list[Record] = []
        self.tick = -1
        self.consolidations = 0

    def advance(self, fact: Fact | None):
        if self.tick + 1 >= self.task.horizon:
            raise ValueError("Public tick budget exhausted")
        self.tick += 1
        # Expiration applies on quiet ticks too. No archive remains accessible.
        self.episodes = [row for row in self.episodes
                         if self.tick - row.observed_tick < self.task.episodic_capacity]
        if fact is not None:
            self.episodes = [row for row in self.episodes if row.fact != fact]
            self.episodes.append(Record(fact, self.tick))
            self.episodes = self.episodes[-self.task.episodic_capacity:]

    def visible(self) -> tuple[Record, ...]:
        unique = {row.key: row for row in self.beliefs}
        unique.update((row.key, row) for row in self.episodes)
        return tuple(sorted(unique.values(), key=lambda row: row.observed_tick))

    def can_consolidate(self) -> bool:
        existing = {row.key for row in self.beliefs}
        return (self.task.belief_capacity > 0 and self.consolidations < self.task.max_consolidations
                and any(row.key not in existing for row in self.episodes))

    def consolidate(self, indices: list[int]):
        if not self.can_consolidate():
            raise ValueError("Consolidation is masked")
        visible = self.visible()
        if len(indices) > self.task.belief_capacity or len(set(indices)) != len(indices):
            raise ValueError("Invalid selected subset")
        if any(i < 0 or i >= len(visible) for i in indices):
            raise ValueError("Selected evidence is not accessible")
        self.beliefs = [visible[i] for i in sorted(indices)]
        self.consolidations += 1

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "task": asdict(self.task), "tick": self.tick, "consolidations": self.consolidations,
                "episodes": [asdict(r) for r in self.episodes], "beliefs": [asdict(r) for r in self.beliefs]}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported delayed-memory state version")
        specification = data["task"]
        specification["entities"] = tuple(specification["entities"])
        specification["abilities"] = tuple(specification["abilities"])
        memory = cls(PublicTask(**specification))
        memory.tick, memory.consolidations = data["tick"], data["consolidations"]
        memory.episodes = [Record(Fact(**r["fact"]), r["observed_tick"], r["source"]) for r in data["episodes"]]
        memory.beliefs = [Record(Fact(**r["fact"]), r["observed_tick"], r["source"]) for r in data["beliefs"]]
        if (not -1 <= memory.tick < memory.task.horizon
                or not 0 <= memory.consolidations <= memory.task.max_consolidations
                or len(memory.episodes) > memory.task.episodic_capacity
                or len(memory.beliefs) > memory.task.belief_capacity
                or any(not 0 <= r.observed_tick <= memory.tick for r in memory.visible())
                or any(memory.tick - r.observed_tick >= memory.task.episodic_capacity for r in memory.episodes)):
            raise ValueError("Invalid persisted resource state")
        if any(len({r.key for r in rows}) != len(rows) for rows in (memory.beliefs, memory.episodes)):
            raise ValueError("Duplicate persisted evidence")
        return memory


def encode(memory: DelayedMemory) -> tuple[torch.Tensor, torch.Tensor]:
    records, task = memory.visible(), memory.task
    proofs = visible_proofs(records, task)
    belief_keys = {row.key for row in memory.beliefs}
    episode_keys = {row.key for row in memory.episodes}
    nqueries = len(task.queries())
    incoming, outgoing, capabilities = {}, {}, {}
    for row in records:
        fact = row.fact
        if fact.relation == "is_a" and fact.positive:
            incoming[fact.object] = incoming.get(fact.object, 0) + 1
            outgoing[fact.subject] = outgoing.get(fact.subject, 0) + 1
        elif fact.relation == "can":
            capabilities[fact.subject] = capabilities.get(fact.subject, 0) + 1
    # Reachability uses only the same bounded records, never the hidden stream.
    entity_reach = {}
    for entity in task.entities:
        reachable = {entity}
        for _ in records:
            reachable.update(r.fact.object for r in records if r.fact.relation == "is_a"
                             and r.fact.positive and r.fact.subject in reachable)
        for subject in reachable:
            entity_reach[subject] = entity_reach.get(subject, 0) + 1
    rows = []
    for i, row in enumerate(records):
        fact = row.fact
        links = sum((r.fact.relation == "is_a" and r.fact.object == fact.subject)
                    or (fact.relation == "is_a" and fact.object == r.fact.subject) for r in records)
        rows.append([fact.relation == "is_a", fact.relation == "can", fact.subject in task.entities,
                     incoming.get(fact.subject, 0) / len(task.entities),
                     outgoing.get(fact.object, 0) / len(task.entities),
                     capabilities.get(fact.subject, 0) / len(task.abilities),
                     entity_reach.get(fact.subject, 0) / len(task.entities),
                     sum(bool(mask & (1 << i)) for _, mask in proofs.values()) / nqueries,
                     links / max(1, task.episodic_capacity + task.belief_capacity),
                     (memory.tick - row.observed_tick) / task.horizon,
                     row.key in belief_keys, row.key in episode_keys])
    rank = torch.tensor(rows, dtype=torch.float32).reshape(-1, len(RANK_FEATURES))
    gate = torch.tensor([len(memory.episodes) / task.episodic_capacity,
                         len(memory.beliefs) / max(1, task.belief_capacity),
                         len(records) / max(1, task.belief_capacity + task.episodic_capacity),
                         len(episode_keys - belief_keys) / task.episodic_capacity,
                         (memory.tick + 1) / task.horizon,
                         (task.max_consolidations - memory.consolidations) / max(1, task.max_consolidations),
                         len(visible_proofs(memory.beliefs, task)) / nqueries, len(proofs) / nqueries], dtype=torch.float32)
    return rank, gate


def structural_value(records: tuple[Record, ...] | list[Record], task: PublicTask) -> tuple:
    links = sum(a.fact.relation == "is_a" and a.fact.object == b.fact.subject
                for a in records for b in records)
    return (len(visible_proofs(records, task)), links,
            sum(r.fact.relation == "is_a" for r in records), len(records))


def graph_proposal(memory: DelayedMemory) -> list[int]:
    visible, capacity = memory.visible(), memory.task.belief_capacity
    if len(visible) <= capacity:
        return list(range(len(visible)))
    return list(max(combinations(range(len(visible)), capacity),
                    key=lambda subset: (structural_value([visible[i] for i in subset], memory.task),
                                        sum(visible[i].observed_tick for i in subset))))


class MemoryController(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(len(RANK_FEATURES), 32), nn.Tanh())
        self.ranker = nn.Linear(32, 1)
        self.gate = nn.Linear(64 + len(GATE_FEATURES), 2)  # WAIT, CONSOLIDATE
        nn.init.zeros_(self.ranker.weight)
        nn.init.zeros_(self.ranker.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, rank, gate, mask=None):
        if rank.ndim == 2:
            rank, gate = rank[None], gate[None]
            single = True
        else:
            single = False
        if mask is None:
            mask = torch.ones(rank.shape[:2], dtype=torch.bool, device=rank.device)
        elif single:
            mask = mask[None]
        encoded = self.encoder(rank)
        means = (encoded * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = encoded.masked_fill(~mask[..., None], -1e6).amax(1)
        scores = self.ranker(encoded).squeeze(-1).masked_fill(~mask, -1e6)
        logits = self.gate(torch.cat([means, maximum, gate], dim=-1))
        return (scores[0], logits[0]) if single else (scores, logits)


def choose(memory: DelayedMemory, policy, rng: random.Random | None = None, sample=False):
    """Return selected visible indices or None, plus differentiable training terms."""
    log_probs, entropies = [], []
    if not memory.can_consolidate():
        return None, log_probs, entropies
    visible = memory.visible()
    if isinstance(policy, MemoryController):
        scores, gate_logits = policy(*encode(memory))
        distribution = torch.distributions.Categorical(logits=gate_logits)
        action = distribution.sample() if sample else gate_logits.argmax()
        if sample:
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
        if int(action) == 0:
            return None, log_probs, entropies
        count = min(len(visible), memory.task.belief_capacity)
        if not sample:
            indices = torch.argsort(scores, descending=True, stable=True)[:count].tolist()
        else:
            # Sample removals rather than six ordered retain decisions.
            remaining = list(range(len(visible)))
            for _ in range(len(visible) - count):
                distribution = torch.distributions.Categorical(logits=-scores[remaining])
                index = distribution.sample()
                log_probs.append(distribution.log_prob(index))
                entropies.append(distribution.entropy())
                remaining.pop(int(index))
            indices = remaining
    elif policy == "random":
        assert rng is not None
        if rng.random() < 0.5:
            return None, log_probs, entropies
        indices = rng.sample(range(len(visible)), min(len(visible), memory.task.belief_capacity))
    elif policy in {"fifo", "graph_scheduled"}:
        # Same known-horizon schedule, same write ceiling; no future stream access.
        now = (memory.tick + 1) * memory.task.max_consolidations // memory.task.horizon
        before = memory.tick * memory.task.max_consolidations // memory.task.horizon
        if now <= before:
            return None, log_probs, entropies
        indices = (list(range(max(0, len(visible) - memory.task.belief_capacity), len(visible)))
                   if policy == "fifo" else graph_proposal(memory))
    elif policy == "graph":
        indices = graph_proposal(memory)
        proposal = [visible[i] for i in indices]
        if structural_value(proposal, memory.task) <= structural_value(memory.beliefs, memory.task):
            return None, log_probs, entropies
    else:
        raise ValueError(f"Unknown memory policy: {policy}")
    return indices, log_probs, entropies
