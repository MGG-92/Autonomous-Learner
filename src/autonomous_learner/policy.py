"""Policy features are derived only from the agent's accessible state."""
from dataclasses import dataclass
import random

import torch
from torch import nn

from .memory import Memory
from .types import Query

FEATURES = ("ask", "goal_subject", "reachable", "goal_property", "parent_query",
            "already_asked", "path_depth", "answer", "answer_available", "abstain", "wait", "budget")


@dataclass(frozen=True)
class Action:
    kind: str
    query: Query | None = None


def candidates(catalog: tuple[Query, ...]) -> list[Action]:
    return [Action("ASK", q) for q in catalog] + [Action("ANSWER"), Action("ABSTAIN"), Action("WAIT")]


def encode(actions: list[Action], memory: Memory, goal: Query, asked: set[Query], remaining: float) -> torch.Tensor:
    reachable = memory.reachable(goal.subject)
    available = memory.answer(goal).value is not None
    rows = []
    for action in actions:
        q = action.query
        rows.append([
            action.kind == "ASK", q is not None and q.subject == goal.subject,
            q is not None and q.subject in reachable,
            q is not None and q.relation == goal.relation and q.object == goal.object,
            q is not None and q.relation == "is_a", q in asked if q else False,
            len(reachable.get(q.subject, ())) / 3 if q else 0,
            action.kind == "ANSWER", action.kind == "ANSWER" and available,
            action.kind == "ABSTAIN", action.kind == "WAIT", remaining,
        ])
    return torch.tensor(rows, dtype=torch.float32)


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class Heuristic:
    def choose(self, features: torch.Tensor) -> int:
        # Only observed reachability, never hidden truth or class roles.
        score = (features[:, 2] * 3 + features[:, 4] * 2 + features[:, 3]
                 + features[:, 6] - features[:, 5] * 100
                 + features[:, 8] * 100 - features[:, 7] * 10
                 - features[:, 9] * 10 - features[:, 10] * 10)
        return int(score.argmax())


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, features: torch.Tensor) -> int:
        return self.rng.randrange(len(features))
