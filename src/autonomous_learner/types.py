from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True, order=True)
class Query:
    subject: str
    relation: str
    object: str = ""

    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}".strip()


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    positive: bool = True


@dataclass(frozen=True)
class Observation:
    query: Query
    facts: tuple[Fact, ...]
    source: str


@dataclass(frozen=True)
class Answer:
    status: str  # SUPPORTED, REFUTED, UNKNOWN, CONFLICT
    evidence_ids: tuple[int, ...] = ()

    @property
    def value(self) -> bool | None:
        return {"SUPPORTED": True, "REFUTED": False}.get(self.status)

    def to_dict(self) -> dict:
        return asdict(self)


class Teacher(Protocol):
    def ask(self, query: Query) -> Observation: ...
