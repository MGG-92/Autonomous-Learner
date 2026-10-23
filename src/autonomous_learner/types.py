from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True, order=True)
class Query:
    subject: str
    relation: str
    object: str = ""

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value.strip() for value in (self.subject, self.relation))
                or not isinstance(self.object, str)):
            raise ValueError("Query names and relation must be strings with a nonempty subject and relation")

    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}".strip()


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    positive: bool = True

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.subject, self.relation, self.object)):
            raise ValueError("Fact names and relation must be nonempty strings")
        if type(self.positive) is not bool:
            raise ValueError("Fact polarity must be a boolean")


@dataclass(frozen=True)
class Observation:
    query: Query
    facts: tuple[Fact, ...]
    source: str

    def __post_init__(self):
        if (not isinstance(self.query, Query) or not isinstance(self.facts, tuple)
                or any(not isinstance(fact, Fact) for fact in self.facts)
                or not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("Observation requires a query, tuple of facts, and nonempty source")


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
