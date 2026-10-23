"""The oracle owns hidden facts. Its public catalog contains no answers."""
import random
from dataclasses import dataclass

from .types import Fact, Observation, Query


@dataclass
class World:
    seed: int
    entities: tuple[str, ...]
    classes: tuple[str, ...]
    abilities: tuple[str, ...]
    facts: tuple[Fact, ...]

    @classmethod
    def generate(cls, seed: int, branches: int = 3) -> "World":
        rng = random.Random(seed)
        # A fresh shared vocabulary prevents stable lexical role cues.
        names = [f"n{rng.getrandbits(48):012x}" for _ in range(branches * 3 + 3)]
        entities = tuple(names[:branches])
        classes = tuple(names[branches:branches * 3])
        abilities = tuple(names[branches * 3:])
        facts = []
        for i, entity in enumerate(entities):
            leaf, root = classes[2 * i:2 * i + 2]
            facts.extend([Fact(entity, "is_a", leaf), Fact(leaf, "is_a", root)])
            for ability in abilities:
                facts.append(Fact(root, "can", ability, bool(rng.getrandbits(1))))
        rng.shuffle(facts)
        return cls(seed, entities, classes, abilities, tuple(facts))

    def catalog(self) -> tuple[Query, ...]:
        queries = [Query(s, "is_a") for s in self.entities + self.classes]
        queries += [Query(s, "can", a) for s in self.classes for a in self.abilities]
        random.Random(self.seed + 19).shuffle(queries)
        return tuple(queries)

    def goal(self) -> Query:
        rng = random.Random(self.seed + 7)
        return Query(rng.choice(self.entities), "can", rng.choice(self.abilities))

    def truth(self, goal: Query) -> bool:
        node = goal.subject
        visited = set()
        while node not in visited:
            visited.add(node)
            for fact in self.facts:
                if (fact.subject, fact.relation, fact.object) == (node, "can", goal.object):
                    return fact.positive
            parent = next((f.object for f in self.facts if f.subject == node and f.relation == "is_a"), None)
            if parent is None:
                break
            node = parent
        raise ValueError("Generated goal has no truth value")


class SyntheticTeacher:
    def __init__(self, world: World):
        self._facts = world.facts
        self._legal = set(world.catalog())
        self.calls = 0

    def ask(self, query: Query) -> Observation:
        if query not in self._legal:
            raise ValueError(f"Illegal query: {query}")
        self.calls += 1
        facts = tuple(f for f in self._facts if f.subject == query.subject
                      and f.relation == query.relation
                      and (not query.object or f.object == query.object))
        return Observation(query, facts, "synthetic-oracle")
