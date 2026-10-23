"""Evaluator-only semantic population and development sampler, never learner input."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
import random

from .action_world import ActionWorld, Catalog, Effect, STATES, effect_space, mechanism_identity


@dataclass(frozen=True)
class Mechanism:
    identity: str
    effects: tuple[Effect, ...]


@lru_cache(maxsize=1)
def reference_identities():
    return frozenset(mechanism_identity(effects) for effects in (
        (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1)),
        (Effect(0, "set0"),) * 3))


def partition(identity):
    if identity in reference_identities():
        return "development"
    bucket = int(identity[:8], 16) % 100
    return "development" if bucket < 70 else "validation" if bucket < 85 else "final"


@lru_cache(maxsize=1)
def mechanism_bank():
    # Action permutations cannot add identities, so combinations suffice. Each
    # canonical identity gets one sampling vote regardless of orbit multiplicity.
    unique = {}
    for effects in combinations_with_replacement(effect_space(), 3):
        identity = mechanism_identity(effects)
        unique.setdefault(identity, effects)
    return tuple(Mechanism(identity, unique[identity]) for identity in sorted(unique))


def sample_development(count, seed):
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Use a positive world count and nonnegative selection seed")
    population = tuple(m for m in mechanism_bank() if partition(m.identity) == "development")
    if count > len(population):
        raise ValueError("Not enough distinct development mechanisms")
    return tuple(random.Random(seed).sample(population, count))


def instantiate_development(mechanism, naming_seed, permutation_seed, initial_seed, budget=12):
    if (mechanism.identity != mechanism_identity(mechanism.effects)
            or partition(mechanism.identity) != "development"):
        raise ValueError("Only verified development mechanisms can be executed here")
    if any(type(seed) is not int or seed < 0 for seed in (naming_seed, permutation_seed, initial_seed)):
        raise ValueError("Use independent nonnegative layout seeds")
    rng = random.Random(permutation_seed)
    objects, actions = list(range(3)), list(range(3))
    rng.shuffle(objects)
    rng.shuffle(actions)
    relabeled = tuple(Effect(objects.index(e.target), e.operation,
                             None if e.guard is None else objects.index(e.guard), e.guard_value)
                      for e in mechanism.effects)
    effects = tuple(relabeled[index] for index in actions)
    initial = random.Random(initial_seed).choice(STATES)
    return ActionWorld(Catalog.generate(naming_seed), effects, initial, budget)
