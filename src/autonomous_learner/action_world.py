"""Evaluator-owned action world and public effect templates; core lives in action_model."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
import hashlib
import random


STATES = tuple(product((0, 1), repeat=3))


def validate_state(values):
    if (not isinstance(values, tuple) or len(values) != 3
            or any(type(value) is not int or value not in (0, 1) for value in values)):
        raise ValueError("A state is an immutable tuple of three integer bits")


@dataclass(frozen=True)
class Catalog:
    objects: tuple[str, ...]
    actions: tuple[str, ...]

    def __post_init__(self):
        for names in (self.objects, self.actions):
            if (not isinstance(names, tuple) or len(names) != 3
                    or any(not isinstance(name, str) or not name for name in names)):
                raise ValueError("Use three immutable object IDs and three action IDs")
        if len(set(self.objects + self.actions)) != 6:
            raise ValueError("Public IDs must be distinct")

    @classmethod
    def generate(cls, public_seed):
        if type(public_seed) is not int or public_seed < 0:
            raise ValueError("Use a nonnegative public naming seed")
        rng = random.Random(public_seed)
        names = []
        while len(names) < 6:
            name = f"n{rng.getrandbits(48):012x}"
            if name not in names:
                names.append(name)
        return cls(tuple(names[:3]), tuple(names[3:]))


@dataclass(frozen=True)
class Observation:
    catalog: Catalog
    values: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.catalog, Catalog):
            raise ValueError("Observation requires a public catalog")
        validate_state(self.values)


@dataclass(frozen=True)
class Transition:
    before: Observation
    action: str
    after: Observation

    def __post_init__(self):
        if not isinstance(self.before, Observation) or not isinstance(self.after, Observation):
            raise ValueError("Transitions require public before/after observations")
        if self.before.catalog != self.after.catalog or self.action not in self.before.catalog.actions:
            raise ValueError("A transition must use one catalog and a legal action")


@dataclass(frozen=True)
class Effect:
    """A hypothetical effect template; true instances stay with the evaluator.

    A guard reads another object's pre-action bit. Failed guards consume an action
    and change nothing. These primitives/structural constraints are supplied priors.
    """
    target: int
    operation: str
    guard: int | None = None
    guard_value: int = 0

    def __post_init__(self):
        if type(self.target) is not int or self.target not in range(3):
            raise ValueError("Target must be a public object position")
        if self.operation not in ("set0", "set1", "toggle"):
            raise ValueError("Unknown supplied effect operation")
        if self.guard is not None and (type(self.guard) is not int
                                       or self.guard not in range(3) or self.guard == self.target):
            raise ValueError("A guard must refer to another object")
        if type(self.guard_value) is not int or self.guard_value not in (0, 1):
            raise ValueError("Guard values are integer bits")
        if self.guard is None and self.guard_value != 0:
            raise ValueError("Unconditional effects use one canonical guard encoding")

    def predict(self, values):
        validate_state(values)
        if self.guard is not None and values[self.guard] != self.guard_value:
            return values
        updated = list(values)
        updated[self.target] = (1 - values[self.target] if self.operation == "toggle"
                                else int(self.operation[-1]))
        return tuple(updated)


def effect_space():
    """45 supplied hypotheses per action, generated without hidden-world access."""
    return tuple(Effect(target, operation, guard, value)
                 for target in range(3) for operation in ("set0", "set1", "toggle")
                 for guard, value in [(None, 0)] + [(g, v) for g in range(3) if g != target for v in (0, 1)])


@lru_cache(maxsize=270)
def _relabeled_table(effect, order):
    relabeled = Effect(order.index(effect.target), effect.operation,
                       None if effect.guard is None else order.index(effect.guard), effect.guard_value)
    return bytes(bit for state in STATES for bit in relabeled.predict(state))


def mechanism_identity(effects):
    """Evaluator-only identity, invariant under object and action renaming.

    Canonicalize full transition behavior, not rule text or names. Splits must use
    this identity before sampling initial states, public names, or goal layouts.
    """
    if (not isinstance(effects, tuple) or len(effects) != 3
            or any(not isinstance(effect, Effect) for effect in effects)):
        raise ValueError("Use one effect per action")
    signatures = []
    for order in permutations(range(3)):
        # Sorting whole per-action tables quotients action names/order as well.
        tables = sorted(_relabeled_table(effect, order) for effect in effects)
        signatures.append(b"".join(tables))
    return hashlib.sha256(min(signatures)).hexdigest()


class ActionWorld:
    """Sequential oracle: observation and paid action, without reset or gold hints.

    Evaluator retains this object. A future learner receives public observations
    and replies, never this instance, Effect objects, or mechanism identities.
    This is an API boundary, not a security sandbox against arbitrary Python code.
    """
    def __init__(self, catalog, effects, initial=(0, 0, 0), budget=12):
        # Validate all inputs before creating mutable execution state.
        mechanism_identity(effects)
        Observation(catalog, initial)
        if type(budget) is not int or budget < 1:
            raise ValueError("Use a positive integer action budget")
        self._catalog, self._effects, self._values = catalog, effects, initial
        self._budget, self._calls = budget, 0

    @property
    def remaining(self):
        return self._budget - self._calls

    def observe(self):
        return Observation(self._catalog, self._values)

    def act(self, action):
        if action not in self._catalog.actions:
            raise ValueError("Unknown public action")
        if self.remaining == 0:
            raise RuntimeError("Action budget exhausted")
        before = self.observe()
        values = self._effects[self._catalog.actions.index(action)].predict(self._values)
        self._values = values
        self._calls += 1  # No-op outcomes are still real interactions.
        return Transition(before, action, self.observe())
