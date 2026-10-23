"""Language-free finite program induction. Search/inference are supplied, not learned."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import random


@dataclass(frozen=True)
class Program:
    op: str
    args: tuple = ()

    def text(self):
        return self.op if not self.args else f"{self.op}({','.join(arg.text() for arg in self.args)})"

    @property
    def nodes(self):
        return 1 + sum(arg.nodes for arg in self.args)


def execute(program: Program, x: int, y: int, modulus: int) -> int:
    if program.op in {"x", "y", "0", "1"}:
        return {"x": x, "y": y, "0": 0, "1": 1}[program.op]
    a, b = (execute(arg, x, y, modulus) for arg in program.args)
    return combine(program.op, a, b, modulus)


def combine(op, a, b, modulus):
    if op == "add_mod":
        return (a + b) % modulus
    if op == "sub_mod":
        return (a - b) % modulus
    if op == "mul_mod":
        return a * b % modulus
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    raise ValueError("Unknown supplied primitive")


@lru_cache(maxsize=1)
def reference_signatures():
    """Named examples/software fixtures are development-only, regardless of hash."""
    pairs = tuple((x, y) for x in range(5) for y in range(5))
    signatures = {tuple(x for x, y in pairs), tuple(y for x, y in pairs),
                  (0,) * 25, (1,) * 25}
    signatures.update(tuple(combine(op, x, y, 5) for x, y in pairs)
                      for op in ("add_mod", "sub_mod", "mul_mod", "min", "max"))
    return frozenset(signatures)


@dataclass(frozen=True)
class Hypothesis:
    program: Program
    values: tuple[int, ...]

    @property
    def identity(self):
        return hashlib.sha256(bytes(self.values)).hexdigest()

    @property
    def partition(self):
        if self.values in reference_signatures():
            return "development"
        bucket = int(self.identity[:8], 16) % 100
        return "development" if bucket < 70 else "validation" if bucket < 85 else "final"


@dataclass(frozen=True)
class RuleSpace:
    modulus: int
    inputs: tuple[tuple[int, int], ...]
    hypotheses: tuple[Hypothesis, ...]


@lru_cache(maxsize=1)
def build_space() -> RuleSpace:
    """Enumerate depth <=2 expressions; quotient by full 5x5 behavior.

    These are *hypothetical* outputs generated from public primitives, not oracle
    observations. Equivalent expressions have one prior vote and cannot cross splits.
    """
    modulus = 5
    inputs = tuple((x, y) for x in range(modulus) for y in range(modulus))
    bank = {}
    for op in ("x", "y", "0", "1"):
        program = Program(op)
        values = tuple(execute(program, *pair, modulus) for pair in inputs)
        bank[values] = program
    for _ in range(2):
        previous = list(bank.items())
        for op in ("add_mod", "sub_mod", "mul_mod", "min", "max"):
            for left_values, left in previous:
                for right_values, right in previous:
                    if op != "sub_mod" and left.text() > right.text():
                        continue
                    values = tuple(combine(op, a, b, modulus) for a, b in zip(left_values, right_values))
                    program = Program(op, (left, right))
                    old = bank.get(values)
                    if old is None or (program.nodes, program.text()) < (old.nodes, old.text()):
                        bank[values] = program
    hypotheses = tuple(sorted((Hypothesis(program, values) for values, program in bank.items()),
                              key=lambda hypothesis: hypothesis.identity))
    return RuleSpace(modulus, inputs, hypotheses)


class RuleLearner:
    """A uniform version space updated only from observed input/output pairs.

    This is evidence-driven belief learning with fixed parameters. It does not learn
    primitives, the inference algorithm, or a neural investigation policy.
    """
    def __init__(self, space: RuleSpace, legal_queries: tuple[tuple[int, int], ...]):
        if not legal_queries or len(set(legal_queries)) != len(legal_queries) or not set(legal_queries) <= set(space.inputs):
            raise ValueError("Use unique public-domain query inputs")
        self.space, self.legal_queries = space, legal_queries
        self.candidates = tuple(range(len(space.hypotheses)))
        self.observations = {}
        self.asked = set()

    def observe(self, pair, value):
        if pair not in self.legal_queries or pair in self.asked:
            raise ValueError("Observation must answer a new legal query")
        if type(value) is not int or not 0 <= value < self.space.modulus:
            raise ValueError("Observation is outside the public output domain")
        index = self.space.inputs.index(pair)
        self.candidates = tuple(i for i in self.candidates if self.space.hypotheses[i].values[index] == value)
        self.observations[pair] = value
        self.asked.add(pair)

    def distribution(self, pair):
        if pair not in self.space.inputs:
            raise ValueError("Prediction input is outside the public domain")
        if not self.candidates:
            return None  # Explicit model-class failure, never invent certainty.
        index = self.space.inputs.index(pair)
        counts = Counter(self.space.hypotheses[i].values[index] for i in self.candidates)
        return tuple(counts[value] / len(self.candidates) for value in range(self.space.modulus))

    def choose(self, strategy: str, rng: random.Random):
        remaining = [pair for pair in self.legal_queries if pair not in self.asked]
        if not remaining or not self.candidates:
            return None
        if strategy == "random":
            return rng.choice(remaining)
        if strategy == "fixed":
            return min(remaining)
        if strategy != "information_gain":
            raise ValueError("Unknown query strategy")
        # Deterministic, noiseless hypotheses: predictive entropy equals expected
        # information gain about the finite function identity. Ties use a public RNG.
        entropies = []
        for pair in remaining:
            probabilities = self.distribution(pair)
            entropies.append(-sum(p * math.log(p) for p in probabilities if p))
        best = max(entropies)
        ties = [pair for pair, entropy in zip(remaining, entropies) if abs(entropy - best) < 1e-12]
        return rng.choice(ties)

    def reset_model(self):
        self.candidates = tuple(range(len(self.space.hypotheses)))
        self.observations.clear()  # Raw observations must not reconstruct the reset model.

    def message(self, pair):
        probabilities = self.distribution(pair)
        return {"input": list(pair), "status": "MODEL_CLASS_FAILURE" if probabilities is None else "PREDICTION",
                "probabilities": probabilities, "remaining_hypotheses": len(self.candidates),
                "uncertainty_scope": "uniform finite supplied rule space; calibration not established",
                "representative": self.space.hypotheses[self.candidates[0]].program.text()
                if len(self.candidates) == 1 else None}
