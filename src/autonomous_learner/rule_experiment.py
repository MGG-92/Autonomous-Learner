"""Evaluation boundary for internal rule discovery; oracle truth never selects queries."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

from .rule_induction import RuleLearner, build_space


@dataclass(frozen=True)
class RuleConfig:
    worlds: int = 100
    queries: int = 4
    heldout_inputs: int = 5
    query_seeds: list[int] = field(default_factory=lambda: [7, 42, 97])
    selection_seed: int = 17
    validation_layout_start: int = 3_200_000
    final_layout_start: int = 3_300_000

    def validate(self):
        counts = (self.worlds, self.queries, self.heldout_inputs, self.selection_seed,
                  self.validation_layout_start, self.final_layout_start)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Counts and seeds must be nonnegative integers")
        if self.worlds < 1 or not 1 <= self.heldout_inputs < 25 or not 1 <= self.queries <= 25 - self.heldout_inputs:
            raise ValueError("Invalid world, query, or held-out input count")
        if (not isinstance(self.query_seeds, list) or not self.query_seeds
                or any(type(seed) is not int or seed < 0 for seed in self.query_seeds)
                or len(set(self.query_seeds)) != len(self.query_seeds)):
            raise ValueError("Use unique nonnegative query seeds")
        a, b = self.validation_layout_start, self.final_layout_start
        if max(a, b) < min(a + self.worlds, b + self.worlds):
            raise ValueError("Validation and final layout seeds must be disjoint")


def investigate(learner: RuleLearner, oracle, strategy, seed, budget):
    """Only this observation boundary contacts the oracle; no hidden program input."""
    rng, trace = random.Random(seed), []
    for _ in range(budget):
        if len(learner.candidates) <= 1:
            break
        pair = learner.choose(strategy, rng)
        if pair is None:
            break
        before = len(learner.candidates)
        observed = oracle(pair)
        learner.observe(pair, observed)
        trace.append({"query": list(pair), "observed": observed, "hypotheses_before": before,
                      "hypotheses_after": len(learner.candidates)})
    return trace


def scores(predictions, truth):
    # Predictions are committed before the evaluator reads reserved targets.
    if len(predictions) != len(truth) or not predictions:
        raise ValueError("Need matched nonempty predictions and evaluation targets")
    correct = brier = log_loss = 0.0
    for probabilities, value in zip(predictions, truth):
        if probabilities is None:
            raise ValueError("In-class evaluation unexpectedly exhausted its model space")
        correct += max(range(len(probabilities)), key=probabilities.__getitem__) == value
        brier += sum((p - (i == value)) ** 2 for i, p in enumerate(probabilities))
        if probabilities[value] <= 0:
            raise ValueError("A true in-class rule was incorrectly removed")
        log_loss -= math.log(probabilities[value])
    n = len(predictions)
    return {"accuracy": correct / n, "brier": brier / n, "log_loss": log_loss / n}


def run(config: RuleConfig, output: Path, phase="validation"):
    config.validate()
    if phase not in {"validation", "final"}:
        raise ValueError("Choose validation or final")
    if output.exists():
        raise FileExistsError("Preserve earlier results; choose a fresh output directory")
    space = build_space()
    worlds = [h for h in space.hypotheses if h.partition == phase]
    if config.worlds > len(worlds):
        raise ValueError("Not enough distinct semantic rules in this partition")
    worlds = random.Random(config.selection_seed).sample(worlds, config.worlds)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    output.mkdir(parents=True)
    began, rows, example = time.perf_counter(), [], None
    start = config.final_layout_start if phase == "final" else config.validation_layout_start
    for index, hidden in enumerate(worlds):
        # Layout depends on an independent public seed, never the hidden rule identity.
        heldout = tuple(random.Random(start + index).sample(space.inputs, config.heldout_inputs))
        legal = tuple(pair for pair in space.inputs if pair not in heldout)
        oracle = lambda pair: hidden.values[space.inputs.index(pair)]
        for seed in config.query_seeds:
            for strategy in ("information_gain", "random", "fixed"):
                learner = RuleLearner(space, legal)
                trace = investigate(learner, oracle, strategy, (seed << 32) ^ (start + index), config.queries)
                predictions = [learner.distribution(pair) for pair in heldout]
                messages = [learner.message(pair) for pair in heldout] if example is None else None
                truth = [oracle(pair) for pair in heldout]  # Evaluator only, after investigation.
                metrics = scores(predictions, truth)
                if example is None:
                    example = {"rule_id": hidden.identity, "layout_seed": start + index, "query_seed": seed,
                               "trace": trace, "core_messages_before_evaluation": messages,
                               "evaluator_targets": truth, "note": "Structured model output; no LLM explanation"}
                row = {"rule_id": hidden.identity, "layout_seed": start + index, "query_seed": seed,
                       "policy": strategy, "queries": len(trace), "remaining_hypotheses": len(learner.candidates),
                       "identified_full_function": len(learner.candidates) == 1, **metrics}
                rows.append(row)
                if strategy == "information_gain":
                    # Surgical intervention: remove learned belief state AND raw observations.
                    learner.reset_model()
                    reset = scores([learner.distribution(pair) for pair in heldout], truth)
                    rows.append({**row, "policy": "model_reset", "remaining_hypotheses": len(learner.candidates),
                                 "identified_full_function": False, **reset})
                    # A no-memory control discards the same replies; equivalent to reset
                    # by construction, not independent evidence from another algorithm.
                    prior = RuleLearner(space, legal)
                    rows.append({**row, "policy": "no_memory", "remaining_hypotheses": len(prior.candidates),
                                 "identified_full_function": False,
                                 **scores([prior.distribution(pair) for pair in heldout], truth)})
    policies = ("information_gain", "random", "fixed", "model_reset", "no_memory")
    metrics = ("accuracy", "brier", "log_loss", "queries", "identified_full_function")
    summary = {}
    for policy in policies:
        group = [row for row in rows if row["policy"] == policy]
        summary[policy] = {metric: sum(row[metric] for row in group) / len(group) for metric in metrics}
    # Average query seeds within each rule before bootstrap: seeds are not training repeats.
    comparisons = {}
    for baseline in ("random", "fixed", "model_reset"):
        differences = []
        for hidden in worlds:
            group = [row for row in rows if row["rule_id"] == hidden.identity]
            active = [row["brier"] for row in group if row["policy"] == "information_gain"]
            other = [row["brier"] for row in group if row["policy"] == baseline]
            differences.append(sum(other) / len(other) - sum(active) / len(active))
        rng = random.Random(197)
        bootstrap = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
        comparisons[baseline] = {"mean_brier_reduction": sum(differences) / len(differences),
                                 "rule_bootstrap95": [bootstrap[50], bootstrap[1950]], "samples": 2000}
    report = {"schema_version": 1, "protocol": "docs/experiments/internal-rule-discovery.md", "phase": phase,
              "config": asdict(config), "hypothesis_count": len(space.hypotheses),
              "partition_counts": {name: sum(h.partition == name for h in space.hypotheses)
                                   for name in ("development", "validation", "final")},
              "summary": summary, "paired_comparisons": comparisons, "per_rule": rows, "example": example,
              "baseline_roles": {"heuristic": "information_gain", "random_policy": "random",
                                 "untrained_policy": "fixed (no trainable policy exists in this baseline)",
                                 "no_memory": "same active observations discarded; equivalent predictions to model reset"},
              "environment": {"git_commit": revision, "git_dirty": dirty, "python": platform.python_version(),
                              "elapsed_seconds": time.perf_counter() - began},
              "limitations": ["Supplied primitives, bounded grammar, uniform prior, filtering, and query heuristic",
                              "Learning updates beliefs; no parameter optimization or learned reasoning strategy",
                              "Finite 5x5 domain and in-class rules; no unbounded arithmetic or causal discovery",
                              "Semantic function splits, not held-out primitive/family generalization",
                              "Named primitive/max/projection reference functions are development-only",
                              "Query seeds are tie/randomness repeats, not optimizer seeds",
                              "Bootstrap averages seeds within rules; exploratory unadjusted intervals",
                              "No-memory and model-reset predictions are equivalent by construction",
                              "Query ceilings matched; information-gain search uses more computation than simple controls",
                              "Posterior probabilities are model-relative; calibration outside this class is untested"]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
