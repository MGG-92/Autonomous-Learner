"""Prediction-only development evaluator. Gold panels never enter investigation."""
from dataclasses import asdict, dataclass, field
from itertools import product
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

from .action_investigation import investigate
from .action_model import TransitionModel
from .action_sampling import instantiate_development, sample_development
from .action_world import STATES


@dataclass(frozen=True)
class PredictionConfig:
    worlds: int = 80
    exploration_actions: int = 12
    two_action_probes: int = 16
    policy_seeds: list[int] = field(default_factory=lambda: [7, 42, 97])
    selection_seed: int = 611
    exploration_layout_start: int = 5_000_000
    naming_start: int = 5_100_000
    permutation_start: int = 5_150_000
    panel_start: int = 5_200_000

    def validate(self):
        if any(type(v) is not int or v < 1 for v in (self.worlds, self.exploration_actions, self.two_action_probes)):
            raise ValueError("Positive integer world/action/probe counts required")
        if self.two_action_probes > 72:
            raise ValueError("Only 72 distinct two-action probes exist")
        starts = (self.exploration_layout_start, self.naming_start, self.permutation_start, self.panel_start)
        if (not isinstance(self.policy_seeds, list) or not self.policy_seeds
                or len(set(self.policy_seeds)) != len(self.policy_seeds)
                or any(type(v) is not int or v < 0 for v in (*starts, self.selection_seed, *self.policy_seeds))):
            raise ValueError("Use nonnegative seeds and unique policy seeds")
        if any(max(a, b) < min(a + self.worlds, b + self.worlds)
               for i, a in enumerate(starts) for b in starts[i + 1:]):
            raise ValueError("Purpose-specific layout seed ranges must be disjoint")


def panels(catalog, seed, two_count):
    one = tuple((state, (action,)) for state in STATES for action in catalog.actions)
    two = tuple((state, actions) for state in STATES for actions in product(catalog.actions, repeat=2))
    return one + tuple(random.Random(seed).sample(two, two_count))


def experienced_probes(trace):
    seen = {(t.before.values, (t.action,)) for t in trace}
    seen.update((first.before.values, (first.action, second.action))
                for first, second in zip(trace, trace[1:]) if first.after == second.before)
    return seen


def commit_predictions(model, panel):
    return tuple(model.distribution(state, actions) for state, actions in panel)


def _targets(effects, catalog, panel):
    # Called only after all policies/seeds have committed all predictions for this world.
    paths = {}
    for probe in panel:
        state, actions = probe
        path = []
        for action in actions:
            state = effects[catalog.actions.index(action)].predict(state)
            path.append(state)
        paths[probe] = tuple(path)
    return paths


def score(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Match predictions and targets")
    if not predictions:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None}
    accuracy = brier = log_loss = 0.0
    for probabilities, target in zip(predictions, targets):
        if (probabilities is None or len(probabilities) != 8
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-9)):
            raise ValueError("Expected a proper in-class joint distribution")
        index = STATES.index(target)
        if probabilities[index] <= 0:
            raise ValueError("A true in-class outcome lost all probability")
        accuracy += max(range(8), key=probabilities.__getitem__) == index
        brier += sum((p - (i == index)) ** 2 for i, p in enumerate(probabilities))
        log_loss -= math.log(probabilities[index])
    n = len(predictions)
    return {"n": n, "accuracy": accuracy / n, "brier": brier / n, "log_loss": log_loss / n}


def aggregate(rows):
    eligible = [row for row in rows if row["n"]]
    return {"world_seed_runs": len(rows), "nonempty_runs": len(eligible),
            "mean_probe_count": sum(row["n"] for row in rows) / len(rows),
            **{metric: sum(row[metric] for row in eligible) / len(eligible) if eligible else None
               for metric in ("accuracy", "brier", "log_loss")}}


def comparisons(rows, identities):
    results = {}
    for panel in ("one_step", "two_step"):
        for scope in ("full", "common_unseen"):
            group = [row for row in rows if row["panel"] == panel and row["scope"] == scope and row["n"]]
            for baseline in ("random", "fixed", "model_reset"):
                differences = []
                for identity in identities:
                    world = [row for row in group if row["mechanism_id"] == identity]
                    active = [row["brier"] for row in world if row["policy"] == "information_gain"]
                    other = [row["brier"] for row in world if row["policy"] == baseline]
                    if active and other:
                        differences.append(sum(other) / len(other) - sum(active) / len(active))
                rng = random.Random(197)
                bootstrap = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                                   for _ in range(2000)) if differences else []
                results[f"{panel}/{scope}/{baseline}"] = {
                    "worlds": len(differences), "mean_brier_reduction": sum(differences) / len(differences) if differences else None,
                    "world_bootstrap95": [bootstrap[50], bootstrap[1950]] if bootstrap else None,
                    "per_world": differences, "bootstrap_samples": 2000}
    return results


def run(config: PredictionConfig, output: Path, progress=None):
    config.validate()
    if output.exists():
        raise FileExistsError("Preserve earlier runs; use a fresh output directory")
    began = time.perf_counter()
    mechanisms = sample_development(config.worlds, config.selection_seed)
    population_seconds = time.perf_counter() - began
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    output.mkdir(parents=True)
    rows, costs, coverage = [], [], []
    with (output / "raw-worlds.jsonl").open("w", encoding="utf-8") as raw:
        for index, mechanism in enumerate(mechanisms):
            runs, seen_union, panel = [], set(), None
            for seed in config.policy_seeds:
                for strategy in ("information_gain", "random", "fixed"):
                    world = instantiate_development(mechanism, config.naming_start + index,
                                                    config.permutation_start + index,
                                                    config.exploration_layout_start + index, config.exploration_actions)
                    catalog = world.observe().catalog
                    if panel is None:
                        panel = panels(catalog, config.panel_start + index, config.two_action_probes)
                    model = TransitionModel(catalog)
                    cost = investigate(model, world.observe(), world.act, strategy,
                                       (seed << 32) ^ (config.exploration_layout_start + index), config.exploration_actions)
                    trace = tuple(model.observations)
                    seen = experienced_probes(trace)
                    seen_union.update(seen)
                    before, start = model.simulated_effects, time.perf_counter()
                    predictions = commit_predictions(model, panel)
                    cost.update(prediction_effect_evaluations=model.simulated_effects - before,
                                prediction_seconds=time.perf_counter() - start)
                    runs.append({"seed": seed, "policy": strategy, "predictions": predictions,
                                 "trace": trace, "seen": seen, "cost": cost})
                    if strategy == "information_gain":
                        model.reset()
                        if model.observations:
                            raise AssertionError("Reset retained raw experience")
                        start = time.perf_counter()
                        reset_predictions = commit_predictions(model, panel)
                        runs.append({"seed": seed, "policy": "model_reset", "predictions": reset_predictions,
                                     "trace": trace, "seen": seen,
                                     "cost": {**cost, "prediction_effect_evaluations": model.simulated_effects,
                                              "prediction_seconds": time.perf_counter() - start}})
            # No panel truth is read until every policy/seed has finished committing.
            paths = _targets(world._effects, catalog, panel)
            common = set(panel) - seen_union
            both_seen = sum(len(actions) == 2 and (state, (actions[0],)) in seen_union
                            and (paths[(state, actions)][0], (actions[1],)) in seen_union
                            for state, actions in common)
            coverage.append({"mechanism_id": mechanism.identity,
                             "common_unseen_one_step": sum(len(actions) == 1 for state, actions in common),
                             "common_unseen_two_step": sum(len(actions) == 2 for state, actions in common),
                             "common_unseen_two_step_both_components_seen_by_any_run": both_seen})
            for result in runs:
                costs.append({"mechanism_id": mechanism.identity, "seed": result["seed"],
                              "policy": result["policy"], **result["cost"]})
                for length, name in ((1, "one_step"), (2, "two_step")):
                    for scope in ("full", "common_unseen"):
                        indices = [i for i, probe in enumerate(panel) if len(probe[1]) == length
                                   and (scope == "full" or probe in common)]
                        metrics = score([result["predictions"][i] for i in indices],
                                        [paths[panel[i]][-1] for i in indices])
                        rows.append({"mechanism_id": mechanism.identity, "seed": result["seed"],
                                     "policy": result["policy"], "panel": name, "scope": scope,
                                     "seen_in_own_trace": sum(panel[i] in result["seen"] for i in indices), **metrics})
            raw.write(json.dumps({"mechanism_id": mechanism.identity,
                                  "panel": [{"initial": state, "actions": actions, "true_path": paths[(state, actions)],
                                             "common_unseen": (state, actions) in common} for state, actions in panel],
                                  "runs": [{"seed": r["seed"], "policy": r["policy"], "predictions": r["predictions"],
                                            "trace": [asdict(t) for t in r["trace"]], "cost": r["cost"]} for r in runs]},
                                 allow_nan=False) + "\n")
            if progress and (index + 1) % 10 == 0:
                progress({"completed_development_worlds": index + 1, "worlds": config.worlds})
    policies = ("information_gain", "random", "fixed", "model_reset")
    summary = {str(seed): {policy: {f"{panel}/{scope}": aggregate([
        row for row in rows if row["seed"] == seed and row["policy"] == policy
        and row["panel"] == panel and row["scope"] == scope])
        for panel in ("one_step", "two_step") for scope in ("full", "common_unseen")}
        for policy in policies} for seed in config.policy_seeds}
    cost_summary = {policy: {key: sum(row[key] for row in costs if row["policy"] == policy)
                            / sum(row["policy"] == policy for row in costs)
                            for key in costs[0] if key not in {"mechanism_id", "seed", "policy"}} for policy in policies}
    report = {"schema_version": 1, "phase": "development_prediction_only", "config": asdict(config),
              "protocol": "docs/experiments/structured-action-world.md", "results_by_seed": summary,
              "paired_brier_reductions": comparisons(rows, [m.identity for m in mechanisms]),
              "coverage": coverage, "mean_costs": cost_summary,
              "baseline_roles": {"heuristic": "information_gain (supplied one-step entropy)",
                                 "untrained_no_memory": "model_reset; same prior, no trainable parameters",
                                 "model_reset": "information-gain trace reused; erase posterior and raw replies before prediction"},
              "environment": {"git_commit": revision, "git_dirty": dirty, "python": platform.python_version(),
                              "population_seconds": population_seconds, "elapsed_seconds": time.perf_counter() - began},
              "limitations": ["Development worlds only; no validation/final performance or family transfer",
                              "Supplied effects, independent uniform prior, filtering, rollout and investigation heuristic",
                              "No parameter training, learned strategy, goal planning or acquired abstraction",
                              "Common-unseen panel excludes experience of every compared policy AND seed",
                              "Unseen coverage is conditional on exploration; empty panels stay unavailable",
                              "New sequences can compose already experienced single transitions",
                              "Metrics are world-macro averages; seeds share worlds, bootstrap is exploratory/unadjusted",
                              "Interaction counts matched; compute measured but not matched; wall times descriptive",
                              "Reset/no-memory/untrained are equivalent prediction controls; trace and exploration cost reused",
                              "Model prior differs from canonical population; proper scores do not establish calibration"]}
    (output / "scores.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    (output / "costs.json").write_text(json.dumps(costs, indent=2, allow_nan=False), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
