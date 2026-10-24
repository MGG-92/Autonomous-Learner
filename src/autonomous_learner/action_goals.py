"""Development-only goal evaluator; hidden reachability is post-commit analysis."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from .action_investigation import investigate
from .action_model import TransitionModel
from .action_planning import Plan, plan
from .action_sampling import instantiate_development, sample_development
from .action_world import STATES, Transition


POLICIES = ("information_gain", "random", "fixed", "model_reset", "random_plan")


@dataclass(frozen=True)
class GoalConfig:
    worlds: int = 80
    exploration_actions: int = 12
    goal_actions: int = 3
    policy_seeds: list[int] = field(default_factory=lambda: [7, 42, 97])
    selection_seed: int = 611
    exploration_layout_start: int = 5_000_000
    naming_start: int = 5_100_000
    permutation_start: int = 5_150_000
    goal_layout_start: int = 5_300_000
    goal_target_start: int = 5_400_000
    planning_start: int = 5_500_000

    def validate(self):
        if any(type(v) is not int or v < 1 for v in (self.worlds, self.exploration_actions)):
            raise ValueError("Positive integer world and exploration counts required")
        if type(self.goal_actions) is not int or not 0 <= self.goal_actions <= 3:
            raise ValueError("Goal budget must be an integer from zero to three")
        starts = (self.exploration_layout_start, self.naming_start, self.permutation_start,
                  self.goal_layout_start, self.goal_target_start, self.planning_start)
        if (not isinstance(self.policy_seeds, list) or not self.policy_seeds
                or any(type(v) is not int or v < 0 for v in (*starts, self.selection_seed, *self.policy_seeds))
                or len(set(self.policy_seeds)) != len(self.policy_seeds)):
            raise ValueError("Use nonnegative seeds and unique policy seeds")
        if any(max(a, b) < min(a + self.worlds, b + self.worlds)
               for i, a in enumerate(starts) for b in starts[i + 1:]):
            raise ValueError("Purpose-specific seed ranges must be disjoint")


def sample_goal(initial, seed):
    """Independent of mechanisms/reachability; never rejection-sample success."""
    return random.Random(seed).choice(tuple(state for state in STATES if state != initial))


def execute(initial, act, selected):
    """Execute an immutable commitment without a model or online replanning."""
    current, trace = initial, []
    for action in selected.actions:
        transition = act(action)
        if (not isinstance(transition, Transition) or transition.before != current
                or transition.action != action):
            raise ValueError("Reply must match committed action and current observation")
        trace.append(transition)
        current = transition.after
    return current.values, tuple(trace)


def reachable(effects, initial, goal, budget):
    """Evaluator-only bounded reachability, called after every plan is committed."""
    states, work = {initial}, 0
    for _ in range(budget):
        expanded = set(states)
        for state in states:
            for effect in effects:
                expanded.add(effect.predict(state))
                work += 1
        states = expanded
    return goal in states, work


def summarize(rows):
    result = {}
    for scope in ("all", "reachable", "unreachable"):
        group = [r for r in rows if scope == "all" or r["reachable"] == (scope == "reachable")]
        n = len(group)
        result[scope] = {"n": n, "success_rate": sum(r["success"] for r in group) / n if n else None,
                         "mean_goal_actions": sum(r["goal_actions"] for r in group) / n if n else None}
    return result


def comparisons(rows, identities):
    result = {}
    for baseline in POLICIES[1:]:
        differences = []
        for identity in identities:
            world = [r for r in rows if r["mechanism_id"] == identity]
            active = [r["success"] for r in world if r["policy"] == "information_gain"]
            other = [r["success"] for r in world if r["policy"] == baseline]
            differences.append(sum(active) / len(active) - sum(other) / len(other))
        rng = random.Random(197)
        bootstrap = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                           for _ in range(2000))
        result[baseline] = {"worlds": len(differences),
                            "mean_success_gain": sum(differences) / len(differences),
                            "world_bootstrap95": [bootstrap[50], bootstrap[1950]],
                            "per_world": differences, "bootstrap_samples": 2000}
    return result


def run(config: GoalConfig, output: Path, progress=None):
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
    with (output / "commitments.jsonl").open("w", encoding="utf-8") as commits, \
            (output / "raw-worlds.jsonl").open("w", encoding="utf-8") as raw:
        for index, mechanism in enumerate(mechanisms):
            def world_at(initial_seed, budget):
                return instantiate_development(mechanism, config.naming_start + index,
                                               config.permutation_start + index, initial_seed, max(1, budget))
            # Ordinary public start; goal selection takes no hidden mechanism argument.
            initial = world_at(config.goal_layout_start + index, config.goal_actions).observe()
            goal = sample_goal(initial.values, config.goal_target_start + index)
            commitments = []
            for seed in config.policy_seeds:
                planning_seed = (seed << 32) ^ (config.planning_start + index)
                for strategy in ("information_gain", "random", "fixed"):
                    exploration = world_at(config.exploration_layout_start + index, config.exploration_actions)
                    model = TransitionModel(initial.catalog)
                    cost = investigate(model, exploration.observe(), exploration.act, strategy,
                                       (seed << 32) ^ (config.exploration_layout_start + index), config.exploration_actions)
                    trace = tuple(model.observations)

                    def commit(policy, selected, planning_seconds, candidate_counts):
                        commitments.append({"policy": policy, "seed": seed, "plan": asdict(selected),
                                            "trace": [asdict(t) for t in trace],
                                            "candidate_counts": candidate_counts,
                                            "cost": {**cost, "planning_seconds": planning_seconds,
                                                     "planning_effect_evaluations": selected.effect_evaluations,
                                                     "evaluated_sequences": selected.evaluated_sequences}})

                    start = time.perf_counter()
                    selected = plan(model, initial.values, goal, config.goal_actions, random.Random(planning_seed))
                    commit(strategy, selected, time.perf_counter() - start,
                           {a: len(c) for a, c in model.candidates.items()})
                    if strategy == "information_gain":
                        model.reset()
                        if model.observations:
                            raise AssertionError("Reset retained raw evidence")
                        start = time.perf_counter()
                        prior = plan(model, initial.values, goal, config.goal_actions, random.Random(planning_seed))
                        commit("model_reset", prior, time.perf_counter() - start,
                               {a: len(c) for a, c in model.candidates.items()})
                        start = time.perf_counter()
                        rng = random.Random(planning_seed)
                        actions = tuple(rng.choice(initial.catalog.actions) for _ in range(config.goal_actions))
                        commit("random_plan", Plan(actions, None, 0, 0), time.perf_counter() - start, None)
            # Persist all policies AND seeds before any goal execution or true reachability.
            commits.write(json.dumps({"mechanism_id": mechanism.identity, "initial": asdict(initial),
                                      "goal": goal, "runs": commitments}, allow_nan=False) + "\n")
            commits.flush()
            is_reachable, reach_work = reachable(world_at(config.goal_layout_start + index, 1)._effects,
                                                initial.values, goal, config.goal_actions)
            coverage.append({"mechanism_id": mechanism.identity, "reachable": is_reachable,
                             "reachability_effect_evaluations": reach_work})
            executions = []
            for record in commitments:
                selected = Plan(**record["plan"])
                execution = world_at(config.goal_layout_start + index, config.goal_actions)
                start = time.perf_counter()
                final, trace = execute(execution.observe(), execution.act, selected)
                elapsed = time.perf_counter() - start
                success = final == goal
                if success and not is_reachable:
                    raise AssertionError("Execution exceeded true bounded reachability")
                rows.append({"mechanism_id": mechanism.identity, "seed": record["seed"], "policy": record["policy"],
                             "reachable": is_reachable, "success": success, "goal_actions": len(trace),
                             "predicted_success": selected.predicted_success})
                costs.append({"mechanism_id": mechanism.identity, "seed": record["seed"], "policy": record["policy"],
                              **record["cost"], "goal_actions": len(trace), "execution_seconds": elapsed,
                              "goal_noops": sum(t.before.values == t.after.values for t in trace)})
                executions.append({"seed": record["seed"], "policy": record["policy"], "final": final,
                                   "trace": [asdict(t) for t in trace], "success": success})
            raw.write(json.dumps({"mechanism_id": mechanism.identity, "reachable": is_reachable,
                                  "executions": executions}, allow_nan=False) + "\n")
            if progress:
                progress({"completed_development_worlds": index + 1, "worlds": config.worlds})
    report = {"schema_version": 1, "phase": "development_goals_only", "config": asdict(config),
              "protocol": "docs/experiments/action-goals.md",
              "results_by_seed": {str(seed): {policy: summarize([r for r in rows if r["seed"] == seed
                                                               and r["policy"] == policy])
                                              for policy in POLICIES} for seed in config.policy_seeds},
              "paired_success_gains": comparisons(rows, [m.identity for m in mechanisms]), "coverage": coverage,
              "mean_costs": {policy: {key: sum(r[key] for r in costs if r["policy"] == policy)
                                      / sum(r["policy"] == policy for r in costs)
                                      for key in costs[0] if key not in {"mechanism_id", "seed", "policy"}}
                             for policy in POLICIES},
              "baseline_roles": {"model_reset": "prior/no-memory/untrained equivalent; erased IG model; exploration costs reused",
                                 "random_plan": "uniform independent action slots, full goal budget; no model; IG exploration costs reused"},
              "environment": {"git_commit": revision, "git_dirty": dirty, "python": platform.python_version(),
                              "population_seconds": population_seconds, "elapsed_seconds": time.perf_counter() - began},
              "limitations": ["Development only; one independent goal per mechanism; seeds share worlds and goals",
                              "Supplied templates, filtering, investigation and exhaustive planning; no parameter training",
                              "No online replanning, structural revision, acquired abstractions, self-model or family transfer",
                              "Model-relative success probabilities are not measured calibration",
                              "Same interaction ceilings; actual goal lengths differ; compute measured, not matched",
                              "Reset and random-plan reuse IG exploration trace/cost; not extra investigation runs",
                              "World bootstrap intervals exploratory, unadjusted; no seed independence assumption"]}
    for name, value in (("scores", rows), ("costs", costs), ("report", report)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return report
