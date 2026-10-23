"""Development-only REINFORCE experiment. No final-world evaluation entry point."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from torch.distributions import Categorical

from .query_policy import QueryPolicy, choose_query, query_features
from .rule_experiment import scores
from .rule_induction import RuleLearner, build_space


@dataclass(frozen=True)
class QueryConfig:
    episodes: int = 800
    worlds: int = 80
    queries: int = 4
    heldout_inputs: int = 5
    optimizer_seeds: list[int] = field(default_factory=lambda: [7, 42, 97])
    data_seed: int = 411
    evaluation_seed: int = 521
    training_layout_start: int = 4_000_000
    validation_layout_start: int = 4_100_000
    learning_rate: float = 0.005
    entropy_bonus: float = 0.01

    def validate(self):
        for value in (self.episodes, self.worlds, self.queries, self.heldout_inputs):
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer counts required")
        seeds = [self.data_seed, self.evaluation_seed, self.training_layout_start,
                 self.validation_layout_start]
        if not isinstance(self.optimizer_seeds, list) or not self.optimizer_seeds:
            raise ValueError("At least one optimizer seed required")
        seeds += self.optimizer_seeds
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Nonnegative integer seeds required")
        if len(set(self.optimizer_seeds)) != len(self.optimizer_seeds):
            raise ValueError("Optimizer seeds must be unique")
        if self.heldout_inputs >= 25 or self.queries > 25 - self.heldout_inputs:
            raise ValueError("Queries and prediction probes must fit the domain")
        a, b = self.training_layout_start, self.validation_layout_start
        if max(a, b) < min(a + self.episodes, b + self.worlds):
            raise ValueError("Training and validation layout seeds must be disjoint")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.entropy_bonus) or self.entropy_bonus < 0):
            raise ValueError("Invalid optimization settings")


def layout(space, seed, heldout_count):
    probes = tuple(random.Random(seed).sample(space.inputs, heldout_count))
    return tuple(pair for pair in space.inputs if pair not in probes), probes


class MeteredLearner(RuleLearner):
    """Count hypothetical output reads, excluding oracle replies and gold targets."""
    def __init__(self, space, legal):
        super().__init__(space, legal)
        self.model_value_reads = 0

    def distribution(self, pair):
        self.model_value_reads += len(self.candidates)
        return super().distribution(pair)

    def observe(self, pair, value):
        self.model_value_reads += len(self.candidates)
        super().observe(pair, value)


def train_policy(config, seed, space):
    # Training truths/rewards exist only here, never in policy features.
    development = tuple(h for h in space.hypotheses if h.partition == "development")
    rng = random.Random(config.data_seed)
    torch.manual_seed(seed)
    policy = QueryPolicy()
    initial = deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    baseline, curve, reward_sum, reads = 0.0, [], 0.0, 0
    began = time.perf_counter()
    for episode in range(config.episodes):
        hidden = rng.choice(development)
        legal, probes = layout(space, config.training_layout_start + episode, config.heldout_inputs)
        learner = MeteredLearner(space, legal)
        # Training-only reward labels are excluded from the policy's inputs.
        truth = [hidden.values[space.inputs.index(pair)] for pair in probes]
        prior_brier = scores([learner.distribution(pair) for pair in probes], truth)["brier"]
        log_probs, entropies = [], []
        for step in range(config.queries):
            if len(learner.candidates) <= 1:
                break
            pairs, features = query_features(learner, config.queries - step, config.queries)
            distribution = Categorical(logits=policy(features))
            action = distribution.sample()
            pair = pairs[action.item()]
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            learner.observe(pair, hidden.values[space.inputs.index(pair)])
        posterior_brier = scores([learner.distribution(pair) for pair in probes], truth)["brier"]
        reward = prior_brier - posterior_brier
        if log_probs:
            loss = (-(reward - baseline) * torch.stack(log_probs).sum()
                    - config.entropy_bonus * torch.stack(entropies).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        baseline = 0.95 * baseline + 0.05 * reward
        reward_sum += reward
        reads += learner.model_value_reads
        if (episode + 1) % 100 == 0 or episode + 1 == config.episodes:
            curve.append({"episode": episode + 1, "mean_training_reward": reward_sum / (episode + 1)})
    return policy.eval(), initial.eval(), {"seconds": time.perf_counter() - began,
                                          "model_value_reads": reads, "curve": curve}


def evaluate_policy(space, hidden, legal, probes, strategy, policy, seed, budget):
    learner = MeteredLearner(space, legal)
    rng, query_seconds, trace, selection_reads = random.Random(seed), 0.0, [], 0
    for step in range(budget):
        if len(learner.candidates) <= 1:
            break
        began = time.perf_counter()
        before = learner.model_value_reads
        if strategy == "prior_features":
            # Remove experience only from query selection; predictions still use it.
            prior = MeteredLearner(space, legal)
            prior.asked = learner.asked.copy()
            pair = choose_query(policy, prior, rng, budget - step, budget)
            selection_reads += prior.model_value_reads
        elif strategy in {"learned", "untrained", "model_reset"}:
            pair = choose_query(policy, learner, rng, budget - step, budget)
        else:
            pair = learner.choose(strategy, rng)
        selection_reads += learner.model_value_reads - before
        query_seconds += time.perf_counter() - began
        if pair is None:
            break
        # The evaluator supplies only the chosen query's actual outcome.
        observed = hidden.values[space.inputs.index(pair)]
        learner.observe(pair, observed)
        trace.append((pair, observed))
    if strategy == "model_reset":
        learner.reset_model()
    predictions = [learner.distribution(pair) for pair in probes]
    truth = [hidden.values[space.inputs.index(pair)] for pair in probes]
    metrics = scores(predictions, truth)
    return {**metrics, "queries": len(trace), "selection_model_value_reads": selection_reads,
            "total_model_value_reads": learner.model_value_reads +
            (selection_reads if strategy == "prior_features" else 0),
            "query_seconds": query_seconds}, trace


def run(config: QueryConfig, output: Path):
    config.validate()
    if output.exists():
        raise FileExistsError("Preserve earlier runs; choose a fresh output directory")
    space = build_space()
    validation = tuple(h for h in space.hypotheses if h.partition == "validation")
    if config.worlds > len(validation):
        raise ValueError("Not enough distinct validation functions")
    worlds = random.Random(config.evaluation_seed).sample(validation, config.worlds)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    output.mkdir(parents=True)
    results, training, paired, traces, raw_rows = {}, {}, {}, {}, []
    for seed in config.optimizer_seeds:
        policy, initial, training[str(seed)] = train_policy(config, seed, space)
        torch.save(policy.state_dict(), output / f"policy-{seed}.pt")
        rows = []
        for index, hidden in enumerate(worlds):
            layout_seed = config.validation_layout_start + index
            legal, probes = layout(space, layout_seed, config.heldout_inputs)
            for strategy in ("learned", "untrained", "information_gain", "random", "fixed",
                             "prior_features", "model_reset"):
                metrics, trace = evaluate_policy(space, hidden, legal, probes, strategy,
                                                 initial if strategy == "untrained" else policy,
                                                 layout_seed ^ config.evaluation_seed, config.queries)
                rows.append({"rule_id": hidden.identity, "policy": strategy, **metrics})
                if index == 0:
                    traces[f"{seed}/{strategy}"] = trace
        raw_rows.extend({"optimizer_seed": seed, **row} for row in rows)
        results[str(seed)] = {}
        for strategy in sorted({row["policy"] for row in rows}):
            group = [row for row in rows if row["policy"] == strategy]
            results[str(seed)][strategy] = {key: sum(row[key] for row in group) / len(group)
                                           for key in group[0] if key not in {"rule_id", "policy"}}
        paired[str(seed)] = {}
        for baseline in ("untrained", "information_gain", "random", "prior_features", "model_reset"):
            differences = []
            for hidden in worlds:
                group = {row["policy"]: row["brier"] for row in rows if row["rule_id"] == hidden.identity}
                differences.append(group[baseline] - group["learned"])
            paired[str(seed)][baseline] = {"mean_brier_reduction": sum(differences) / len(differences),
                                          "per_world": differences}
    report = {"schema_version": 1, "phase": "development_validation_only", "config": asdict(config),
              "protocol": "docs/experiments/learned-investigation.md", "results_by_seed": results,
              "training": training, "paired_brier_reduction": paired, "example_traces": traces,
              "validation_rule_ids": [h.identity for h in worlds],
              "environment": {"git_commit": revision, "git_dirty": dirty,
                              "python": platform.python_version(), "torch": torch.__version__},
              "checkpoint_selection": "last fixed training episode; no validation selection",
              "baseline_roles": {"no_memory": "model_reset erases posterior and raw replies after learned queries",
                                 "untrained": "same seed's exact initial network, evaluated greedily",
                                 "prior_features": "query selection cannot access updated model; prediction still can"},
              "limitations": ["Development validation, not a new final result or family-transfer test",
                              "Supplied finite grammar, uniform prior, exact inference, features and Brier reward",
                              "Learns query ranking only; no learned primitives, inference, abstractions or planner",
                              "All model-based scorers scan every legal query; no search speedup claimed",
                              "Read counts exclude catalog construction, neural arithmetic and optimizer work",
                              "Wall times are descriptive; compute is measured, not equalized across baselines",
                              "Shared validation worlds across optimizer seeds; seeds are not independent worlds",
                              "No-memory and post-query reset are equivalent for prediction in this model",
                              "Sorted masses hide output identities and cannot represent every useful query strategy"]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "validation-rows.json").write_text(json.dumps(raw_rows, indent=2, allow_nan=False), encoding="utf-8")
    return report
