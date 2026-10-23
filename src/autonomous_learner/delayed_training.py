"""Reproducible delayed-query experiments; hidden truth stays at reward/evaluation boundaries."""
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import combinations
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from torch.nn import functional as F

from .delayed_memory import (GATE_FEATURES, RANK_FEATURES, DelayedMemory, MemoryController,
                             PublicTask, Record, choose, encode, graph_proposal, visible_proofs)
from .world import World


@dataclass(frozen=True)
class DelayedConfig:
    optimizer_seeds: list[int] = field(default_factory=lambda: [7, 42, 97])
    bootstrap_policy: str = "graph_scheduled"
    train_start: int = 700_000
    imitation_worlds: int = 120
    imitation_epochs: int = 16
    rl_episodes: int = 240
    validation_start: int = 800_000
    validation_worlds: int = 40
    test_start: int = 2_000_000
    test_worlds: int = 200
    episodic_capacity: int = 2
    belief_capacity: int = 6
    max_consolidations: int = 6
    quiet_ticks: int = 2
    write_cost: float = 0.01
    learning_rate: float = 0.003
    entropy_bonus: float = 0.002

    def validate(self):
        if self.bootstrap_policy not in {"graph", "graph_scheduled"}:
            raise ValueError("Bootstrap policy must be a public graph heuristic")
        if not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds):
            raise ValueError("Use nonempty unique optimizer seeds")
        if min(self.imitation_worlds, self.imitation_epochs, self.validation_worlds, self.test_worlds, self.episodic_capacity) < 1:
            raise ValueError("Counts and episodic capacity must be positive")
        if min(self.rl_episodes, self.belief_capacity, self.max_consolidations, self.write_cost, self.entropy_bonus) < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid resource or optimization settings")
        if self.quiet_ticks < self.episodic_capacity:
            raise ValueError("Quiet delay must expire episodic evidence before the question")
        spans = [(self.train_start, self.train_start + self.imitation_worlds + self.rl_episodes),
                 (self.validation_start, self.validation_start + self.validation_worlds),
                 (self.test_start, self.test_start + self.test_worlds)]
        for i, (a, b) in enumerate(spans):
            if any(max(a, c) < min(b, d) for c, d in spans[i + 1:]):
                raise ValueError("Training, validation, and test seeds must be disjoint")

    def task(self, world: World) -> PublicTask:
        return PublicTask(world.entities, world.abilities, len(world.facts) + self.quiet_ticks,
                          self.episodic_capacity, self.belief_capacity, self.max_consolidations)


def episode(policy, world_seed: int, config: DelayedConfig, optimizer_seed=42, sample=False,
            capture=False):
    world = World.generate(world_seed)  # Environment/evaluator, not a policy input.
    memory = DelayedMemory(config.task(world))
    rng = random.Random((optimizer_seed << 32) ^ world_seed)
    log_probs, entropies, trace = [], [], []
    max_episodes = max_beliefs = 0
    for fact in (*world.facts, *([None] * config.quiet_ticks)):
        memory.advance(fact)
        indices, logs, entropy = choose(memory, policy, rng, sample)
        log_probs.extend(logs)
        entropies.extend(entropy)
        if indices is not None:
            memory.consolidate(indices)
        max_episodes = max(max_episodes, len(memory.episodes))
        max_beliefs = max(max_beliefs, len(memory.beliefs))
        if capture:
            trace.append({"tick": memory.tick, "action": "WAIT" if indices is None else "CONSOLIDATE",
                          "episodic_size": len(memory.episodes), "belief_size": len(memory.beliefs),
                          "writes": memory.consolidations})
    # Only now reveal/evaluate the particular possible questions and hidden truths.
    assert not memory.episodes, "Expired episodes must not answer delayed questions"
    answers = visible_proofs(memory.beliefs, memory.task)
    correct = sum(query in answers and answers[query][0] == world.truth(query) for query in memory.task.queries())
    count = len(memory.task.queries())
    accuracy, coverage = correct / count, len(answers) / count
    assert max_episodes <= config.episodic_capacity and max_beliefs <= config.belief_capacity
    assert memory.consolidations <= config.max_consolidations
    result = {"world_seed": world_seed, "accuracy": accuracy, "coverage": coverage,
              "selective_accuracy": correct / len(answers) if answers else None,
              "writes": memory.consolidations, "utility": accuracy - config.write_cost * memory.consolidations,
              "max_episodic_size": max_episodes, "max_belief_size": max_beliefs,
              "capacity_violations": 0}
    if capture:
        result["trace"] = trace
    return result, log_probs, entropies


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("Need at least one evaluation world")
    return {"worlds": n, **{key: sum(row[key] for row in rows) / n
            for key in ("accuracy", "coverage", "utility", "writes")},
            "selective_accuracy": (sum(row["accuracy"] for row in rows) / sum(row["coverage"] for row in rows))
            if any(row["coverage"] for row in rows) else None,
            "capacity_violations": sum(row["capacity_violations"] for row in rows),
            "max_episodic_size": max(row["max_episodic_size"] for row in rows),
            "max_belief_size": max(row["max_belief_size"] for row in rows)}


def evaluate(policy, start: int, worlds: int, config: DelayedConfig, seed=42):
    with torch.no_grad():
        rows = [episode(policy, world_seed, config, seed)[0] for world_seed in range(start, start + worlds)]
    return summarize(rows), rows


def demonstrations(config: DelayedConfig):
    examples = []
    for seed in range(config.train_start, config.train_start + config.imitation_worlds):
        world = World.generate(seed)
        memory = DelayedMemory(config.task(world))
        for fact in (*world.facts, *([None] * config.quiet_ticks)):
            memory.advance(fact)
            if not memory.can_consolidate():
                continue
            rank, gate = encode(memory)
            indices, _, _ = choose(memory, config.bootstrap_policy)
            proposal = graph_proposal(memory)
            target = torch.tensor([i in proposal for i in range(len(rank))], dtype=torch.float32)
            examples.append((rank, gate, int(indices is not None), target))
            if indices is not None:
                memory.consolidate(indices)
    return examples


def padded_examples(examples, capacity):
    n = len(examples)
    rank = torch.zeros(n, capacity, len(RANK_FEATURES))
    gate = torch.stack([row[1] for row in examples])
    targets = torch.zeros(n, capacity)
    mask = torch.zeros(n, capacity, dtype=torch.bool)
    for i, (features, _, _, labels) in enumerate(examples):
        size = len(features)
        rank[i, :size], targets[i, :size], mask[i, :size] = features, labels, True
    return rank, gate, torch.tensor([row[2] for row in examples]), targets, mask


def train(config: DelayedConfig, seed: int, progress=None):
    config.validate()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    policy, initial = MemoryController(), MemoryController()
    initial.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rank, gate, actions, targets, mask = padded_examples(demonstrations(config),
                                                        config.episodic_capacity + config.belief_capacity)
    curve, best_state, best_utility, selected = [], None, float("-inf"), None

    def measure(stage, iteration, loss=None):
        nonlocal best_state, best_utility, selected
        metrics, _ = evaluate(policy, config.validation_start, config.validation_worlds, config, seed)
        row = {"stage": stage, "iteration": iteration, "loss": loss, **metrics}
        curve.append(row)
        if metrics["utility"] > best_utility:
            best_utility = metrics["utility"]
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            selected = {"stage": stage, "iteration": iteration, "validation_utility": best_utility}
        if progress:
            progress({"optimizer_seed": seed, **row})

    measure("initial", 0)
    for epoch in range(config.imitation_epochs):
        losses = []
        for batch in torch.randperm(len(rank)).split(64):
            scores, logits = policy(rank[batch], gate[batch], mask[batch])
            chosen = targets[batch].bool()
            pairs = chosen[:, :, None] & ~chosen[:, None, :] & mask[batch, :, None] & mask[batch, None, :]
            differences = scores[:, :, None] - scores[:, None, :]
            rank_loss = (F.softplus(-differences) * pairs).sum() / pairs.sum().clamp_min(1)
            loss = F.cross_entropy(logits, actions[batch]) + rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        measure("imitation", epoch + 1, sum(losses) / len(losses))

    # Copy without consuming RNG: adding an ablation must not change RL sampling.
    imitation = deepcopy(policy)
    imitation.load_state_dict(best_state)
    imitation_selection = dict(selected)
    baseline = 0.0
    for i in range(config.rl_episodes):
        result, log_probs, entropies = episode(policy, config.train_start + config.imitation_worlds + i,
                                              config, seed, sample=True)
        if log_probs:
            loss = (-torch.stack(log_probs).sum() * (result["utility"] - baseline)
                    - config.entropy_bonus * torch.stack(entropies).mean())
            optimizer.zero_grad()
            loss.backward()
            nnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            if not torch.isfinite(nnorm):
                raise FloatingPointError("Nonfinite policy gradient")
            optimizer.step()
            value = float(loss.detach())
        else:
            value = None
        baseline = 0.9 * baseline + 0.1 * result["utility"]
        if (i + 1) % 40 == 0 or i + 1 == config.rl_episodes:
            measure("reinforce", i + 1, value)
    policy.load_state_dict(best_state)
    return policy.eval(), initial.eval(), curve, selected, (imitation.eval(), imitation_selection)


def offline_bound(world: World, capacity: int) -> float:
    records = [Record(f, i) for i, f in enumerate(world.facts)]
    task = PublicTask(world.entities, world.abilities)
    masks = [mask for _, mask in visible_proofs(records, task).values()]
    best = 0
    for subset in combinations(range(len(records)), min(capacity, len(records))):
        selected = sum(1 << i for i in subset)
        count = sum(selected & required == required for required in masks)
        best = max(best, count)
    return best / len(task.queries())


def save_checkpoint(path: Path, policy: MemoryController, config: DelayedConfig, selected: dict):
    torch.save({"schema_version": 1, "state_dict": policy.state_dict(), "config": asdict(config), "selected": selected,
                "rank_features": RANK_FEATURES, "gate_features": GATE_FEATURES}, path)


def load_checkpoint(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported delayed-memory checkpoint version")
    if tuple(data["rank_features"]) != RANK_FEATURES or tuple(data["gate_features"]) != GATE_FEATURES:
        raise ValueError("Delayed-memory feature schema differs")
    config = DelayedConfig(**data["config"])
    config.validate()
    policy = MemoryController()
    policy.load_state_dict(data["state_dict"])
    return policy.eval(), config


def paired_interval(rows: list[dict], baseline: str, samples=2000) -> dict:
    by_world = {}
    for row in rows:
        by_world.setdefault(row["world_seed"], {}).setdefault(row["optimizer_seed"], {})[row["policy"]] = row["utility"]
    differences = [sum(policies["learned"] - policies[baseline] for policies in seeds.values()) / len(seeds)
                   for seeds in by_world.values()]
    rng = random.Random(197)
    resampled = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(samples))
    return {"mean_paired_utility_difference": sum(differences) / len(differences),
            "world_bootstrap95": [resampled[int(samples * 0.025)], resampled[int(samples * 0.975)]],
            "bootstrap_samples": samples, "interpretation": "Optimizer seeds averaged within world; interval over worlds only"}


def experiment(config: DelayedConfig, output: Path, phase: str = "validation", progress=print):
    config.validate()
    if phase not in {"validation", "final"}:
        raise ValueError("phase must be validation or final")
    if output.exists():
        raise FileExistsError("Preserve previous experiments; choose a new output directory")
    output.mkdir(parents=True)
    start, worlds = ((config.test_start, config.test_worlds) if phase == "final"
                     else (config.validation_start, config.validation_worlds))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    began = time.perf_counter()
    results, all_rows, curves, selections, imitation_selections = {}, [], {}, {}, {}
    for seed in config.optimizer_seeds:
        callback = lambda row: progress(json.dumps(row)) if progress else None
        policy, initial, curve, selected, (imitation, imitation_selected) = train(config, seed, callback)
        save_checkpoint(output / f"seed-{seed}.pt", policy, config, selected)
        save_checkpoint(output / f"seed-{seed}-imitation.pt", imitation, config, imitation_selected)
        curves[str(seed)], selections[str(seed)] = curve, selected
        imitation_selections[str(seed)] = imitation_selected
        results[str(seed)] = {}
        for name, model in [("learned", policy), ("imitation_only", imitation), ("untrained", initial), ("random", "random"),
                            ("fifo", "fifo"), ("graph", "graph"), ("graph_scheduled", "graph_scheduled")]:
            metrics, rows = evaluate(model, start, worlds, config, seed)
            results[str(seed)][name] = metrics
            all_rows.extend({"optimizer_seed": seed, "policy": name, **row} for row in rows)
    bounds = [offline_bound(World.generate(i), config.belief_capacity) for i in range(start, start + worlds)]
    comparisons = {name: paired_interval(all_rows, name) for name in ("imitation_only", "untrained", "fifo", "graph", "graph_scheduled")}
    improved = all(result["learned"]["utility"] > result["untrained"]["utility"] for result in results.values())
    passed = improved and comparisons["fifo"]["world_bootstrap95"][0] > 0
    report = {"protocol": "docs/experiments/delayed-goal-memory.md", "phase": phase, "config": asdict(config),
              "evaluation_start": start, "evaluation_worlds": worlds, "results_by_seed": results,
              "selected_checkpoints": selections, "imitation_checkpoints": imitation_selections,
              "curves": curves, "paired_comparisons": comparisons,
              "references": {"no_memory_accuracy": 0.0, "unlimited_storage_accuracy": 1.0,
                             "offline_best_subset_mean_accuracy": sum(bounds) / len(bounds),
                             "offline_bound_is_operational": False},
              "gate": {"passed": passed, "final_evidence": phase == "final",
                       "improves_over_initial_each_seed": improved,
                       "fifo_paired_interval_excludes_zero": comparisons["fifo"]["world_bootstrap95"][0] > 0},
              "environment": {"git_commit": revision, "git_dirty": dirty, "python": platform.python_version(),
                              "torch": str(torch.__version__), "elapsed_seconds": time.perf_counter() - began},
              "limitations": ["Supplied monotonic inference and engineered structural features",
                              "Public vocabulary, query distribution, and horizon; only specific question withheld",
                              "Supervised graph-heuristic bootstrap; not independent discovery of memory rules",
                              "Retention fills available belief slots; admission count is not learned",
                              "Fixed episode TTL and capacities; retrieval and belief revision not learned",
                              "Only three optimizer seeds in main protocol; bootstrap interval covers worlds only"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "world-metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    return report
