"""Evaluation-only shifts of frozen delayed-memory controllers; never selects models."""
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from .delayed_memory import MemoryController
from .delayed_training import DelayedConfig, evaluate, load_checkpoint, paired_interval


# Freeze before opening the reserved worlds. Each shift changes one factor.
SCENARIOS = {
    "reference": {},
    "reverse_order": {"reverse_stream": True},
    "observation_gaps": {"gap_ticks": 2},
    "long_quiet_delay": {"quiet_ticks": 10},
    "small_belief": {"belief_capacity": 3},
    "large_belief": {"belief_capacity": 9},
    "few_writes": {"max_consolidations": 3},
    "many_writes": {"max_consolidations": 9},
}


def scenario_config(config, changes):
    resources = {k: v for k, v in changes.items() if k not in {"reverse_stream", "gap_ticks"}}
    shifted = replace(config, **resources)
    shifted.validate()
    return shifted, {k: v for k, v in changes.items() if k not in resources}


def robustness(checkpoints: Path, output: Path, start=2_100_000, worlds=100, progress=print):
    """Load already selected policies, with paired worlds across all shifts/baselines."""
    if output.exists():
        raise FileExistsError("Preserve previous experiments; choose a new output directory")
    if start < 0 or worlds < 1:
        raise ValueError("Use a nonnegative start and positive world count")
    source = json.loads((checkpoints / "report.json").read_text(encoding="utf-8"))
    config = DelayedConfig(**source["config"])
    config.validate()
    # Keep the new range out of every range previously used for this experiment.
    for a, b in ((config.train_start, config.train_start + config.imitation_worlds + config.rl_episodes),
                 (config.validation_start, config.validation_start + config.validation_worlds),
                 (config.test_start, config.test_start + config.test_worlds)):
        if max(a, start) < min(b, start + worlds):
            raise ValueError("Robustness worlds must be disjoint from prior experiment ranges")
    policies, identities = {}, {}
    for seed in config.optimizer_seeds:
        models = {}
        for name, suffix, selection in (("learned", "", "selected_checkpoints"),
                                         ("imitation_only", "-imitation", "imitation_checkpoints")):
            path = checkpoints / f"seed-{seed}{suffix}.pt"
            model, stored_config = load_checkpoint(path)
            metadata = torch.load(path, map_location="cpu", weights_only=True)
            if stored_config != config or metadata["selected"] != source[selection][str(seed)]:
                raise ValueError("Checkpoint configuration/selection differs from source report")
            models[name] = model
            identities[f"seed-{seed}{suffix}"] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selection": metadata["selected"],
            }
        # Exactly the initialization used before training; independent of load RNG.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            models["untrained"] = MemoryController().eval()
        policies[seed] = models
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    output.mkdir(parents=True)
    began = time.perf_counter()
    results, all_rows, comparisons = {}, [], {}
    for scenario, changes in SCENARIOS.items():
        shifted, options = scenario_config(config, changes)
        scenario_rows = []
        results[scenario] = {}
        for seed, models in policies.items():
            results[scenario][str(seed)] = {}
            for name, model in [*models.items(), ("random", "random"), ("fifo", "fifo"),
                                ("graph", "graph"), ("graph_scheduled", "graph_scheduled"),
                                ("no_memory", models["learned"])]:
                active = replace(shifted, belief_capacity=0) if name == "no_memory" else shifted
                metrics, rows = evaluate(model, start, worlds, active, seed, **options)
                results[scenario][str(seed)][name] = metrics
                scenario_rows.extend({"scenario": scenario, "optimizer_seed": seed, "policy": name, **row}
                                     for row in rows)
            if progress:
                progress(f"{scenario}: optimizer seed {seed} evaluated", flush=True)
        comparisons[scenario] = {name: paired_interval(scenario_rows, name)
                                 for name in ("fifo", "graph_scheduled", "imitation_only")}
        all_rows.extend(scenario_rows)
    report = {
        "schema_version": 1, "protocol": "docs/experiments/delayed-memory-robustness.md",
        "evaluation_start": start, "evaluation_worlds": worlds,
        "training_config": asdict(config), "scenarios": SCENARIOS,
        "checkpoint_source": source["environment"], "checkpoints": identities,
        "results": results, "paired_comparisons": comparisons,
        "environment": {"git_commit": revision, "git_dirty": dirty, "python": platform.python_version(),
                        "torch": str(torch.__version__), "elapsed_seconds": time.perf_counter() - began},
        "limitations": ["Evaluation only: no retraining, selection, or new success gate",
                        "Exploratory unadjusted intervals; worlds shared across seeds and scenarios",
                        "Fixed graph family and public horizons; supplied inference and features",
                        "Quiet-delay extension cannot decay permanent belief storage",
                        "Observation gaps change horizon and opportunities to consolidate together",
                        "Episodic capacity and TTL remain coupled and fixed at training values"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "world-metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    return report
