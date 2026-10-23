from dataclasses import asdict, replace
import json

import pytest
import torch

from autonomous_learner.delayed_memory import MemoryController
from autonomous_learner.delayed_robustness import SCENARIOS, robustness, scenario_config
from autonomous_learner.delayed_training import DelayedConfig, episode, save_checkpoint


def test_shifted_stream_horizon_expiration_and_resource_masks():
    config = DelayedConfig(quiet_ticks=2)
    for name, changes in SCENARIOS.items():
        shifted, options = scenario_config(config, changes)
        result, _, _ = episode("graph_scheduled", 1_300_300, shifted, capture=True, **options)
        ticks = 15 + shifted.quiet_ticks + 14 * options.get("gap_ticks", 0)
        assert len(result["trace"]) == ticks, name
        assert result["trace"][-1]["episodic_size"] == 0
        assert result["writes"] <= shifted.max_consolidations
        assert result["max_belief_size"] <= shifted.belief_capacity
        assert result["capacity_violations"] == 0
    base = episode("graph", 1_300_300, config, capture=True)[0]
    assert base == episode("graph", 1_300_300, config, capture=True,
                           reverse_stream=False, gap_ticks=0)[0]
    with pytest.raises(ValueError, match="gaps"):
        episode("graph", 1_300_300, config, gap_ticks=-1)


def test_evaluation_only_runner_preserves_checkpoints_and_rejects_reuse(tmp_path):
    torch.set_num_threads(1)
    config = DelayedConfig(optimizer_seeds=[42])
    selected = {"stage": "initial", "iteration": 0, "validation_utility": 0.0}
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = MemoryController().eval()
    for suffix in ("", "-imitation"):
        save_checkpoint(checkpoints / f"seed-42{suffix}.pt", model, config, selected)
    source = {"config": asdict(config), "selected_checkpoints": {"42": selected},
              "imitation_checkpoints": {"42": selected}, "environment": {"git_commit": "synthetic-test"}}
    (checkpoints / "report.json").write_text(json.dumps(source))
    before = {p.name: p.read_bytes() for p in checkpoints.iterdir()}
    output = tmp_path / "results"
    report = robustness(checkpoints, output, start=1_300_300, worlds=2, progress=None)
    assert set(report["results"]) == set(SCENARIOS)
    for scenario in report["results"].values():
        assert scenario["42"]["no_memory"]["accuracy"] == 0
        assert scenario["42"]["untrained"] == scenario["42"]["learned"]
    assert before == {p.name: p.read_bytes() for p in checkpoints.iterdir()}
    assert len((output / "world-metrics.csv").read_text().splitlines()) == 1 + 8 * 8 * 2
    with pytest.raises(FileExistsError):
        robustness(checkpoints, output, worlds=1, progress=None)
    with pytest.raises(ValueError, match="disjoint"):
        robustness(checkpoints, tmp_path / "overlap", start=config.test_start, worlds=1)
    save_checkpoint(checkpoints / "seed-42.pt", model, replace(config, belief_capacity=3), selected)
    with pytest.raises(ValueError, match="differs"):
        robustness(checkpoints, tmp_path / "mismatch", worlds=1)
