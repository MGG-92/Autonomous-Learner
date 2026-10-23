from dataclasses import replace
import json
import math
import random

import pytest

from autonomous_learner.action_investigation import choose_action, investigate
from autonomous_learner.action_model import TransitionModel
import autonomous_learner.action_prediction as experiment
from autonomous_learner.action_prediction import PredictionConfig, aggregate, experienced_probes, score
from autonomous_learner.action_world import ActionWorld, Catalog, Effect, Observation, STATES, Transition


def test_information_gain_uses_current_model_disagreement_and_no_gold():
    catalog = Catalog.generate(71)
    model = TransitionModel(catalog)
    model.candidates = {a: (Effect(2, "set0"),) for a in catalog.actions}
    model.candidates[catalog.actions[0]] = (Effect(0, "toggle"), Effect(0, "set1"))
    current = Observation(catalog, (1, 0, 0))
    assert choose_action(model, current, "information_gain", random.Random(7), 0) == catalog.actions[0]
    assert model.simulated_effects == 4


def test_investigation_pays_for_noops_and_rejects_mismatched_reply_before_learning():
    catalog = Catalog.generate(71)
    model = TransitionModel(catalog)
    world = ActionWorld(catalog, (Effect(0, "set0"),) * 3, budget=3)
    calls = []
    def act(action):
        calls.append(action)
        return world.act(action)
    cost = investigate(model, world.observe(), act, "fixed", 7, 3)
    assert len(calls) == cost["real_actions"] == cost["noop_actions"] == 3
    assert cost["selection_effect_evaluations"] == 0 and world.remaining == 0
    fresh = TransitionModel(catalog)
    initial = Observation(catalog, (0, 0, 0))
    with pytest.raises(ValueError, match="Reply"):
        investigate(fresh, initial, lambda action: Transition(initial, catalog.actions[1], initial), "fixed", 7, 1)
    assert not fresh.observations and fresh.simulated_effects == 0


def test_proper_joint_scores_and_empty_panels_have_honest_denominators():
    p = (0.5, 0.5, 0, 0, 0, 0, 0, 0)
    metrics = score([p, p], [STATES[0], STATES[1]])
    assert metrics == {"n": 2, "accuracy": 0.5, "brier": 0.5, "log_loss": math.log(2)}
    empty = score([], [])
    assert empty["n"] == 0 and empty["accuracy"] is None
    assert aggregate([empty])["nonempty_runs"] == 0
    with pytest.raises(ValueError):
        score([(1, 0, 0, 0, 0, 0, 0, 0)], [STATES[1]])


def test_experience_marks_only_contiguous_two_action_sequences():
    catalog = Catalog.generate(71)
    world = ActionWorld(catalog, (Effect(0, "toggle"),) * 3)
    trace = [world.act(catalog.actions[0]), world.act(catalog.actions[1])]
    seen = experienced_probes(trace)
    assert ((0, 0, 0), catalog.actions[:2]) in seen
    assert ((0, 0, 0), (catalog.actions[1], catalog.actions[0])) not in seen


def test_runner_commits_all_predictions_before_targets_and_preserves_outputs(tmp_path, monkeypatch):
    committed = []
    original_commit, original_targets = experiment.commit_predictions, experiment._targets
    def commit(model, panel):
        result = original_commit(model, panel)
        committed.append(True)
        return result
    def targets(*args):
        assert len(committed) == 8  # Three controls plus reset, across BOTH seeds.
        return original_targets(*args)
    monkeypatch.setattr(experiment, "commit_predictions", commit)
    monkeypatch.setattr(experiment, "_targets", targets)
    config = PredictionConfig(worlds=1, exploration_actions=2, two_action_probes=2, policy_seeds=[7, 42])
    report = experiment.run(config, tmp_path / "first")
    assert report["phase"] == "development_prediction_only"
    raw = json.loads((tmp_path / "first/raw-worlds.jsonl").read_text())
    seen = set()
    def observation(record):
        catalog = Catalog(**{key: tuple(value) for key, value in record["catalog"].items()})
        return Observation(catalog, tuple(record["values"]))
    for r in raw["runs"]:
        trace = [Transition(observation(t["before"]), t["action"], observation(t["after"])) for t in r["trace"]]
        seen.update(experienced_probes(trace))
        assert len(trace) == r["cost"]["real_actions"] == 2
    for p in raw["panel"]:
        assert p["common_unseen"] == ((tuple(p["initial"]), tuple(p["actions"])) not in seen)
    rows = json.loads((tmp_path / "first/scores.json").read_text())
    assert all(row["seen_in_own_trace"] == 0 for row in rows if row["scope"] == "common_unseen")
    committed.clear()
    repeated = experiment.run(config, tmp_path / "second")
    assert report["results_by_seed"] == repeated["results_by_seed"]
    assert report["paired_brier_reductions"] == repeated["paired_brier_reductions"]
    with pytest.raises(FileExistsError):
        experiment.run(config, tmp_path / "first")


@pytest.mark.parametrize("changes", [{"worlds": 0}, {"policy_seeds": [7, 7]}, {"policy_seeds": [True]},
                                    {"two_action_probes": 73}, {"naming_start": 5_000_001}])
def test_bad_config_fails_before_world_execution(changes):
    with pytest.raises(ValueError):
        replace(PredictionConfig(), **changes).validate()
