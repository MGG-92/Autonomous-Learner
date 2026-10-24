from dataclasses import replace
import json

import pytest

import autonomous_learner.action_goals as experiment
from autonomous_learner.action_goals import GoalConfig, execute, reachable, sample_goal, summarize
from autonomous_learner.action_planning import Plan
from autonomous_learner.action_world import ActionWorld, Catalog, Effect, Observation, STATES, Transition


def test_commitments_across_all_seeds_precede_gold_and_execution(tmp_path, monkeypatch):
    committed, models = [], []
    original_plan, original_reachable, original_execute = experiment.plan, experiment.reachable, experiment.execute
    def plan(model, *args):
        selected = original_plan(model, *args)
        committed.append(selected)
        models.append(model)
        return selected
    def check_committed():
        assert len(committed) == 8  # Three learned + prior, both seeds; random has no model call.
        persisted = json.loads((tmp_path / "run/commitments.jsonl").read_text())
        assert len(persisted["runs"]) == 10
    def gold(*args):
        check_committed()
        return original_reachable(*args)
    def execution(*args):
        check_committed()
        snapshots = [(tuple(m.observations), m.simulated_effects) for m in models]
        result = original_execute(*args)
        assert snapshots == [(tuple(m.observations), m.simulated_effects) for m in models]
        return result
    monkeypatch.setattr(experiment, "plan", plan)
    monkeypatch.setattr(experiment, "reachable", gold)
    monkeypatch.setattr(experiment, "execute", execution)
    config = GoalConfig(worlds=1, exploration_actions=3, goal_actions=1, policy_seeds=[7, 42])
    report = experiment.run(config, tmp_path / "run")
    assert report["phase"] == "development_goals_only"
    # IG references were explicitly reset, not augmented with goal execution evidence.
    assert all(not m.observations for m in models[::4])
    with pytest.raises(FileExistsError):
        experiment.run(config, tmp_path / "run")


def test_goals_do_not_condition_on_reachability_and_empty_denominators_stay_unknown():
    initial = (0, 0, 0)
    effects = (Effect(0, "set0"),) * 3
    goals = {sample_goal(initial, seed) for seed in range(100)}
    assert goals == set(STATES) - {initial}
    assert all(not reachable(effects, initial, goal, 3)[0] for goal in goals)
    rows = [{"reachable": False, "success": False, "goal_actions": 0}]
    assert summarize(rows)["reachable"] == {"n": 0, "success_rate": None, "mean_goal_actions": None}
    assert summarize(rows)["all"]["success_rate"] == 0


def test_committed_execution_counts_noops_and_rejects_wrong_reply():
    catalog = Catalog.generate(71)
    world = ActionWorld(catalog, (Effect(0, "set0"),) * 3, budget=3)
    selected = Plan(catalog.actions, None, 0, 0)
    final, trace = execute(world.observe(), world.act, selected)
    assert final == (0, 0, 0) and len(trace) == 3 and world.remaining == 0
    initial = Observation(catalog, (0, 0, 0))
    with pytest.raises(ValueError, match="Reply"):
        execute(initial, lambda a: Transition(initial, catalog.actions[1], initial), selected)


def test_reachability_respects_order_and_budget():
    effects = (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1))
    assert not reachable(effects, (0, 0, 0), (1, 1, 1), 2)[0]
    assert reachable(effects, (0, 0, 0), (1, 1, 1), 3)[0]
    assert reachable(effects, (0, 0, 0), (0, 0, 0), 0) == (True, 0)


def test_zero_goal_budget_never_executes_or_reports_success(tmp_path):
    report = experiment.run(GoalConfig(worlds=1, exploration_actions=1, goal_actions=0,
                                       policy_seeds=[7]), tmp_path / "zero")
    for result in report["results_by_seed"]["7"].values():
        assert result["all"] == {"n": 1, "success_rate": 0, "mean_goal_actions": 0}
        assert result["reachable"]["n"] == 0
    assert all(c["goal_actions"] == 0 for c in report["mean_costs"].values())


def test_runner_reproducible_and_raw_committed_plans_reconstruct_results(tmp_path):
    config = GoalConfig(worlds=1, exploration_actions=3, goal_actions=1, policy_seeds=[7])
    first = experiment.run(config, tmp_path / "a")
    second = experiment.run(config, tmp_path / "b")
    assert first["results_by_seed"] == second["results_by_seed"]
    assert first["paired_success_gains"] == second["paired_success_gains"]
    record = json.loads((tmp_path / "a/commitments.jsonl").read_text())
    raw = json.loads((tmp_path / "a/raw-worlds.jsonl").read_text())
    scores = json.loads((tmp_path / "a/scores.json").read_text())
    costs = json.loads((tmp_path / "a/costs.json").read_text())
    assert record["initial"]["values"] != record["goal"]
    for committed, executed, score, cost in zip(record["runs"], raw["executions"], scores, costs):
        assert [t["action"] for t in executed["trace"]] == committed["plan"]["actions"]
        assert score["success"] == (executed["final"] == record["goal"])
        assert cost["goal_actions"] == len(executed["trace"])
        assert cost["real_actions"] == len(committed["trace"]) == 3
        if committed["policy"] == "model_reset":
            assert set(committed["candidate_counts"].values()) == {45}
        if committed["policy"] == "random_plan":
            assert cost["planning_effect_evaluations"] == 0
            assert score["predicted_success"] is None


@pytest.mark.parametrize("changes", [{"worlds": 0}, {"goal_actions": 4}, {"goal_actions": True},
                                    {"policy_seeds": [7, 7]}, {"policy_seeds": [True]},
                                    {"goal_target_start": 5_300_001}])
def test_bad_config_rejected_before_world_execution(changes):
    with pytest.raises(ValueError):
        replace(GoalConfig(), **changes).validate()
