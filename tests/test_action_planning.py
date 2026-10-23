import random

import pytest

from autonomous_learner.action_model import TransitionModel
from autonomous_learner.action_planning import plan
from autonomous_learner.action_world import ActionWorld, Catalog, Effect


def test_supplied_planner_uses_acquired_transitions_and_commits_before_execution():
    catalog = Catalog.generate(71)
    effects = (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1))
    exploration = ActionWorld(catalog, effects, budget=3)
    model = TransitionModel(catalog)
    for action in catalog.actions:
        model.observe(exploration.act(action))
    learned_count = len(model.observations)
    selected = plan(model, (0, 0, 0), (1, 1, 1), 3, random.Random(7))
    assert selected.predicted_success == 1 and len(selected.actions) == 3
    assert selected.evaluated_sequences == 1 + 3 + 9 + 27
    assert len(model.observations) == learned_count  # Planning adds no true feedback.
    execution = ActionWorld(catalog, effects, budget=3)
    for action in selected.actions:
        execution.act(action)
    assert execution.observe().values == (1, 1, 1)
    assert len(model.observations) == learned_count


def test_shorter_plans_win_equal_probability_and_ties_are_seeded():
    catalog = Catalog.generate(71)
    model = TransitionModel(catalog)
    # Public hypothetical-model fixture: all action aliases set the same bit.
    model.candidates = {action: (Effect(0, "set1"),) for action in catalog.actions}
    first = plan(model, (0, 0, 0), (1, 0, 0), 2, random.Random(7))
    second = plan(model, (0, 0, 0), (1, 0, 0), 2, random.Random(7))
    assert first == second
    assert len(first.actions) == 1 and first.predicted_success == 1
    assert first.effect_evaluations == 3 + 9 * 2
    already_there = plan(model, (1, 0, 0), (1, 0, 0), 1, random.Random(7))
    assert already_there.actions == () and already_there.predicted_success == 1


def test_zero_budget_does_not_invent_goal_reachability_or_certainty():
    model = TransitionModel(Catalog.generate(71))
    selected = plan(model, (0, 0, 0), (1, 1, 1), 0, random.Random(7))
    assert selected.actions == () and selected.predicted_success == 0
    assert selected.evaluated_sequences == 1 and selected.effect_evaluations == 0
    model.candidates[model.catalog.actions[0]] = ()
    with pytest.raises(ValueError, match="failure"):
        plan(model, (0, 0, 0), (1, 1, 1), 1, random.Random(7))


@pytest.mark.parametrize("budget", [-1, 4, True])
def test_bad_budgets_fail_before_model_simulation(budget):
    model = TransitionModel(Catalog.generate(71))
    with pytest.raises(ValueError):
        plan(model, (0, 0, 0), (1, 1, 1), budget, random.Random(7))
    assert model.simulated_effects == 0
