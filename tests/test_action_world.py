from dataclasses import asdict, FrozenInstanceError
from itertools import permutations

import pytest

from autonomous_learner.action_world import ActionWorld, Catalog, Effect, STATES, effect_space, mechanism_identity


def chain():
    # A named development-only software fixture, not a learning result.
    return (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1))


def test_unknown_action_effects_compose_and_order_matters():
    catalog = Catalog.generate(71)
    world = ActionWorld(catalog, chain(), budget=4)
    old = world.observe()
    states = [world.act(catalog.actions[index]).after.values for index in (1, 0, 1, 2)]
    assert states == [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    assert old.values == (0, 0, 0) and world.remaining == 0
    first = ActionWorld(catalog, chain())
    second = ActionWorld(catalog, chain())
    for index in (0, 1):
        first.act(catalog.actions[index])
    for index in (1, 0):
        second.act(catalog.actions[index])
    assert first.observe().values != second.observe().values


def test_noop_costs_budget_and_rejected_calls_do_not_mutate():
    catalog = Catalog.generate(71)
    world = ActionWorld(catalog, chain(), budget=1)
    before = world.observe()
    with pytest.raises(ValueError):
        world.act("not-an-action")
    assert world.observe() == before and world.remaining == 1
    assert world.act(catalog.actions[1]).before == before
    assert world.observe() == before and world.remaining == 0
    with pytest.raises(RuntimeError):
        world.act(catalog.actions[0])
    assert world.observe() == before and world.remaining == 0


def test_public_snapshots_are_immutable_and_do_not_encode_hidden_effects():
    catalog = Catalog.generate(71)
    world = ActionWorld(catalog, chain())
    other = ActionWorld(catalog, (Effect(0, "set0"),) * 3)
    assert asdict(world.observe()) == asdict(other.observe())
    assert set(asdict(world.observe())) == {"catalog", "values"}
    assert set(asdict(world.observe())["catalog"]) == {"objects", "actions"}
    with pytest.raises(FrozenInstanceError):
        world.observe().values = (1, 1, 1)
    transition = world.act(catalog.actions[0])
    assert set(asdict(transition)) == {"before", "action", "after"}


def test_public_naming_seed_does_not_change_dynamics():
    first, second = [ActionWorld(Catalog.generate(seed), chain()) for seed in (1, 2)]
    assert first.observe().catalog != second.observe().catalog
    for action in (0, 1, 2, 0):
        a = first.act(first.observe().catalog.actions[action])
        b = second.act(second.observe().catalog.actions[action])
        assert a.before.values == b.before.values and a.after.values == b.after.values


def test_effect_hypotheses_are_behaviorally_distinct_and_state_is_not_aliased():
    hypotheses = effect_space()
    assert len(hypotheses) == 45
    tables = {tuple(effect.predict(state) for state in STATES) for effect in hypotheses}
    assert len(tables) == 45
    state = (0, 1, 0)
    assert Effect(0, "toggle", 1, 1).predict(state) == (1, 1, 0)
    assert state == (0, 1, 0)


def test_semantic_identity_groups_all_object_and_action_renamings():
    expected = mechanism_identity(chain())
    for objects in permutations(range(3)):
        renamed = tuple(Effect(objects.index(e.target), e.operation,
                                None if e.guard is None else objects.index(e.guard), e.guard_value)
                        for e in chain())
        for actions in permutations(range(3)):
            assert mechanism_identity(tuple(renamed[i] for i in actions)) == expected
    assert mechanism_identity((Effect(0, "set0"),) * 3) != expected


@pytest.mark.parametrize("kwargs", [{"target": True, "operation": "toggle"},
                                   {"target": 0, "operation": "toggle", "guard": 0},
                                   {"target": 0, "operation": "toggle", "guard_value": 1},
                                   {"target": 0, "operation": "other"}])
def test_invalid_effects_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Effect(**kwargs)


@pytest.mark.parametrize("initial,budget", [((0, 0, True), 12), ([0, 0, 0], 12),
                                         ((0, 0, 0), True), ((0, 0, 0), 0)])
def test_invalid_world_configuration_is_rejected(initial, budget):
    with pytest.raises(ValueError):
        ActionWorld(Catalog.generate(71), chain(), initial, budget)
