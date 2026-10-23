from dataclasses import replace
import hashlib
from itertools import permutations
import random
import subprocess
import sys

import pytest

from autonomous_learner.action_model import TransitionModel
from autonomous_learner.action_sampling import (Mechanism, instantiate_development, mechanism_bank,
                                               partition, reference_identities, sample_development)
from autonomous_learner.action_world import (ActionWorld, Catalog, Effect, Observation, STATES,
                                            Transition, effect_space, mechanism_identity)


def test_observed_transition_preserves_fixed_effect_across_rollout():
    catalog = Catalog.generate(71)
    effects = (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1))
    world = ActionWorld(catalog, effects, initial=(1, 0, 0))
    model = TransitionModel(catalog)
    action = catalog.actions[0]
    model.observe(world.act(action))  # Could be set0 or toggle, with several guards.
    assert len(model.candidates[action]) == 6
    assert effects[0] in model.candidates[action]
    # Both fixed set0 and fixed toggle end at 000 after two actions from 000.
    work_before = model.simulated_effects
    joint = model.distribution((0, 0, 0), (action, action))
    assert joint == (1, 0, 0, 0, 0, 0, 0, 0)
    assert model.simulated_effects - work_before == 6 * 2
    # The incorrect independent-step approximation instead assigns 25% to 100.
    first = model.distribution((0, 0, 0), (action,))
    resampled = [0.0] * 8
    for state, probability in zip(STATES, first):
        for index, p in enumerate(model.distribution(state, (action,))):
            resampled[index] += probability * p
    assert resampled[STATES.index((1, 0, 0))] == 0.25
    assert tuple(resampled) != joint


def test_reset_erases_model_and_raw_experience():
    catalog = Catalog.generate(71)
    action = catalog.actions[0]
    model = TransitionModel(catalog)
    prior = model.distribution((1, 0, 0), (action,))
    world = ActionWorld(catalog, (Effect(0, "toggle"),) * 3, initial=(1, 0, 0))
    model.observe(world.act(action))
    assert model.distribution((1, 0, 0), (action,)) != prior
    model.reset()
    assert model.observations == [] and model.simulated_effects == 0
    assert all(len(candidates) == 45 for candidates in model.candidates.values())
    assert model.distribution((1, 0, 0), (action,)) == prior


def test_sampler_is_uniform_over_unique_development_identities_and_reproducible():
    bank = mechanism_bank()
    assert len(bank) == len({m.identity for m in bank})
    assert reference_identities() <= {m.identity for m in bank}
    assert all(partition(identity) == "development" for identity in reference_identities())
    expected = random.Random(611).sample([m for m in bank if partition(m.identity) == "development"], 80)
    sampled = sample_development(80, 611)
    assert sampled == tuple(expected) == sample_development(80, 611)
    assert len({m.identity for m in sampled}) == 80
    for name in ("development", "validation", "final"):
        assert any(partition(m.identity) == name for m in bank)


def test_sampled_development_world_retains_true_hypotheses_and_call_budget():
    for index, mechanism in enumerate(sample_development(8, 611)):
        world = instantiate_development(mechanism, 5_100_000 + index, 5_150_000 + index,
                                       5_000_000 + index, budget=12)
        assert mechanism_identity(world._effects) == mechanism.identity  # Researcher assertion only.
        model = TransitionModel(world.observe().catalog)
        for step in range(12):
            action = model.catalog.actions[step % 3]
            model.observe(world.act(action))
            assert model.valid
            assert all(effect in model.candidates[action]
                       for action, effect in zip(model.catalog.actions, world._effects))
        assert world.remaining == 0 and len(model.observations) == 12
        for action in model.catalog.actions:
            assert sum(model.distribution(world.observe().values, (action,))) == pytest.approx(1)


def test_independent_slot_permutations_change_roles_without_changing_names_or_identity():
    effects = (Effect(0, "toggle"), Effect(1, "set1", 0, 1), Effect(2, "toggle", 1, 1))
    mechanism = Mechanism(mechanism_identity(effects), effects)
    worlds = [instantiate_development(mechanism, 71, seed, 31) for seed in range(12)]
    assert len({world.observe().catalog for world in worlds}) == 1
    assert len({world.observe().values for world in worlds}) == 1
    assert len({world._effects for world in worlds}) > 1
    assert {mechanism_identity(world._effects) for world in worlds} == {mechanism.identity}
    reserved = next(m for m in mechanism_bank() if partition(m.identity) == "final")
    with pytest.raises(ValueError, match="development"):
        instantiate_development(reserved, 71, 0, 31)
    with pytest.raises(ValueError):
        instantiate_development(replace(mechanism, identity="forged"), 71, 0, 31)


def test_cached_identity_matches_original_uncached_definition():
    rng = random.Random(314)
    for _ in range(12):
        effects = tuple(rng.choices(effect_space(), k=3))
        signatures = []
        for order in permutations(range(3)):
            relabeled = tuple(Effect(order.index(e.target), e.operation,
                                     None if e.guard is None else order.index(e.guard), e.guard_value)
                              for e in effects)
            signatures.append(b"".join(sorted(bytes(bit for state in STATES for bit in e.predict(state))
                                                for e in relabeled)))
        assert mechanism_identity(effects) == hashlib.sha256(min(signatures)).hexdigest()


def test_malformed_transitions_do_not_mutate_model_and_misspecification_is_explicit():
    catalog = Catalog.generate(71)
    model = TransitionModel(catalog)
    before = Observation(catalog, (0, 0, 0))
    with pytest.raises(ValueError):
        Transition(before, "unknown", before)
    with pytest.raises(ValueError):
        Transition(before, catalog.actions[0], Observation(Catalog.generate(72), (0, 0, 0)))
    with pytest.raises(ValueError):
        model.observe({"before": before})
    assert not model.observations and model.simulated_effects == 0
    # No supplied action changes two properties at once.
    model.observe(Transition(before, catalog.actions[0], Observation(catalog, (1, 1, 0))))
    assert not model.valid and model.distribution((0, 0, 0), ()) is None
    assert model.message((0, 0, 0), (catalog.actions[1],))["status"] == "MODEL_CLASS_FAILURE"
    model.reset()
    assert model.valid and model.distribution((0, 0, 0), ()) == (1, 0, 0, 0, 0, 0, 0, 0)


def test_core_import_does_not_load_sampler_language_or_torch():
    code = ("import sys; from autonomous_learner.action_model import TransitionModel; "
            "assert not {'torch','transformers','autonomous_learner.action_sampling',"
            "'autonomous_learner.language'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_scripted_demo_preserves_old_outputs_and_erases_learned_predictions(tmp_path):
    from autonomous_learner.action_demo import run
    output = tmp_path / "demo.json"
    report = run(output)
    assert report["kind"] == "scripted_software_demo_not_benchmark"
    assert report["real_actions"] == 5 and report["remaining_actions"] == 0
    assert report["before"] == report["after_reset"]
    assert report["after_experience"] != report["before"]
    with pytest.raises(FileExistsError):
        run(output)
