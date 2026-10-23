from dataclasses import replace
import json
import random

import pytest
import torch

from autonomous_learner.delayed_memory import (DelayedMemory, MemoryController, PublicTask, Record,
                                              choose, encode, visible_proofs)
from autonomous_learner.delayed_training import (DelayedConfig, episode, evaluate, load_checkpoint,
                                                offline_bound, paired_interval, save_checkpoint, train)
from autonomous_learner.memory import Memory
from autonomous_learner.types import Fact, Observation, Query
from autonomous_learner.world import World


def small_config(**kwargs):
    return DelayedConfig(optimizer_seeds=[42], train_start=1_100_000, imitation_worlds=8,
                         imitation_epochs=4, rl_episodes=2, validation_start=1_200_000,
                         validation_worlds=4, test_start=1_300_000, test_worlds=4, **kwargs)


def test_expiration_forgetting_write_mask_and_restart(tmp_path):
    task = PublicTask(("entity",), ("ability",), horizon=7, max_consolidations=2, belief_capacity=3)
    memory = DelayedMemory(task)
    memory.advance(Fact("entity", "is_a", "leaf"))
    memory.consolidate([0])
    memory.advance(Fact("leaf", "is_a", "root"))
    memory.advance(Fact("root", "can", "ability", False))
    memory.consolidate([0, 1, 2])
    memory.advance(None)
    memory.advance(None)
    assert not memory.episodes
    query = Query("entity", "can", "ability")
    assert visible_proofs(memory.beliefs, task)[query][0] is False
    path = tmp_path / "snapshot.json"
    memory.save(path)
    restored = DelayedMemory.load(path)
    assert restored.visible() == memory.visible()
    assert restored.tick == memory.tick and restored.consolidations == 2
    restored.advance(Fact("other", "can", "ability"))
    assert choose(restored, "graph")[0] is None
    with pytest.raises(ValueError, match="masked"):
        restored.consolidate([0])
    # Evidence that was never consolidated really disappears.
    empty = DelayedMemory(task)
    empty.advance(Fact("entity", "can", "ability"))
    empty.advance(None)
    empty.advance(None)
    assert not empty.visible() and not visible_proofs(empty.beliefs, task)


def test_reobservation_and_invalid_snapshot(tmp_path):
    memory = DelayedMemory(PublicTask(("a",), ("x",)))
    memory.advance(Fact("a", "can", "x"))
    memory.advance(Fact("a", "can", "x"))
    assert len(memory.episodes) == 1
    path = tmp_path / "state.json"
    memory.save(path)
    assert DelayedMemory.load(path).visible() == memory.visible()
    data = json.loads(path.read_text())
    data["consolidations"] = 100
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resource"):
        DelayedMemory.load(path)


def test_public_features_cannot_see_hidden_truth_or_suffix_or_future_query():
    world = World.generate(17)
    memory = DelayedMemory(PublicTask(world.entities, world.abilities))
    for fact in world.facts[:3]:
        memory.advance(fact)
    before = encode(memory)
    world.facts = tuple(Fact(f.subject, f.relation, f.object, not f.positive) for f in reversed(world.facts))
    # No world reference or selected future goal is held by memory/feature encoding.
    after = encode(memory)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert set(vars(memory)) == {"task", "episodes", "beliefs", "tick", "consolidations"}


def test_benchmark_never_uses_the_specific_future_goal(monkeypatch):
    monkeypatch.setattr(World, "goal", lambda self: (_ for _ in ()).throw(AssertionError("future goal leaked")))
    result, _, _ = episode("fifo", 1_300_040, small_config(), capture=True)
    assert result["capacity_violations"] == 0
    assert len(result["trace"]) == 17


def test_visible_inference_matches_existing_engine_on_random_subsets():
    world = World.generate(41)
    task = PublicTask(world.entities, world.abilities)
    rng = random.Random(7)
    for _ in range(15):
        facts = rng.sample(world.facts, rng.randrange(16))
        records = [Record(f, i) for i, f in enumerate(facts)]
        answers = visible_proofs(records, task)
        reference = Memory()
        reference.store(Observation(Query("reference", "test"), tuple(facts), "test"))
        for query in task.queries():
            assert (answers[query][0] if query in answers else None) == reference.answer(query).value
        reference.close()


def test_all_operational_policies_respect_identical_hard_budgets():
    config = small_config()
    for policy in (MemoryController(), "fifo", "random", "graph", "graph_scheduled"):
        metrics, rows = evaluate(policy, 1_300_050, 4, config)
        assert metrics["max_episodic_size"] <= 2 and metrics["max_belief_size"] <= 6
        assert metrics["capacity_violations"] == 0
        assert all(row["writes"] <= 6 for row in rows)
        assert all(row["utility"] == pytest.approx(row["accuracy"] - 0.01 * row["writes"]) for row in rows)
    no_memory = replace(config, belief_capacity=0)
    metrics, _ = evaluate("graph", 1_300_050, 4, no_memory)
    assert metrics["accuracy"] == metrics["writes"] == 0


def test_offline_storage_bound_is_below_perfect_recall():
    world = World.generate(4)
    assert offline_bound(world, 0) == 0
    assert offline_bound(world, 6) == pytest.approx(1 / 3)
    assert offline_bound(world, 15) == 1.0


def test_training_loss_checkpoint_and_masked_gradient(tmp_path):
    config = small_config()
    trained, initial, curve, selected = train(config, 42)
    imitation = [row for row in curve if row["stage"] == "imitation"]
    assert imitation[-1]["loss"] < imitation[0]["loss"]
    assert all(torch.isfinite(p).all() for p in trained.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained, config, selected)
    restored, restored_config = load_checkpoint(path)
    assert restored_config == config
    assert evaluate(restored, 1_300_070, 4, config)[0] == evaluate(trained, 1_300_070, 4, config)[0]


def test_paired_interval_averages_optimizer_seeds_within_world():
    rows = []
    for world in range(4):
        for seed in (7, 42):
            for name, utility in (("learned", 0.2), ("fifo", 0.1)):
                rows.append({"world_seed": world, "optimizer_seed": seed, "policy": name, "utility": utility})
    result = paired_interval(rows, "fifo", samples=100)
    assert result["mean_paired_utility_difference"] == pytest.approx(0.1)
    assert result["world_bootstrap95"] == pytest.approx([0.1, 0.1])


def test_invalid_configs_and_inaccessible_evidence():
    with pytest.raises(ValueError, match="disjoint"):
        replace(small_config(), test_start=1_100_000).validate()
    with pytest.raises(ValueError, match="expire"):
        replace(small_config(), quiet_ticks=1).validate()
    memory = DelayedMemory(PublicTask(("a",), ("x",), horizon=1))
    memory.advance(Fact("a", "can", "x"))
    with pytest.raises(ValueError, match="accessible"):
        memory.consolidate([-1])
    with pytest.raises(ValueError, match="tick budget"):
        memory.advance(None)
