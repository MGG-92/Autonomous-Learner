import pytest
import torch

from autonomous_learner.agent import Agent, run_episode
from autonomous_learner.evaluation import evaluate
from autonomous_learner.memory import Memory
from autonomous_learner.policy import Controller, Heuristic, encode
from autonomous_learner.training import TrainConfig, expert_index, load_checkpoint, save_checkpoint, train
from autonomous_learner.world import SyntheticTeacher, World


def test_generated_worlds_and_expert_proofs():
    for seed in range(12):
        world = World.generate(seed)
        assert world == World.generate(seed)
        memory = Memory()
        goal = world.goal()
        agent = Agent(memory, world.catalog(), goal)
        teacher = SyntheticTeacher(world)
        assert memory.answer(goal).value is None
        for step in range(4):
            action = agent.actions[expert_index(agent, world)]
            if action.kind == "ANSWER":
                break
            agent.execute(action, teacher)
        assert memory.answer(goal).value == world.truth(goal)
        assert teacher.calls == 3
        memory.close()


def test_features_cannot_see_hidden_polarity():
    world = World.generate(20)
    agent = Agent(Memory(), world.catalog(), world.goal())
    before = agent.features(0)
    # Encoder has no oracle/world argument; changing hidden truth changes no feature.
    world.facts = tuple(type(f)(f.subject, f.relation, f.object, not f.positive)
                        if f.relation == "can" else f for f in world.facts)
    assert torch.equal(before, agent.features(0))
    agent.memory.close()


def test_budgets_and_no_memory_ablation():
    metrics = evaluate(Heuristic(), 10_000, 20)
    assert metrics["accuracy"] == 1.0
    assert metrics["mean_steps"] <= 8
    assert evaluate(Heuristic(), 10_000, 20, capacity=0)["accuracy"] == 0


def test_disjoint_splits_and_invalid_config():
    with pytest.raises(ValueError, match="disjoint"):
        TrainConfig(test_start=0).validate()
    with pytest.raises(ValueError):
        TrainConfig(epochs=0).validate()


def test_optimizer_improves_held_out_policy_and_checkpoint(tmp_path):
    config = TrainConfig(imitation_worlds=40, epochs=24, rl_episodes=0,
                         validation_worlds=12, test_worlds=20)
    trained, initial, curve = train(config)
    before = evaluate(initial, config.test_start, config.test_worlds)
    after = evaluate(trained, config.test_start, config.test_worlds)
    assert after["accuracy"] >= 0.9
    assert after["mean_reward"] > before["mean_reward"] + 0.3
    assert curve[-1]["loss"] < curve[1]["loss"]
    path = tmp_path / "controller.pt"
    save_checkpoint(path, trained, config)
    loaded, restored_config = load_checkpoint(path)
    assert restored_config == config
    assert evaluate(loaded, config.test_start, config.test_worlds)["accuracy"] == after["accuracy"]
