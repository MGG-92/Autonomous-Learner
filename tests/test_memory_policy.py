import torch

from autonomous_learner.memory_policy import DualMemory, evaluate_retention, retention_stream, train_retention


def test_dual_store_retention_restart_and_fixed_baselines(tmp_path):
    world, goal, proof, stream = retention_stream(200_014)
    path = tmp_path / "dual"
    memory = DualMemory(goal, "heuristic", path)
    for observation in stream:
        memory.store(observation)
    assert len(memory.episodes.inspect()) <= 4
    assert len(memory.beliefs.inspect()) == 3
    before = memory.answer(goal)
    assert before.value == world.truth(goal)
    memory.close()
    memory = DualMemory(goal, "heuristic", path)
    assert memory.answer(goal) == before
    memory.close()
    assert evaluate_retention("fifo", 200_000, 10)["retention_accuracy"] == 0


def test_learned_retention_improves_over_fifo():
    scorer, initial, losses = train_retention(worlds=12, epochs=8)
    result = evaluate_retention(scorer, 210_000, 10)
    assert result["retention_accuracy"] >= 0.9
    assert losses[-1] < losses[0]
