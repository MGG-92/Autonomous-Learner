"""Learned input/output connectivity and private checkpoint recovery."""
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "neural-volume-lab"))
    import neural
    import checkpoint
    torch.set_num_threads(2)
    return neural, checkpoint


def model(neural):
    return neural.Learner(neural.Config(side=5, degree=4, slots=3, vocabulary=12,
        input_dimensions=6, input_units=8, output_units=8))


def sequence(ids):
    return {"tokens": [{"id": i, "text": str(i), "vector": [i * .1, -.2, .3, i * -.1, .1, .5],
        "points": [[0, 0, 0]]} for i in ids]}


IDENTITY = {"model": "synthetic-fixture", "dimensions": 6, "revision": "1", "vocabulary_size": 100}


def test_full_input_and_output_connections_learn_without_coordinate_stimulation(modules):
    neural, _ = modules
    learner = model(neural)
    net = learner.network
    assert net.input_adapter.weight.shape == (8, 6)
    assert net.input_to_core.weight.shape == (125, 8)
    assert net.encoder.weight.shape == (8, 125)
    assert net.input_to_core.weight.count_nonzero() == 125 * 8
    token = sequence([1])["tokens"][0]
    changed_points = dict(token, points=[[1, 1, 1], [-1, -1, -1]])
    assert torch.equal(net.stimulus(token), net.stimulus(changed_points))
    different_vector = dict(token, vector=[.5, .4, -.3, .2, -.1, -.5])
    state_a = net.tick(torch.zeros(125), net.stimulus(token))
    state_b = net.tick(torch.zeros(125), net.stimulus(different_vector))
    assert not torch.equal(state_a, state_b)
    with pytest.raises(ValueError, match="preserved spatial"):
        net.pulse(token["points"])
    originals = [layer.weight.detach().clone() for layer in (net.input_adapter, net.input_to_core, net.encoder)]
    learner.admit(sequence([1]), sequence([9]))
    learner.train(5)
    assert all(not torch.equal(old, layer.weight) for old, layer in zip(originals, (net.input_adapter, net.input_to_core, net.encoder)))


def test_dense_interface_feedback_works_and_online_replay_stays_bounded(modules):
    neural, _ = modules
    changes = {}
    for reward in (-1, 1):
        learner = model(neural)
        learner.admit(sequence([1]), sequence([8, 9]))
        _, trace = learner.sample(sequence([1]))
        result = learner.feedback(trace, reward)
        changes[reward] = result["mean_log_probability_after"] - result["mean_log_probability_before"]
    assert changes[1] > 0 and changes[-1] < 0
    learner = model(neural)
    for i in range(1, 9):
        learner.admit(sequence([i]), sequence([9]))
    metadata = list(learner.example_metadata)
    result = learner.train_online(3)
    assert result["online_updates"] == 3 and result["replay_pairs"] == 2
    assert len(learner.examples) == 8 and learner.example_metadata == metadata


def test_checkpoint_restores_weights_optimizer_replay_and_sampling_state(modules, tmp_path):
    neural, checkpoint = modules
    learner = model(neural)
    learner.admit(sequence([1]), sequence([9]))
    learner.train(5)
    store = checkpoint.BrainStore(tmp_path / "brain.pt", IDENTITY)
    store.save(learner)
    store.close()
    expected_ids, _ = learner.sample(sequence([1]))
    restored = model(neural)
    store = checkpoint.BrainStore(tmp_path / "brain.pt", IDENTITY)
    assert store.restore(restored)
    assert restored.updates == learner.updates and restored.vocabulary == learner.vocabulary
    assert len(restored.examples) == 1 and restored.example_metadata == learner.example_metadata
    assert all(torch.equal(value, restored.network.state_dict()[key]) for key, value in learner.network.state_dict().items())
    assert restored.optimizer.state_dict()["param_groups"] == learner.optimizer.state_dict()["param_groups"]
    assert restored.sample(sequence([1]))[0] == expected_ids
    store.close()


def test_previous_checkpoint_recovers_corruption_and_reset_clears_both_generations(modules, tmp_path):
    neural, checkpoint = modules
    learner = model(neural)
    learner.admit(sequence([1]), sequence([9]))
    learner.train(1)
    store = checkpoint.BrainStore(tmp_path / "brain.pt", IDENTITY)
    store.save(learner)
    learner.train(1)
    store.save(learner)
    store.path.write_bytes(b"interrupted-file")
    restored = model(neural)
    assert store.restore(restored) and store.recovered and restored.updates == 1
    store.save(model(neural), reset=True)
    store.path.write_bytes(b"interrupted-file")
    fresh = model(neural)
    assert store.restore(fresh) and fresh.updates == 0 and not fresh.vocabulary
    store.close()


def test_incompatible_identity_and_write_failure_preserve_saved_weights(modules, tmp_path, monkeypatch):
    neural, checkpoint = modules
    learner = model(neural)
    path = tmp_path / "brain.pt"
    store = checkpoint.BrainStore(path, IDENTITY)
    store.save(learner)
    original = path.read_bytes()
    replace = checkpoint.os.replace

    def fail_primary(source, destination):
        if Path(destination) == path:
            raise OSError("synthetic disk failure")
        return replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "replace", fail_primary)
    with pytest.raises(OSError, match="disk failure"):
        store.save(learner)
    assert path.read_bytes() == original
    store.close()
    wrong = checkpoint.BrainStore(path, dict(IDENTITY, revision="wrong"))
    with pytest.raises(RuntimeError, match="embedding source"):
        wrong.restore(model(neural))
    assert path.read_bytes() == original
    wrong.close()


def test_saved_brain_has_one_writer_and_rejects_nonfinite_weights(modules, tmp_path):
    neural, checkpoint = modules
    path = tmp_path / "brain.pt"
    store = checkpoint.BrainStore(path, IDENTITY)
    with pytest.raises(RuntimeError, match="Another app"):
        checkpoint.BrainStore(path, IDENTITY)
    learner = model(neural)
    with torch.no_grad():
        learner.network.decoder.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        store.save(learner)
    assert not path.exists()
    store.close()


def test_engine_autosaves_and_restores_in_a_fresh_process_object(modules, tmp_path, monkeypatch):
    neural, _ = modules
    import worker

    class Store:
        def load(self): pass
        tokenizer = None
        def encode(self, text): return sequence([int(word) for word in text.split()])
        def decode(self, ids): return " ".join(map(str, ids))

    source = Store()
    source.tokenizer = source
    engine = worker.Engine(source, model(neural), checkpoint=tmp_path / "brain.pt", identity=IDENTITY)
    assert engine.handle({"op": "init"})["storage"]["saved"]
    engine.handle({"op": "enqueue", "message": "1"})
    turn = engine.handle({"op": "live_tick"})
    engine.handle({"op": "correct", "turn_id": turn["turn_id"], "target": "9"})
    engine.close()
    resumed = worker.Engine(source, model(neural), checkpoint=tmp_path / "brain.pt", identity=IDENTITY)
    result = resumed.handle({"op": "init"})
    assert result["storage"]["restored"] and result["updates"] == 12
    assert resumed.learner.vocabulary == [9] and not resumed.recent_turns
    assert not resumed.handle({"op": "live_tick"})["reply"]
    resumed.close()
