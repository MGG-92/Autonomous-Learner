"""Controlled numerical checks, synthetic targets only; no downloaded teacher."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SPEC = importlib.util.spec_from_file_location("volume_neural", Path(__file__).resolve().parents[1] / "neural-volume-lab" / "neural.py")
# dataclasses needs its module registered when resolving annotations.
import sys
neural = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = neural
SPEC.loader.exec_module(neural)


def sequence(ids, points):
    return {"tokens": [{"id": i, "points": [p]} for i, p in zip(ids, points)]}


def learner():
    return neural.Learner(neural.Config(side=5, degree=4, hidden=24, slots=3, vocabulary=12))


def test_topology_is_directed_without_self_or_reciprocal_edges_and_is_seeded():
    a, b = learner(), learner()
    graph = {(int(source), dest) for dest, sources in enumerate(a.network.sources) for source in sources}
    assert all(source != dest and (dest, source) not in graph for source, dest in graph)
    assert len(graph) == 125 * 4
    assert torch.equal(a.network.edges, b.network.edges)
    assert torch.count_nonzero(a.network.edges) > 0
    assert not torch.count_nonzero(a.network.activity)


def test_only_vertices_stimulate_and_exact_zero_stays_zero():
    net = learner().network
    pulse = net.pulse([[-1, -1, -1], [1, 1, 1]])
    assert pulse.nonzero().flatten().tolist() == [0, 124]
    assert pulse.sum() == 2  # Connecting segment does not stimulate intermediate units.
    assert not net.tick(torch.zeros(125)).count_nonzero()
    state = net.tick(torch.zeros(125), pulse)
    assert state.count_nonzero() == 2
    net.constrain()
    for _ in range(100):
        state = net.tick(state)
    assert state.max() < 1e-5
    with pytest.raises(ValueError):
        net.pulse([[float("nan"), 0, 0]])


def test_backprop_fits_noncopy_pairs_replays_old_pair_and_inference_never_reads_replay():
    torch.set_num_threads(2)
    model = learner()
    a = sequence([1], [[-.5, 0, 0]])
    b = sequence([2], [[.5, 0, 0]])
    target_a = sequence([8, 9], [[0, 0, 0]] * 2)
    target_b = sequence([10], [[0, 0, 0]])
    assert model.predict(a) == []
    original = model.network.edges.detach().clone()
    model.admit(a, target_a)
    result = model.train(100)
    assert result["loss"] < result["initial_loss"] / 10
    assert model.predict(a) == [8, 9]
    model.admit(b, target_b)
    model.train(100)
    assert model.predict(a) == [8, 9]
    assert model.predict(b) == [10]
    assert not torch.equal(original, model.network.edges)
    assert model.network.edges.abs().sum(-1).max() <= .700001
    model.examples.clear()
    assert model.predict(a) == [8, 9]
    assert model.predict(b) == [10]
    # Silencing state is independent of fitted parameters.
    model.network.activity.zero_()
    assert model.predict(a) == [8, 9]


def test_temporal_encoding_preserves_token_order_but_vertex_order_is_lost():
    model = learner()
    a = sequence([1, 2], [[-.5, 0, 0], [.5, 0, 0]])
    b = sequence([2, 1], [[.5, 0, 0], [-.5, 0, 0]])
    _, state_a = model.network(model.encode(a))
    _, state_b = model.network(model.encode(b))
    assert not torch.equal(state_a, state_b)
    points = [[-.5, 0, 0], [.5, 0, 0]]
    assert torch.equal(model.network.pulse(points), model.network.pulse(points[::-1]))


def test_rest_uses_perturbations_and_task_replay_without_erasing_readout():
    model = learner()
    message = sequence([1], [[0, 0, 0]])
    model.admit(message, sequence([8], [[0, 0, 0]]))
    model.train(40)
    before = model.network.edges.detach().clone()
    result = model.rest(10)
    assert np.isfinite(result["rest_loss"]) and result["initial_rest_loss"] > 0
    assert not torch.equal(before, model.network.edges)
    assert model.predict(message) == [8]


def test_limits_are_atomic_and_replay_is_bounded():
    model = learner()
    message = sequence([1], [[0, 0, 0]])
    with pytest.raises(ValueError):
        model.admit(message, {"tokens": []})
    assert not model.vocabulary and not model.examples
    for _ in range(10):
        model.admit(message, message)
    assert len(model.examples) == model.config.replay
    with pytest.raises(ValueError):
        model.train(101)
