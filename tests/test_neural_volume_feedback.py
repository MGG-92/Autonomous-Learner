"""Actual immediate reward updates and real tick/credit boundaries, no model download."""
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "neural-volume-lab"))
    import neural
    import worker
    torch.set_num_threads(2)
    return neural, worker


def sequence(ids):
    return {"text": " ".join(map(str, ids)), "tokens": [
        {"id": i, "text": str(i), "points": [[(i % 3 - 1) * .5, 0, 0]]} for i in ids]}


def learner(neural, seed=42):
    return neural.Learner(neural.Config(side=5, degree=4, hidden=24, slots=3, vocabulary=12, seed=seed, encoding="spatial"))


def test_positive_negative_and_zero_reward_change_only_the_sampled_reply_probability(modules):
    neural, _ = modules
    values = {}
    for reward in (-1, 0, 1):
        model = learner(neural)
        model.admit(sequence([1]), sequence([8, 9]))
        ids, trace = model.sample(sequence([1]))
        replay_before = len(model.examples)
        result = model.feedback(trace, reward)
        values[reward] = result["mean_log_probability_after"] - result["mean_log_probability_before"]
        assert len(model.examples) == replay_before  # No hidden positive/negative target label.
        with pytest.raises(ValueError, match="already rated"):
            model.feedback(trace, reward)
    assert values[1] > 0 and values[-1] < 0 and values[0] == 0


def test_human_feedback_strength_and_source_have_real_effect(modules):
    neural, _ = modules
    changes = {}
    for source, strength in [("human", 1), ("human", 5), ("professor", 1)]:
        model = learner(neural)
        model.admit(sequence([1]), sequence([8]))
        _, trace = model.sample(sequence([1]))
        result = model.feedback(trace, 1, source, strength=strength)
        changes[source, strength] = result["mean_log_probability_after"] - result["mean_log_probability_before"]
    assert changes["human", 5] > changes["human", 1] > changes["professor", 1] > 0


@pytest.fixture
def engine(modules):
    neural, worker = modules

    class Store:
        def load(self): pass
        tokenizer = None

        def encode(self, text):
            return sequence([int(word) for word in text.split()])

        def decode(self, ids):
            return " ".join(map(str, ids))

    store = Store()
    store.tokenizer = store

    class Professor:
        def assess(self, message, answer):
            return {"proposal": "8", "reward": -1 if answer else 0, "verdict": "Synthetic vote",
                    "source": "synthetic fixture", "relative_support": .1 if answer else None}

    return worker.Engine(store, learner(neural), Professor())


def test_queue_consumes_one_token_per_tick_and_does_not_peek_at_future_input(engine):
    engine.handle({"op": "enqueue", "message": "1 2 3"})
    assert engine.learner.network.activity.count_nonzero() == 0
    result = engine.handle({"op": "live_tick"})
    assert result["received"] == 1 and result["token"]["id"] == 1
    assert result["reply"] is False and engine.last_turn is None
    pulse = engine.learner.network.pulse(sequence([1])["tokens"][0]["points"])
    assert torch.equal(engine.learner.network.activity, .85 * pulse)
    assert result["input_finished"] is False
    with pytest.raises(ValueError, match="Finish"):
        engine.handle({"op": "enqueue", "message": "2"})
    assert engine.handle({"op": "live_tick"})["token"]["id"] == 2
    final = engine.handle({"op": "live_tick"})
    assert final["token"]["id"] == 3 and final["input_finished"]
    assert final["reply"] and engine.last_turn["message"]["text"] == "1 2 3"
    undriven = engine.handle({"op": "live_tick"})
    assert undriven["token"] is None and undriven["received"] == 3
    assert not undriven["reply"] and engine.last_turn["id"] == final["turn_id"]


def test_feedback_targets_the_exact_response_and_recent_credit_really_expires(engine):
    engine.handle({"op": "enqueue", "message": "1"})
    first = engine.handle({"op": "live_tick"})
    engine.handle({"op": "speak"})
    result = engine.handle({"op": "feedback", "turn_id": first["turn_id"], "reward": 1})
    assert result["turn_id"] == first["turn_id"]
    for _ in range(8):
        engine.handle({"op": "speak"})
    assert len(engine.recent_turns) == 8
    with pytest.raises(ValueError, match="eight messages"):
        engine.handle({"op": "feedback", "turn_id": first["turn_id"], "reward": -1})


def test_correction_uses_live_snapshot_and_human_example_overrides_professor(engine):
    engine.handle({"op": "enqueue", "message": "1"})
    turn = engine.handle({"op": "live_tick"})["turn_id"]
    engine.handle({"op": "coach", "turn_id": turn})
    before = engine.learner.updates
    result = engine.handle({"op": "correct", "turn_id": turn, "target": "9"})
    assert result["kind"] == "supervised_correction" and engine.learner.updates == before + 12
    assert len(engine.learner.examples) == 1
    context, labels = engine.learner.examples[0]
    assert isinstance(context, dict)
    logits, _ = engine.learner.evaluate(context)
    indices = logits[:, :len(engine.learner.vocabulary)+1].argmax(-1).tolist()
    assert engine.learner.vocabulary[indices[0] - 1] == 9 and indices[1] == 0
    assert not engine.learner.admit(sequence([1]), sequence([8]), source="professor", replace=True)
    with pytest.raises(ValueError, match="already"):
        engine.handle({"op": "correct", "turn_id": turn, "target": "8"})


def test_partial_input_and_reset_do_not_leak_future_targets_or_stale_feedback(engine):
    engine.handle({"op": "enqueue", "message": "1 2"})
    partial = engine.handle({"op": "live_tick"})
    assert not partial["reply"]
    with pytest.raises(ValueError):
        engine.handle({"op": "correct", "turn_id": None, "target": "9"})
    turn = engine.handle({"op": "live_tick"})["turn_id"]
    engine.handle({"op": "correct", "turn_id": turn, "target": "9"})
    assert engine.learner.example_metadata[0][0] == (1, 2)
    engine.handle({"op": "reset"})
    assert not engine.recent_turns and engine.incoming is None
    with pytest.raises(ValueError):
        engine.handle({"op": "feedback", "turn_id": turn, "reward": 1})
