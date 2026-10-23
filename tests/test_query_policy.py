from dataclasses import replace
import random

import pytest
import torch

from autonomous_learner.query_policy import QueryPolicy, choose_query, query_features
from autonomous_learner.query_training import QueryConfig, MeteredLearner, evaluate_policy, layout, run, train_policy
from autonomous_learner.rule_experiment import scores
from autonomous_learner.rule_induction import Hypothesis, RuleLearner, RuleSpace, build_space


def test_features_are_outcome_permutation_invariant_but_change_with_experience():
    space = build_space()
    learner = RuleLearner(space, space.inputs)
    pairs, before = query_features(learner, 4, 4)
    learner.observe((2, 1), 2)
    pairs, after = query_features(learner, 3, 4)
    assert len(pairs) == 24 and after.shape == (24, 7)
    remaining_prior = before[[pair != (2, 1) for pair in space.inputs], :5]
    assert not torch.equal(remaining_prior, after[:, :5])
    # Relabel output categories consistently in a public prediction-table fixture.
    permuted = RuleSpace(5, space.inputs, tuple(Hypothesis(h.program, tuple((v + 1) % 5 for v in h.values))
                                              for h in space.hypotheses))
    other = RuleLearner(permuted, space.inputs)
    other.observe((2, 1), 3)
    assert torch.equal(after, query_features(other, 3, 4)[1])
    torch.manual_seed(5)
    policy = QueryPolicy()
    assert choose_query(policy, learner, random.Random(3), 3, 4) == choose_query(policy, other, random.Random(3), 3, 4)


def test_learned_query_uses_only_legal_inputs_and_exact_scan_costs():
    space = build_space()
    legal, probes = layout(space, 123, 5)
    learner = MeteredLearner(space, legal)
    policy = QueryPolicy()
    pair = choose_query(policy, learner, random.Random(0), 1, 1)
    assert pair in legal and pair not in probes
    assert learner.model_value_reads == len(legal) * len(space.hypotheses)
    hidden = next(h for h in space.hypotheses if h.partition == "development")
    metrics, trace = evaluate_policy(space, hidden, legal, probes, "learned", policy, 0, 1)
    assert metrics["selection_model_value_reads"] == len(legal) * len(space.hypotheses)
    assert len(trace) == metrics["queries"] == 1 and trace[0][0] not in probes


def test_reset_removes_predictions_without_changing_query_trajectory():
    space = build_space()
    hidden = next(h for h in space.hypotheses if h.partition == "development")
    legal, probes = layout(space, 123, 5)
    policy = QueryPolicy()
    _, active_trace = evaluate_policy(space, hidden, legal, probes, "learned", policy, 21, 4)
    reset, reset_trace = evaluate_policy(space, hidden, legal, probes, "model_reset", policy, 21, 4)
    assert active_trace == reset_trace
    prior = RuleLearner(space, legal)
    truth = [hidden.values[space.inputs.index(pair)] for pair in probes]
    expected = scores([prior.distribution(pair) for pair in probes], truth)
    assert all(reset[metric] == expected[metric] for metric in expected)


def test_training_is_reproducible_changes_weights_and_preserves_initial_network():
    torch.set_num_threads(1)
    config = QueryConfig(episodes=8, worlds=2, optimizer_seeds=[7])
    learned, initial, _ = train_policy(config, 7, build_space())
    again, _, _ = train_policy(config, 7, build_space())
    assert all(torch.equal(a, b) for a, b in zip(learned.parameters(), again.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(learned.parameters(), initial.parameters()))
    torch.manual_seed(7)
    original = QueryPolicy()
    assert all(torch.equal(a, b) for a, b in zip(initial.parameters(), original.parameters()))


def test_runner_is_validation_only_and_records_baselines_without_overwriting(tmp_path):
    torch.set_num_threads(1)
    config = QueryConfig(episodes=2, worlds=2, optimizer_seeds=[7])
    report = run(config, tmp_path / "one")
    assert report["phase"] == "development_validation_only"
    assert {h.partition for h in build_space().hypotheses if h.identity in report["validation_rule_ids"]} == {"validation"}
    assert set(report["results_by_seed"]["7"]) == {"learned", "untrained", "random", "fixed",
                                                        "information_gain", "model_reset", "prior_features"}
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "one")


@pytest.mark.parametrize("changes", [{"episodes": 0}, {"optimizer_seeds": [7, 7]},
                                    {"queries": 25}, {"learning_rate": float("nan")},
                                    {"validation_layout_start": 4_000_001}])
def test_config_rejects_overlap_and_invalid_settings(changes):
    with pytest.raises(ValueError):
        replace(QueryConfig(), **changes).validate()
