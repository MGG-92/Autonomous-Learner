from dataclasses import replace
import json
import random
import subprocess
import sys

import pytest

from autonomous_learner.rule_experiment import RuleConfig, investigate, run
from autonomous_learner.rule_induction import RuleLearner, build_space, execute, reference_signatures


def test_executable_rules_match_behavior_and_semantic_splits_are_disjoint():
    space = build_space()
    assert len({h.values for h in space.hypotheses}) == len(space.hypotheses)
    groups = {}
    for hypothesis in space.hypotheses:
        assert tuple(execute(hypothesis.program, *pair, space.modulus) for pair in space.inputs) == hypothesis.values
        groups.setdefault(hypothesis.partition, set()).add(hypothesis.values)
    assert set(groups) == {"development", "validation", "final"}
    assert all(not groups[a] & groups[b] for a in groups for b in groups if a != b)
    assert reference_signatures() <= groups["development"]


def test_experience_changes_predictions_and_reset_removes_the_change():
    space = build_space()
    learner = RuleLearner(space, space.inputs)
    before = learner.distribution((2, 3))
    learner.observe((2, 3), 3)
    assert len(learner.candidates) < len(space.hypotheses)
    assert learner.distribution((2, 3)) == (0, 0, 0, 1, 0)
    learner.reset_model()
    assert not learner.observations
    assert learner.distribution((2, 3)) == before


def test_unfamiliar_operation_discovery_predicts_unobserved_inputs():
    space = build_space()
    heldout = ((0, 4), (4, 0), (3, 4), (1, 4), (2, 4))
    legal = tuple(pair for pair in space.inputs if pair not in heldout)
    learner = RuleLearner(space, legal)
    calls = []
    def oracle(pair):
        assert pair in legal
        calls.append(pair)
        return max(pair)
    trace = investigate(learner, oracle, "information_gain", 42, 4)
    assert len(trace) == len(calls) <= 4 and len(set(calls)) == len(calls)
    assert learner.observations == {pair: max(pair) for pair in calls}
    for pair in heldout:
        probabilities = learner.distribution(pair)
        assert max(range(5), key=probabilities.__getitem__) == max(pair)
        assert 0 < probabilities[max(pair)] < 1
    # Four examples leave several possible rules: correct predictions do not
    # justify claiming certainty or unique rule identification.
    assert len(learner.candidates) > 1 and learner.message(heldout[0])["representative"] is None
    assert all(row["hypotheses_after"] <= row["hypotheses_before"] for row in trace)


def test_information_gain_targets_disagreement_not_repeated_observations():
    space = build_space()
    learner = RuleLearner(space, space.inputs)
    # Restrict to two internally represented possibilities: x versus y.
    learner.candidates = tuple(i for i, h in enumerate(space.hypotheses) if h.program.op in {"x", "y"})
    pair = learner.choose("information_gain", random.Random(7))
    assert pair[0] != pair[1]
    learner.observe(pair, pair[0])
    assert len(learner.candidates) == 1
    assert learner.distribution((4, 2))[4] == 1


def test_out_of_class_evidence_is_explicit_and_illegal_input_does_not_mutate():
    space = build_space()
    learner = RuleLearner(space, space.inputs)
    with pytest.raises(ValueError):
        learner.observe((9, 9), 1)
    assert len(learner.candidates) == len(space.hypotheses) and not learner.observations
    rng = random.Random(901)
    # An arbitrary random table is outside the supplied bounded program class.
    for pair in space.inputs:
        learner.observe(pair, rng.randrange(5))
    assert not learner.candidates
    assert learner.distribution((0, 0)) is None
    assert learner.message((0, 0))["status"] == "MODEL_CLASS_FAILURE"


def test_core_runs_without_torch_transformers_or_language_modules():
    code = ("import sys; from autonomous_learner.rule_induction import build_space, RuleLearner; "
            "from autonomous_learner.rule_experiment import investigate; "
            "assert not {'torch','transformers','autonomous_learner.language'} & set(sys.modules); "
            "s=build_space(); assert RuleLearner(s,s.inputs).distribution((2,1))")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_reproducible_evaluation_uses_paired_unqueried_inputs(tmp_path):
    config = RuleConfig(worlds=3, query_seeds=[42])
    first = run(config, tmp_path / "first")
    second = run(config, tmp_path / "second")
    assert first["summary"] == second["summary"]
    assert first["per_rule"] == second["per_rule"]
    assert len(first["per_rule"]) == 3 * 5
    for metric in ("accuracy", "brier", "log_loss"):
        assert first["summary"]["model_reset"][metric] == first["summary"]["no_memory"][metric]
    assert all(row["queries"] <= config.queries for row in first["per_rule"])
    assert all(h.partition == "validation" for h in build_space().hypotheses
               if h.identity in {r["rule_id"] for r in first["per_rule"]})
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "first")
    assert json.loads((tmp_path / "first/report.json").read_text())["phase"] == "validation"


@pytest.mark.parametrize("config", [RuleConfig(queries=0), RuleConfig(worlds=0), RuleConfig(heldout_inputs=25),
                                   RuleConfig(query_seeds=[True]), RuleConfig(query_seeds=[7, 7]),
                                   RuleConfig(validation_layout_start=3_300_000)])
def test_bad_experiment_settings_fail_before_observation(config):
    with pytest.raises(ValueError):
        config.validate()
