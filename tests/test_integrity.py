"""Regression cases found in the critic audit; no benchmark/test-world tuning."""
import json
from dataclasses import replace
import sqlite3

import pytest
import torch

from autonomous_learner.agent import Agent
from autonomous_learner.delayed_memory import DelayedMemory, PublicTask
from autonomous_learner.delayed_training import paired_interval
from autonomous_learner.delayed_training import DelayedConfig
from autonomous_learner.loop import PersistentLoop
from autonomous_learner.memory import Memory
from autonomous_learner.policy import Controller, Heuristic
from autonomous_learner.types import Fact, Observation, Query
from autonomous_learner.training import TrainConfig
from autonomous_learner.world import SyntheticTeacher, World


def test_incompatible_open_never_trims_evidence(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory = Memory(path, capacity=3)
    memory.store(Observation(Query("a", "can", "x"), (Fact("a", "can", "x"),), "test"))
    memory.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="capacity"):
        Memory(path, capacity=0)
    assert path.read_bytes() == before
    assert len(Memory.inspect_file(path)) == 1


def test_session_rejects_replaced_memory_and_changed_weights(tmp_path):
    world = World.generate(1_300_400)
    session = tmp_path / "session.json"
    memory = Memory(tmp_path / "memory.sqlite")
    policy = Controller().eval()
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), policy)
    loop.tick(SyntheticTeacher(world))
    with torch.no_grad():
        next(policy.parameters()).add_(1)
    with pytest.raises(ValueError, match="differ"):
        PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), policy)
    memory.close()

    # A task-matched session must not accept a replacement empty database.
    session = tmp_path / "heuristic.json"
    memory = Memory(tmp_path / "heuristic.sqlite")
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    loop.tick(SyntheticTeacher(world))
    empty = Memory()
    with pytest.raises(ValueError, match="memory"):
        PersistentLoop(session, Agent(empty, world.catalog(), world.goal()), Heuristic())
    memory.close()
    empty.close()


@pytest.mark.parametrize("field,value", [("ticks", -1), ("ticks", True), ("questions", 99),
                                          ("finished", "false"), ("last_tick_utc", float("nan"))])
def test_session_rejects_invalid_counters(tmp_path, field, value):
    world = World.generate(1_300_401)
    session = tmp_path / "session.json"
    memory = Memory()
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    loop.tick(SyntheticTeacher(world))
    state = json.loads(session.read_text())
    state[field] = value
    session.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="state"):
        PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    memory.close()


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_fact_polarity_is_always_a_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        Fact("a", "can", "x", value)


def test_snapshot_checks_all_records_even_when_deduplication_hides_one(tmp_path):
    memory = DelayedMemory(PublicTask(("a",), ("x",)))
    memory.advance(Fact("a", "can", "x"))
    memory.consolidate([0])
    path = tmp_path / "memory.json"
    memory.save(path)
    data = json.loads(path.read_text())
    data["beliefs"][0]["observed_tick"] = 999  # Same key in episodes used to mask this record.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resource"):
        DelayedMemory.load(path)


def paired_rows():
    return [{"world_seed": world, "optimizer_seed": seed, "policy": policy, "utility": utility}
            for world in (1, 2) for seed in (7, 42)
            for policy, utility in (("learned", 0.2), ("fifo", 0.1))]


@pytest.mark.parametrize("defect", ["duplicate", "missing_pair", "missing_seed", "nonfinite", "empty"])
def test_paired_statistics_reject_invalid_measurements(defect):
    rows = paired_rows()
    if defect == "duplicate":
        rows.append(dict(rows[0], utility=0.9))
    elif defect == "missing_pair":
        rows.pop()
    elif defect == "missing_seed":
        rows = [r for r in rows if not (r["world_seed"] == 2 and r["optimizer_seed"] == 42)]
    elif defect == "nonfinite":
        rows[0]["utility"] = float("nan")
    else:
        rows = []
    with pytest.raises(ValueError):
        paired_interval(rows, "fifo", samples=100)


def test_interrupted_commit_never_repeats_teacher_call(tmp_path, monkeypatch):
    world = World.generate(1_300_402)
    path = tmp_path / "session.json"
    memory = Memory(tmp_path / "memory.sqlite")
    loop = PersistentLoop(path, Agent(memory, world.catalog(), world.goal()), Heuristic())
    save = loop._save
    def fail_final_save():
        if loop.state["pending_action"] is None:
            raise OSError("simulated crash after evidence commit")
        save()
    monkeypatch.setattr(loop, "_save", fail_final_save)
    teacher = SyntheticTeacher(world)
    with pytest.raises(OSError, match="simulated"):
        loop.tick(teacher)
    assert teacher.calls == 1 and len(memory.inspect()) == 1
    assert json.loads(path.read_text())["questions"] == 1
    with pytest.raises(ValueError, match="interrupted"):
        loop.tick(teacher)
    with pytest.raises(ValueError, match="interrupted"):
        PersistentLoop(path, Agent(memory, world.catalog(), world.goal()), Heuristic())
    assert teacher.calls == 1
    memory.close()


def test_teacher_error_reserves_budget_without_admitting_wrong_evidence(tmp_path):
    class WrongTeacher:
        calls = 0
        def ask(self, query):
            self.calls += 1
            return Observation(query, (Fact("unrelated", "can", "x"),), "test")
    world = World.generate(1_300_403)
    memory = Memory()
    agent = Agent(memory, world.catalog(), world.goal())
    loop = PersistentLoop(tmp_path / "session.json", agent, Heuristic())
    teacher = WrongTeacher()
    with pytest.raises(ValueError, match="requested query"):
        loop.tick(teacher)
    assert not agent.asked and not memory.inspect()
    assert loop.state["questions"] == teacher.calls == 1
    with pytest.raises(ValueError, match="interrupted"):
        loop.tick(teacher)
    memory.close()


def test_teacher_type_cannot_change_mid_session(tmp_path):
    from autonomous_learner.teachers import HumanTeacher
    world = World.generate(1_300_404)
    memory = Memory()
    session = tmp_path / "session.json"
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    loop.tick(SyntheticTeacher(world))
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    teacher = HumanTeacher(read=lambda _: pytest.fail("teacher was contacted"), write=lambda _: None)
    with pytest.raises(ValueError, match="teacher type"):
        loop.tick(teacher)
    memory.close()


def test_dual_memory_signature_includes_retention_weights(tmp_path):
    from autonomous_learner.memory_policy import DualMemory, RetentionController
    world = World.generate(1_300_405)
    scorer = RetentionController().eval()
    memory = DualMemory(world.goal(), scorer, tmp_path / "dual")
    session = tmp_path / "session.json"
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    loop.tick(SyntheticTeacher(world))
    before = memory.session_snapshot()
    with torch.no_grad():
        next(scorer.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="differ"):
        PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    assert memory.session_snapshot() == before
    memory.close()


def test_store_rolls_back_entire_observation_on_database_error():
    memory = Memory(capacity=2)
    memory.db.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON evidence
                         WHEN NEW.subject = 'second' BEGIN SELECT RAISE(ABORT, 'simulated'); END""")
    observation = Observation(Query("test", "test"), (Fact("first", "can", "x"),
                                                      Fact("second", "can", "x")), "test")
    with pytest.raises(sqlite3.IntegrityError, match="simulated"):
        memory.store(observation)
    assert memory.inspect() == []
    memory.close()


@pytest.mark.parametrize("config", [replace(DelayedConfig(), learning_rate=float("nan")),
                                   replace(DelayedConfig(), entropy_bonus=float("inf")),
                                   replace(DelayedConfig(), belief_capacity=True),
                                   replace(TrainConfig(), learning_rate=float("nan")),
                                   replace(TrainConfig(), epochs=1.5)])
def test_invalid_configuration_never_reaches_optimizer(config):
    with pytest.raises(ValueError):
        config.validate()


def test_legacy_session_is_preserved_and_refused(tmp_path):
    path = tmp_path / "legacy.json"
    original = '{"ticks": 1, "questions": 1}'
    path.write_text(original)
    world = World.generate(1_300_406)
    memory = Memory()
    with pytest.raises(ValueError, match="legacy"):
        PersistentLoop(path, Agent(memory, world.catalog(), world.goal()), Heuristic())
    assert path.read_text() == original
    memory.close()


def test_stats_preserve_valid_frozen_report_comparisons():
    from pathlib import Path
    import csv
    root = Path(__file__).parents[1] / "reports/delayed-robustness-v1"
    report = json.loads((root / "report.json").read_text())
    # Test the published data's pairing contract without reopening any worlds.
    with (root / "reference.csv").open() as file:
        rows = [{**row, "utility": float(row["utility"])} for row in csv.DictReader(file)]
    actual = paired_interval(rows, "graph_scheduled")
    expected = report["paired_comparisons"]["reference"]["graph_scheduled"]
    assert actual["mean_paired_utility_difference"] == pytest.approx(expected["mean_paired_utility_difference"], abs=1e-12)
    assert actual["world_bootstrap95"] == pytest.approx(expected["world_bootstrap95"], abs=1e-12)
    assert actual["bootstrap_samples"] == expected["bootstrap_samples"]


def test_cli_missing_database_is_not_silently_recreated(tmp_path, monkeypatch):
    from autonomous_learner.cli import main
    import sys
    world = World.generate(900_001)
    session = tmp_path / "session.json"
    database = session.with_suffix(".sqlite")
    memory = Memory(database)
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    loop.tick(SyntheticTeacher(world))
    memory.close()
    preserved = tmp_path / "preserved.sqlite"
    database.rename(preserved)
    before = preserved.read_bytes()
    monkeypatch.setattr(sys, "argv", ["autonomous-learner", "run", "--session", str(session)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not database.exists() and preserved.read_bytes() == before


def test_same_weights_from_a_new_instance_can_resume(tmp_path):
    world = World.generate(1_300_407)
    memory = Memory(tmp_path / "memory.sqlite")
    session = tmp_path / "session.json"
    policy = Controller().eval()
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), policy)
    loop.tick(SyntheticTeacher(world))
    restored = Controller().eval()
    restored.load_state_dict(policy.state_dict())
    resumed = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), restored)
    assert resumed.state == loop.state
    memory.close()
