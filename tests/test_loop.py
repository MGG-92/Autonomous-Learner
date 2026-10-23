import json

import pytest

from autonomous_learner.agent import Agent
from autonomous_learner.loop import PersistentLoop
from autonomous_learner.memory import Memory
from autonomous_learner.policy import Heuristic
from autonomous_learner.teachers import HumanTeacher
from autonomous_learner.types import Query
from autonomous_learner.world import SyntheticTeacher, World


def test_ticks_resume_and_stop_without_extra_teacher_calls(tmp_path):
    world = World.generate(90)
    memory = Memory(tmp_path / "memory.sqlite")
    teacher = SyntheticTeacher(world)
    session = tmp_path / "session.json"
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    assert len(list(loop.run(teacher, 2))) == 2
    memory.close()
    memory = Memory(tmp_path / "memory.sqlite")
    resumed = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic())
    events = list(resumed.run(teacher, 8))
    assert events[-1]["action"] == "ANSWER"
    assert memory.answer(world.goal()).value == world.truth(world.goal())
    calls = teacher.calls
    assert resumed.tick(teacher)["action"] == "IDLE"
    assert teacher.calls == calls == 4
    assert json.loads(session.read_text())["ticks"] == 5
    memory.close()


def test_teacher_budget_and_task_identity(tmp_path):
    world = World.generate(91)
    memory = Memory()
    session = tmp_path / "session.json"
    teacher = SyntheticTeacher(world)
    loop = PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic(), max_questions=0)
    assert list(loop.run(teacher, 8))[0]["reason"] == "teacher budget exhausted"
    assert teacher.calls == 0
    with pytest.raises(ValueError, match="differ"):
        PersistentLoop(session, Agent(memory, world.catalog(), world.goal()), Heuristic(), max_questions=4)
    memory.close()


def test_human_teacher_rejects_nonboolean_and_mismatched_answers():
    responses = iter(['{"positive":"false"}', '{"object":"wrong"}', '{"positive":false}'])
    teacher = HumanTeacher(read=lambda _: next(responses), write=lambda _: None)
    answer = teacher.ask(Query("a", "can", "x"))
    assert answer.source == "human-unverified"
    assert answer.facts[0].positive is False


def test_tick_controller_operates_with_bounded_dual_memory(tmp_path):
    from autonomous_learner.memory_policy import DualMemory
    world = World.generate(290)
    memory = DualMemory(world.goal(), "heuristic", tmp_path / "dual")
    loop = PersistentLoop(tmp_path / "session.json", Agent(memory, world.catalog(), world.goal()), Heuristic())
    events = list(loop.run(SyntheticTeacher(world), 8))
    assert events[-1]["action"] == "ANSWER"
    assert memory.answer(world.goal()).value == world.truth(world.goal())
    assert len(memory.beliefs.inspect()) == 3
    memory.close()
