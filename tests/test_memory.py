from autonomous_learner.memory import Memory
from autonomous_learner.types import Fact, Observation, Query


def store(memory, *facts):
    memory.store(Observation(Query("test", "test"), tuple(facts), "test-teacher"))


def test_proof_survives_restart_and_supports_new_query(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory = Memory(path)
    goal = Query("dax", "can", "zep")
    assert memory.answer(goal).status == "UNKNOWN"
    store(memory, Fact("dax", "is_a", "miro"), Fact("miro", "is_a", "root"), Fact("root", "can", "zep"))
    before = memory.answer(goal)
    assert before.value is True and len(before.evidence_ids) == 3
    memory.close()
    memory = Memory(path)
    assert memory.answer(goal) == before
    assert memory.answer(Query("miro", "can", "zep")).value is True
    memory.close()


def test_negative_conflict_cycles_and_unknown():
    memory = Memory()
    goal = Query("a", "can", "x")
    store(memory, Fact("a", "is_a", "b"), Fact("b", "is_a", "a"), Fact("b", "can", "x", False))
    assert memory.answer(goal).status == "REFUTED"
    store(memory, Fact("a", "can", "x", True))
    assert memory.answer(goal).status == "CONFLICT"
    assert memory.answer(Query("a", "can", "missing")).status == "UNKNOWN"
    memory.close()


def test_capacity_and_no_memory():
    memory = Memory(capacity=1)
    store(memory, Fact("a", "is_a", "b"), Fact("b", "can", "x"))
    assert len(memory.inspect()) == 1
    assert memory.answer(Query("a", "can", "x")).status == "UNKNOWN"
    memory.close()
    disabled = Memory(capacity=0)
    store(disabled, Fact("b", "can", "x"))
    assert disabled.inspect() == []
    disabled.close()


def test_inspection_never_trims_persistent_memory(tmp_path):
    path = tmp_path / "large.sqlite"
    memory = Memory(path, capacity=100)
    store(memory, *(Fact(str(i), "can", "x") for i in range(80)))
    memory.close()
    assert len(Memory.inspect_file(path)) == 80
    assert len(Memory.inspect_file(path)) == 80
