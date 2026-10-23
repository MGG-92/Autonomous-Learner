"""Bounded persistent ticks. No hidden teacher state is serialized as agent memory."""
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .agent import Agent
from .integrity import controller_identity, digest
from .policy import Controller, FEATURES, Heuristic
from .types import Query, Teacher


class PersistentLoop:
    def __init__(self, session: Path, agent: Agent, policy, max_questions: int = 8):
        if type(max_questions) is not int or max_questions < 0:
            raise ValueError("max_questions must be nonnegative")
        if not isinstance(policy, Controller) and type(policy) is not Heuristic:
            raise ValueError("Persistent sessions require a supported deterministic controller")
        self.session, self.agent, self.policy = session, agent, policy
        self.session.parent.mkdir(parents=True, exist_ok=True)
        self.signature = {"goal": asdict(agent.goal),
                          "catalog": [asdict(a.query) for a in agent.actions if a.query],
                          "max_steps": agent.max_steps, "max_questions": max_questions,
                          "memory": agent.memory.session_signature(),
                          "controller": controller_identity(policy), "features": list(FEATURES)}
        if session.exists():
            self.state = json.loads(session.read_text(encoding="utf-8"))
            self._validate_state()
            if self.state["signature"] != self.signature:
                raise ValueError("Session task, budgets, controller, or memory settings differ; choose a new --session path")
            if self.state["memory_digest"] != digest(agent.memory.session_snapshot()):
                raise ValueError("Session memory differs from saved state; preserve files and investigate interrupted writes")
            agent.asked = {Query(**q) for q in self.state["asked"]}
        else:
            if agent.memory.facts():
                raise ValueError("Existing memory has no session state; preserve it and choose fresh session/memory files")
            # Include episodic evidence too: facts() intentionally exposes beliefs only.
            snapshot = agent.memory.session_snapshot()
            if isinstance(snapshot, dict) and any(snapshot.values()):
                raise ValueError("Existing memory has no session state; preserve it and choose fresh session/memory files")
            self.state = {"schema_version": 2, "signature": self.signature, "ticks": 0, "questions": 0,
                          "asked": [], "finished": False, "last_tick_utc": None,
                          "teacher_type": None, "pending_action": None, "memory_digest": digest(snapshot)}

    def _validate_state(self):
        state = self.state
        if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 2:
            raise ValueError("Unsupported session state schema; preserve legacy sessions and use their original source")
        try:
            if state["pending_action"] is not None:
                raise ValueError("Session state has an interrupted tick; preserve files and investigate before resuming")
            ticks, questions, timestamp = state["ticks"], state["questions"], state["last_tick_utc"]
            if (type(ticks) is not int or not 0 <= ticks <= self.agent.max_steps
                    or type(questions) is not int or not 0 <= questions <= min(ticks, self.signature["max_questions"])
                    or type(state["finished"]) is not bool
                    or (ticks == self.agent.max_steps and not state["finished"])
                    or not isinstance(state["asked"], list)
                    or not isinstance(state["signature"], dict)
                    or not isinstance(state["memory_digest"], str) or len(state["memory_digest"]) != 64
                    or (ticks > 0 and (not isinstance(state["teacher_type"], str) or not state["teacher_type"]))
                    or (ticks == 0 and state["teacher_type"] is not None)):
                raise ValueError("Invalid session state counters or metadata")
            if ((ticks == 0 and timestamp is not None) or
                    (ticks > 0 and (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0))):
                raise ValueError("Invalid session state timestamp")
            asked = [Query(**q) for q in state["asked"]]
            catalog = {action.query for action in self.agent.actions if action.query}
            if len(asked) != len(set(asked)) or not set(asked) <= catalog or len(asked) > questions:
                raise ValueError("Invalid session state question history")
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid session state structure") from exc

    def _save(self):
        temporary = self.session.with_suffix(self.session.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.session)

    def tick(self, teacher: Teacher) -> dict:
        self._validate_state()
        if (controller_identity(self.policy) != self.signature["controller"]
                or self.agent.memory.session_signature() != self.signature["memory"]):
            raise ValueError("Session controller or memory settings differ")
        if digest(self.agent.memory.session_snapshot()) != self.state["memory_digest"]:
            raise ValueError("Session memory differs from saved state; refusing further actions")
        teacher_type = f"{type(teacher).__module__}.{type(teacher).__qualname__}"
        if self.state["teacher_type"] not in (None, teacher_type):
            raise ValueError("Session teacher type differs")
        if self.state["finished"]:
            return {"action": "IDLE", "reason": "session finished"}
        step = self.state["ticks"]
        if step >= self.agent.max_steps:
            self.state["finished"] = True
            self._save()
            return {"action": "IDLE", "reason": "compute budget exhausted"}
        features = self.agent.features(step)
        with torch.no_grad():
            index = int(self.policy(features).argmax()) if isinstance(self.policy, Controller) else self.policy.choose(features)
        action = self.agent.actions[index]
        if action.kind == "ASK" and self.state["questions"] >= self.signature["max_questions"]:
            event = {"action": "WAIT", "reason": "teacher budget exhausted"}
            self.state["finished"] = True
        else:
            # Persist intent and reserve the teacher call before external side effects.
            # A crash between SQLite and JSON commits must stop, not repeat the call.
            self.state["pending_action"] = {"kind": action.kind, "tick": step}
            self.state["teacher_type"] = teacher_type
            if action.kind == "ASK":
                self.state["questions"] += 1
            self._save()
            _, event = self.agent.execute(action, teacher)
            if action.kind == "ANSWER":
                event["answer"] = self.agent.memory.answer(self.agent.goal).to_dict()
                self.state["finished"] = True
            elif action.kind == "ABSTAIN":
                self.state["finished"] = True
        now = time.time()
        previous = self.state["last_tick_utc"]
        event.update(tick=step, timestamp_utc=now,
                     elapsed_seconds=None if previous is None else max(0.0, now - previous))
        self.state["ticks"] += 1
        self.state["last_tick_utc"] = now
        self.state["asked"] = [asdict(q) for q in sorted(self.agent.asked)]
        self.state["teacher_type"] = teacher_type
        pending = self.state["pending_action"]
        self.state["pending_action"] = None
        self.state["memory_digest"] = digest(self.agent.memory.session_snapshot())
        self.state["finished"] |= self.state["ticks"] >= self.agent.max_steps
        try:
            self._save()
        except Exception:
            self.state["pending_action"] = pending or {"kind": "COMMIT", "tick": step}
            raise
        # Researcher audit output is never read by policy features.
        with self.session.with_suffix(".audit.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")
        return event

    def run(self, teacher: Teacher, ticks: int, interval: float = 0.0):
        if (type(ticks) is not int or ticks < 1 or type(interval) not in (int, float)
                or not math.isfinite(interval) or interval < 0 or interval > 60):
            raise ValueError("ticks must be positive; interval must be between 0 and 60 seconds")
        for i in range(ticks):
            event = self.tick(teacher)
            yield event
            if self.state["finished"]:
                break
            if interval and i + 1 < ticks:
                time.sleep(interval)
