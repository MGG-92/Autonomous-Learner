"""Bounded persistent ticks. No hidden teacher state is serialized as agent memory."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .agent import Agent
from .policy import Controller
from .types import Query, Teacher


class PersistentLoop:
    def __init__(self, session: Path, agent: Agent, policy, max_questions: int = 8):
        if max_questions < 0:
            raise ValueError("max_questions must be nonnegative")
        self.session, self.agent, self.policy = session, agent, policy
        self.session.parent.mkdir(parents=True, exist_ok=True)
        self.signature = {"goal": asdict(agent.goal),
                          "catalog": [asdict(a.query) for a in agent.actions if a.query],
                          "max_steps": agent.max_steps, "max_questions": max_questions,
                          "memory_type": type(agent.memory).__name__}
        if session.exists():
            self.state = json.loads(session.read_text(encoding="utf-8"))
            if self.state["signature"] != self.signature:
                raise ValueError("Session task or budgets differ; choose a new --session path")
            agent.asked = {Query(**q) for q in self.state["asked"]}
        else:
            self.state = {"signature": self.signature, "ticks": 0, "questions": 0,
                          "asked": [], "finished": False, "last_tick_utc": None}

    def _save(self):
        temporary = self.session.with_suffix(self.session.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(self.session)

    def tick(self, teacher: Teacher) -> dict:
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
            _, event = self.agent.execute(action, teacher)
            if action.kind == "ASK":
                self.state["questions"] += 1
            elif action.kind == "ANSWER":
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
        self.state["finished"] |= self.state["ticks"] >= self.agent.max_steps
        self._save()
        # Researcher audit output is never read by policy features.
        with self.session.with_suffix(".audit.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")
        return event

    def run(self, teacher: Teacher, ticks: int, interval: float = 0.0):
        if ticks < 1 or interval < 0 or interval > 60:
            raise ValueError("ticks must be positive; interval must be between 0 and 60 seconds")
        for i in range(ticks):
            event = self.tick(teacher)
            yield event
            if self.state["finished"]:
                break
            if interval and i + 1 < ticks:
                time.sleep(interval)
