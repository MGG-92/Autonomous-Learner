from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .memory import Memory
from .policy import Action, Controller, candidates, encode
from .types import Observation, Query, Teacher


@dataclass
class EpisodeResult:
    correct: bool
    answered: bool
    reward: float
    questions: int
    steps: int
    status: str
    trace: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class Agent:
    def __init__(self, memory: Memory, catalog: tuple[Query, ...], goal: Query, max_steps: int = 8):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.memory, self.goal = memory, goal
        self.actions = candidates(catalog)
        self.asked: set[Query] = set()
        self.max_steps = max_steps

    def features(self, step: int) -> torch.Tensor:
        return encode(self.actions, self.memory, self.goal, self.asked, (self.max_steps - step) / self.max_steps)

    def execute(self, action: Action, teacher: Teacher) -> tuple[float, dict]:
        event = {"action": action.kind}
        if action.kind == "ASK":
            assert action.query is not None
            observation = teacher.ask(action.query)
            if (not isinstance(observation, Observation) or observation.query != action.query or any(
                    fact.subject != action.query.subject or fact.relation != action.query.relation
                    or (action.query.object and fact.object != action.query.object) for fact in observation.facts)):
                raise ValueError("Teacher evidence does not address the requested query")
            self.memory.store(observation)
            self.asked.add(action.query)
            event.update(query=action.query.text(), facts=[asdict(f) for f in observation.facts])
            return -0.03, event
        if action.kind == "WAIT":
            return -0.01, event
        return 0.0, event


def run_episode(agent: Agent, teacher: Teacher, policy, truth: bool,
                sample: bool = False, observe: Callable | None = None):
    """truth is read only by the reward/evaluation boundary after an action."""
    reward, trace, log_probs = 0.0, [], []
    answered = correct = False
    status = "UNKNOWN"
    for step in range(agent.max_steps):
        features = agent.features(step)
        if isinstance(policy, Controller):
            logits = policy(features)
            dist = torch.distributions.Categorical(logits=logits)
            index = dist.sample() if sample else logits.argmax()
            if sample:
                log_probs.append(dist.log_prob(index))
            index = int(index)
        else:
            index = policy.choose(features)
        action = agent.actions[index]
        cost, event = agent.execute(action, teacher)
        reward += cost
        if action.kind == "ANSWER":
            answer = agent.memory.answer(agent.goal)
            status = answer.status
            answered = answer.value is not None
            correct = answered and answer.value == truth
            reward += (1.0 if correct else -1.0) if answered else -0.1
            event.update(answer=answer.to_dict())
        trace.append(event)
        if observe:
            observe(event)
        if action.kind in {"ANSWER", "ABSTAIN"}:
            break
    result = EpisodeResult(correct, answered, reward, sum(e["action"] == "ASK" for e in trace),
                           len(trace), status, trace)
    return result, log_probs
