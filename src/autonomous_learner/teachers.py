"""Explicit teacher interfaces. Human claims are evidence, not verified truth."""
import json

from .types import Fact, Observation, Query


class HumanTeacher:
    def __init__(self, read=input, write=print):
        self.read, self.write = read, write

    def ask(self, query: Query) -> Observation:
        self.write(f"Agent asks: {query.text()}")
        self.write('Reply ? for unknown, or JSON: {"object":"name", "positive":true}.')
        for _ in range(3):
            response = self.read("> ").strip()
            if response == "?":
                return Observation(query, (), "human-unverified")
            try:
                payload = json.loads(response)
                obj = payload.get("object", query.object)
                positive = payload.get("positive", True)
                if not isinstance(obj, str) or not obj or type(positive) is not bool:
                    raise ValueError("object must be a name and positive must be a JSON boolean")
                if query.object and query.object != obj:
                    raise ValueError("reply must address the requested object")
                fact = Fact(query.subject, query.relation, obj, positive)
                return Observation(query, (fact,), "human-unverified")
            except (ValueError, TypeError, AttributeError) as exc:
                self.write(f"Invalid teacher response: {exc}")
        raise ValueError("Three invalid teacher responses; no evidence stored")


class LocalLMTeacher:
    """A separately invoked language engine; its answers are unverified evidence.

    knowledge belongs to the teacher and is never supplied to the agent controller.
    Invalid model output is unknown, never silently promoted to a verified fact.
    """
    def __init__(self, engine, knowledge: str):
        self.engine, self._knowledge = engine, knowledge

    def ask(self, query: Query) -> Observation:
        prompt = ("Use only the following teacher notes. If the answer is absent return ?. "
                  'Otherwise return only JSON {"object":"NAME","positive":true} or false.\n'
                  f"Teacher notes:\n{self._knowledge}\nQuestion: {query.text()}")
        response = self.engine.complete(prompt)
        try:
            data = json.loads(response)
            obj, positive = data.get("object", query.object), data.get("positive")
            if not isinstance(obj, str) or not obj or type(positive) is not bool:
                raise ValueError("Invalid teacher response")
            if query.object and query.object != obj:
                raise ValueError("Object mismatch")
            return Observation(query, (Fact(query.subject, query.relation, obj, positive),), "local-llm-unverified")
        except (ValueError, TypeError, AttributeError):
            return Observation(query, (), "local-llm-unverified")
