"""Bounded accessible evidence; researcher logs live outside this database."""
import sqlite3
import time
from collections import deque
from pathlib import Path

from .types import Answer, Fact, Observation, Query


class Memory:
    @staticmethod
    def inspect_file(path: Path) -> list[dict]:
        """Inspection cannot trim, create, or modify the agent's memory."""
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM evidence ORDER BY id")]

    def __init__(self, path: str | Path = ":memory:", capacity: int = 64):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.db = sqlite3.connect(str(path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, relation TEXT,
            object TEXT, positive INTEGER, source TEXT, observed_at REAL,
            UNIQUE(subject, relation, object, positive, source))""")
        self._trim()

    def close(self) -> None:
        self.db.close()

    def _trim(self) -> None:
        self.db.execute("DELETE FROM evidence WHERE id NOT IN (SELECT id FROM evidence ORDER BY id DESC LIMIT ?)", (self.capacity,))
        self.db.commit()

    def store(self, observation: Observation) -> None:
        for f in observation.facts:
            self.db.execute("""INSERT OR IGNORE INTO evidence
                (subject, relation, object, positive, source, observed_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (f.subject, f.relation, f.object, int(f.positive), observation.source, time.time()))
        self._trim()

    def inspect(self) -> list[dict]:
        self.db.row_factory = sqlite3.Row
        rows = [dict(row) for row in self.db.execute("SELECT * FROM evidence ORDER BY id")]
        self.db.row_factory = None
        return rows

    def facts(self) -> list[tuple[int, Fact]]:
        return [(r[0], Fact(r[1], r[2], r[3], bool(r[4]))) for r in self.db.execute(
            "SELECT id, subject, relation, object, positive FROM evidence ORDER BY id")]

    def reachable(self, subject: str) -> dict[str, tuple[int, ...]]:
        edges = {}
        for evidence_id, f in self.facts():
            if f.relation == "is_a" and f.positive:
                edges.setdefault(f.subject, []).append((f.object, evidence_id))
        paths = {subject: ()}
        queue = deque([subject])
        while queue:
            node = queue.popleft()
            for parent, evidence_id in edges.get(node, []):
                if parent not in paths:
                    paths[parent] = paths[node] + (evidence_id,)
                    queue.append(parent)
        return paths

    def answer(self, goal: Query) -> Answer:
        paths = self.reachable(goal.subject)
        proofs = {True: [], False: []}
        for evidence_id, f in self.facts():
            if f.subject in paths and f.relation == goal.relation and f.object == goal.object:
                proofs[f.positive].append(paths[f.subject] + (evidence_id,))
        if proofs[True] and proofs[False]:
            return Answer("CONFLICT", tuple(dict.fromkeys(proofs[True][0] + proofs[False][0])))
        for value, status in [(True, "SUPPORTED"), (False, "REFUTED")]:
            if proofs[value]:
                return Answer(status, min(proofs[value], key=len))
        return Answer("UNKNOWN")
