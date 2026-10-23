"""Bounded accessible evidence; researcher logs live outside this database."""
import sqlite3
import math
import time
from collections import deque
from pathlib import Path

from .types import Answer, Fact, Observation, Query


def validate_record(row):
    positive = row["positive"]
    if type(positive) not in (int, bool) or positive not in (0, 1):
        raise ValueError("Persisted polarity must be a boolean or SQLite 0/1")
    Fact(row["subject"], row["relation"], row["object"], bool(positive))
    if (not isinstance(row["source"], str) or not row["source"].strip()
            or type(row["observed_at"]) not in (int, float) or not math.isfinite(row["observed_at"])
            or row["observed_at"] < 0):
        raise ValueError("Invalid evidence provenance or timestamp")


class Memory:
    @staticmethod
    def inspect_file(path: Path) -> list[dict]:
        """Inspection cannot trim, create, or modify the agent's memory."""
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM evidence ORDER BY id")]

    def __init__(self, path: str | Path = ":memory:", capacity: int = 64):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.db = sqlite3.connect(str(path))
        try:
            tables = {row[0] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "memory_settings" in tables:
                settings = self.db.execute("SELECT schema_version, capacity FROM memory_settings").fetchall()
                if settings != [(1, capacity)]:
                    raise ValueError("Persisted memory capacity or schema differs; preserve this database")
            if "evidence" in tables and self.db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] > capacity:
                raise ValueError("Existing evidence exceeds requested capacity; refusing to trim on open")
            if "evidence" in tables:
                for row in self.inspect():
                    validate_record(row)
            with self.db:
                self.db.execute("""CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, relation TEXT,
                    object TEXT, positive INTEGER, source TEXT, observed_at REAL,
                    UNIQUE(subject, relation, object, positive, source))""")
                if "memory_settings" not in tables:
                    self.db.execute("CREATE TABLE memory_settings (schema_version INTEGER, capacity INTEGER)")
                    self.db.execute("INSERT INTO memory_settings VALUES (1, ?)", (capacity,))
        except Exception:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def _trim(self) -> None:
        self.db.execute("DELETE FROM evidence WHERE id NOT IN (SELECT id FROM evidence ORDER BY id DESC LIMIT ?)", (self.capacity,))

    def store(self, observation: Observation) -> None:
        with self.db:
            for f in observation.facts:
                self.db.execute("""INSERT OR IGNORE INTO evidence
                    (subject, relation, object, positive, source, observed_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (f.subject, f.relation, f.object, int(f.positive), observation.source, time.time()))
            self._trim()

    def session_signature(self):
        return {"type": "Memory", "capacity": self.capacity}

    def session_snapshot(self):
        return self.inspect()

    def retain(self, records: list[dict]) -> None:
        """Atomically select an accessible snapshot, preserving IDs of retained facts."""
        if len(records) > self.capacity:
            raise ValueError("Selected evidence exceeds capacity")
        for row in records:
            validate_record(row)
        fields = ("subject", "relation", "object", "positive", "source")
        selected = {tuple(r[k] for k in fields) for r in records}
        with self.db:
            for row in self.inspect():
                if tuple(row[k] for k in fields) not in selected:
                    self.db.execute("DELETE FROM evidence WHERE id = ?", (row["id"],))
            for row in records:
                self.db.execute("""INSERT OR IGNORE INTO evidence
                    (subject, relation, object, positive, source, observed_at) VALUES (?, ?, ?, ?, ?, ?)""",
                    tuple(row[k] for k in fields) + (row["observed_at"],))

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
