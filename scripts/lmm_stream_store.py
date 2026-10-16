"""Bounded compressed inputs and append-only, crash-durable fit journals."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import sqlite3
import time
from pathlib import Path

LIMIT = 8 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(value):
    raw = canonical(value)
    if len(raw) > LIMIT:
        raise ValueError("record/chunk exceeds bounded payload limit")
    data = gzip.compress(raw, mtime=0)
    return data, sha(data)


def unpack(data, expected):
    if sha(data) != expected:
        raise ValueError("payload checksum mismatch")
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        raw = bytearray()
        while len(raw) <= LIMIT:
            block = stream.read(min(65536, LIMIT + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
    if len(raw) > LIMIT:
        raise ValueError("expanded payload exceeds limit")

    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")

    return json.loads(raw, parse_constant=invalid)


@contextlib.contextmanager
def case_lock(path):
    """OS releases this advisory lock on process death; never use stale PID files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Journal:
    """Caller holds case_lock for the full read/recover/run operation.

    SQLite transactions atomically persist compressed chunks and their inventory;
    no two-file rename/index crash window. Commit each start before computation.
    """

    def __init__(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=0)
        self.metadata = metadata
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1),
                value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (first INTEGER PRIMARY KEY, count INTEGER NOT NULL,
                payload BLOB NOT NULL, sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inputs (replicate INTEGER PRIMARY KEY,
                chunk INTEGER NOT NULL REFERENCES chunks(first), sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS starts (engine TEXT NOT NULL, replicate INTEGER NOT NULL
                REFERENCES inputs(replicate), input_sha TEXT NOT NULL, started REAL NOT NULL,
                PRIMARY KEY(engine,replicate));
            CREATE TABLE IF NOT EXISTS outcomes (engine TEXT NOT NULL, replicate INTEGER NOT NULL,
                payload BLOB NOT NULL, sha TEXT NOT NULL, PRIMARY KEY(engine,replicate),
                FOREIGN KEY(engine,replicate) REFERENCES starts(engine,replicate));
        """)
        for table in ("metadata", "chunks", "inputs", "starts", "outcomes"):
            for action in ("UPDATE", "DELETE"):
                self.db.execute(f"""CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action}
                    BEFORE {action} ON {table} BEGIN
                    SELECT RAISE(ABORT,'append-only journal'); END""")
        saved = self.db.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if saved is None:
            with self.db:
                self.db.execute("INSERT INTO metadata VALUES(1,?)", (canonical(metadata),))
        elif saved[0] != canonical(metadata):
            self.db.close()
            raise ValueError("journal manifest/case/budget identity changed")

    def close(self):
        self.db.close()

    def input_count(self):
        return self.db.execute("SELECT count(*) FROM inputs").fetchone()[0]

    def add_chunk(self, rows):
        begin = self.input_count()
        if not rows or len(rows) > self.metadata["chunk_size"]:
            raise ValueError("empty or oversized input chunk")
        for index, row in enumerate(rows, begin):
            if (
                type(row.get("replicate")) is not int
                or row["replicate"] != index
                or row.get("case") != self.metadata["case"]
                or index >= self.metadata["n"]
            ):
                raise ValueError("input chunk keys exceed or differ from fixed inventory")
        data, digest = pack(rows)
        with self.db:
            self.db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (begin, len(rows), data, digest))
            self.db.executemany(
                "INSERT INTO inputs VALUES(?,?,?)",
                [(r["replicate"], begin, sha(canonical(r))) for r in rows],
            )

    def iter_inputs(self):
        expected = 0
        for begin, count, data, digest in self.db.execute("SELECT * FROM chunks ORDER BY first"):
            rows = unpack(data, digest)
            if (
                begin != expected
                or len(rows) != count
                or not 0 < count <= self.metadata["chunk_size"]
            ):
                raise ValueError("input chunk inventory differs")
            for row in rows:
                indexed = self.db.execute(
                    "SELECT chunk,sha FROM inputs WHERE replicate=?", (expected,)
                ).fetchone()
                if (
                    row.get("case") != self.metadata["case"]
                    or type(row.get("replicate")) is not int
                    or row["replicate"] != expected
                    or expected >= self.metadata["n"]
                    or indexed != (begin, sha(canonical(row)))
                ):
                    raise ValueError("input identity/index differs")
                yield row
                expected += 1
        if expected != self.input_count():
            raise ValueError("unindexed or missing input chunk")

    def attempted(self, engine, replicate):
        return (
            self.db.execute(
                "SELECT 1 FROM starts WHERE engine=? AND replicate=?", (engine, replicate)
            ).fetchone()
            is not None
        )

    def start(self, engine, replicate):
        if type(replicate) is not int or engine not in self.metadata["engines"]:
            raise ValueError("invalid attempt key")
        if engine.startswith("reference:") and replicate >= self.metadata["reference_n"]:
            raise ValueError("reference key outside predesignated subset")
        indexed = self.db.execute(
            "SELECT sha FROM inputs WHERE replicate=?", (replicate,)
        ).fetchone()
        if indexed is None:
            raise ValueError("input must be committed before starting a fit")
        with self.db:
            self.db.execute(
                "INSERT INTO starts VALUES(?,?,?,?)", (engine, replicate, indexed[0], time.time())
            )

    def finish(self, engine, replicate, value):
        data, digest = pack(value)
        with self.db:
            self.db.execute(
                "INSERT INTO outcomes VALUES(?,?,?,?)", (engine, replicate, data, digest)
            )

    def recover(self):
        """Interrupted attempts get explicit failures, never a second fit."""
        pending = self.db.execute("""SELECT s.engine,s.replicate FROM starts s LEFT JOIN outcomes o
            USING(engine,replicate) WHERE o.engine IS NULL ORDER BY s.engine,s.replicate""")
        for engine, replicate in pending:
            self.finish(
                engine,
                replicate,
                {"converged": False, "methods": {}, "fit_error": "interrupted_after_durable_start"},
            )

    def iter_outcomes(self, engine):
        for replicate, data, digest in self.db.execute(
            "SELECT replicate,payload,sha FROM outcomes WHERE engine=? ORDER BY replicate",
            (engine,),
        ):
            yield replicate, unpack(data, digest)

    def validate(self):
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        if self.db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("journal foreign-key violation")
        for _ in self.iter_inputs():
            pass
        for engine, rep, input_sha in self.db.execute(
            "SELECT engine,replicate,input_sha FROM starts"
        ):
            expected = self.db.execute(
                "SELECT sha FROM inputs WHERE replicate=?", (rep,)
            ).fetchone()
            if (
                engine not in self.metadata["engines"]
                or expected != (input_sha,)
                or (engine.startswith("reference:") and rep >= self.metadata["reference_n"])
            ):
                raise ValueError("attempt input/key differs")
        for data, digest in self.db.execute("SELECT payload,sha FROM outcomes"):
            if not isinstance(unpack(data, digest), dict):
                raise ValueError("outcome must be a JSON object")
