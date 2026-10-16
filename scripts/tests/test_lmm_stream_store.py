"""Failure injection against durable, bounded scientific attempt storage."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmm_stream_store import Journal, case_lock  # noqa: E402


def metadata(n=3, chunk_size=2):
    return {
        "case": "case",
        "n": n,
        "chunk_size": chunk_size,
        "reference_n": 1,
        "engines": ["candidate", "reference:baseline"],
        "manifest_sha": "frozen",
    }


def rows(first, count):
    return [{"case": "case", "replicate": i, "y": [i, 2, 3]} for i in range(first, first + count)]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "case.sqlite"
        self.journal = Journal(self.path, metadata())

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def test_input_first_no_duplicate_outcomes_and_append_only(self):
        with self.assertRaises(ValueError):
            self.journal.start("candidate", 0)
        self.journal.add_chunk(rows(0, 2))
        self.journal.start("candidate", 0)
        with self.assertRaises(sqlite3.IntegrityError):
            self.journal.start("candidate", 0)
        self.journal.finish("candidate", 0, {"fit_error": "retained failure"})
        with self.assertRaises(sqlite3.IntegrityError):
            self.journal.finish("candidate", 0, {"fit_error": "replacement"})
        for statement in ("DELETE FROM outcomes", "UPDATE starts SET started=0"):
            with self.assertRaises(sqlite3.IntegrityError):
                self.journal.db.execute(statement)
        self.journal.validate()

    def test_forced_process_exit_after_committed_start_is_not_retried(self):
        self.journal.add_chunk(rows(0, 2))
        script = """import sys,json,os
sys.path.insert(0, sys.argv[1])
from lmm_stream_store import Journal,case_lock
with case_lock(sys.argv[2]+'.lock'):
 j=Journal(sys.argv[2],json.loads(sys.argv[3]))
 j.start('candidate',0)
 os._exit(17)
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(Path(__file__).resolve().parents[1]),
                str(self.path),
                json.dumps(metadata()),
            ],
            check=False,
        )
        self.assertEqual(result.returncode, 17)
        with case_lock(str(self.path) + ".lock"):
            self.journal.recover()
            self.journal.recover()
            self.assertTrue(self.journal.attempted("candidate", 0))
            self.assertFalse(self.journal.attempted("candidate", 1))
            self.journal.start("candidate", 1)
            self.journal.finish("candidate", 1, {"converged": True})
        outcomes = list(self.journal.iter_outcomes("candidate"))
        self.assertEqual([r for r, _ in outcomes], [0, 1])
        self.assertEqual(outcomes[0][1]["fit_error"], "interrupted_after_durable_start")
        self.journal.validate()

    def test_exclusive_lock_blocks_another_process(self):
        script = """import sys
sys.path.insert(0,sys.argv[1])
from lmm_stream_store import case_lock
try:
 with case_lock(sys.argv[2]): pass
except OSError:
 sys.exit(23)
"""
        with case_lock(str(self.path) + ".lock"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(Path(__file__).resolve().parents[1]),
                    str(self.path) + ".lock",
                ],
                check=False,
            )
        self.assertEqual(result.returncode, 23)

    def test_invalid_and_nonfinite_chunks_are_atomic(self):
        for bad in (
            rows(1, 2),
            rows(0, 3),
            [dict(rows(0, 1)[0], case="other")],
            [dict(rows(0, 1)[0], replicate=True)],
            [dict(rows(0, 1)[0], y=[float("nan")])],
        ):
            with self.assertRaises(ValueError):
                self.journal.add_chunk(bad)
            self.assertEqual(self.journal.input_count(), 0)

    def test_unknown_reference_subset_or_noninteger_key_cannot_start(self):
        self.journal.add_chunk(rows(0, 2))
        for engine, rep in (("reference:baseline", 1), ("other", 0), ("candidate", True)):
            with self.assertRaises(ValueError):
                self.journal.start(engine, rep)

    def test_manifest_budget_changes_are_rejected(self):
        with self.assertRaises(ValueError):
            Journal(self.path, metadata(n=4))

    def test_truncated_input_payload_is_detected(self):
        self.journal.add_chunk(rows(0, 2))
        self.journal.db.execute("DROP TRIGGER immutable_chunks_UPDATE")
        with self.journal.db:
            self.journal.db.execute("UPDATE chunks SET payload=substr(payload,1,10)")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.journal.validate()

    def test_changed_outcome_payload_is_detected(self):
        self.journal.add_chunk(rows(0, 2))
        self.journal.start("candidate", 0)
        self.journal.finish("candidate", 0, {"fit_error": "original"})
        self.journal.db.execute("DROP TRIGGER immutable_outcomes_UPDATE")
        with self.journal.db:
            self.journal.db.execute("UPDATE outcomes SET payload=substr(payload,1,5)")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.journal.validate()

    def test_iteration_is_bounded_instead_of_materializing_campaign(self):
        path = Path(self.temp.name) / "large.sqlite"
        journal = Journal(path, metadata(n=10000, chunk_size=32))
        try:
            for first in range(0, 10000, 32):
                journal.add_chunk(rows(first, min(32, 10000 - first)))
            tracemalloc.start()
            count = sum(1 for _ in journal.iter_inputs())
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertEqual(count, 10000)
            self.assertLess(peak, 2 * 1024 * 1024)
        finally:
            journal.close()


if __name__ == "__main__":
    unittest.main()
