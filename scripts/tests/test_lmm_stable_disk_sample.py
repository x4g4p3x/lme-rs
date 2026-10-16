"""Sample transient-file deletion without hiding real measurement failures."""

import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmm_resource_caps_stable as caps  # noqa: E402


class StableDiskSampleTests(unittest.TestCase):
    def test_disappearing_child_does_not_abort_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable, transient = root / "journal.sqlite", root / "journal.sqlite-wal"
            stable.write_bytes(b"retained")
            real_stat = Path.stat

            def measured(path, *args, **kwargs):
                if path == transient:
                    raise FileNotFoundError(os.fspath(transient))
                return real_stat(path, *args, **kwargs)

            # A file can disappear after is_file and before the second stat.
            # The old two-stat implementation fails; a single-stat sampler can
            # skip a vanished child while still charging every surviving file.
            entries = [SimpleNamespace(path=str(p)) for p in (stable, transient)]
            with (
                patch.object(os, "scandir", return_value=nullcontext(iter(entries))),
                patch.object(Path, "rglob", return_value=iter([stable, transient])),
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "stat", measured),
            ):
                sample = caps.disk_sample(root)
            self.assertEqual(sample["bytes"], len(b"retained"))
            self.assertGreater(sample["free_bytes"], 0)

    def test_permission_and_missing_root_errors_are_not_swallowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, child = Path(directory), Path(directory) / "secret"
            real_stat = Path.stat

            def denied(path, *args, **kwargs):
                if path == child:
                    raise PermissionError("injected permission failure")
                return real_stat(path, *args, **kwargs)

            entries = [SimpleNamespace(path=str(child))]
            with (
                patch.object(os, "scandir", return_value=nullcontext(iter(entries))),
                patch.object(Path, "rglob", return_value=iter([child])),
                patch.object(Path, "stat", denied),
            ):
                with self.assertRaises(PermissionError):
                    caps.disk_sample(root)
            with self.assertRaises(FileNotFoundError):
                caps.disk_sample(root / "missing")

    def test_root_disappearing_during_sample_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root.stat()
            with (
                patch.object(os, "scandir", return_value=nullcontext(iter([]))),
                patch.object(Path, "stat", side_effect=[info, FileNotFoundError("root vanished")]),
            ):
                with self.assertRaises(FileNotFoundError):
                    caps.disk_sample(root)

    def test_directory_enumeration_permission_errors_are_not_suppressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "blocked"
            blocked.mkdir()
            (blocked / "retained.bin").write_bytes(b"must not disappear from count")
            real_scan = os.scandir

            def scan(path):
                if Path(path) == blocked:
                    raise PermissionError("injected denied subtree")
                return real_scan(path)

            with patch.object(os, "scandir", side_effect=scan):
                with self.assertRaisesRegex(PermissionError, "denied subtree"):
                    caps.disk_sample(root)

    @unittest.skipUnless(os.name == "nt", "requires Windows owned jobs")
    def test_real_sqlite_wal_churn_completes_under_owned_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = [
                sys.executable,
                "-c",
                "import sqlite3,sys,time; from pathlib import Path\n"
                "for i in range(100):\n"
                " db=sqlite3.connect(str(Path(sys.argv[1])/'data.sqlite'))\n"
                " db.execute('PRAGMA journal_mode=WAL')\n"
                " db.execute('CREATE TABLE IF NOT EXISTS data (value)')\n"
                " db.execute('INSERT INTO data VALUES(?)',(i,))\n"
                " db.commit(); db.close(); time.sleep(.002)\n",
                str(root),
            ]
            result = caps.run_capped(args, caps.Caps(), root, root / "receipts", "churn")
            self.assertEqual(result["exit_codes"], [0])
            self.assertEqual(caps.acceptance_failures(result, caps.Caps().record()), [])


if __name__ == "__main__":
    unittest.main()
