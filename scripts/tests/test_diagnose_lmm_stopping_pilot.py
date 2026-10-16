"""Reanalysis must preserve failed-study input/output identity."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_lmm_stopping_pilot as diagnostic  # noqa: E402


def fixture(root):
    folder = root / "study"
    (folder / "analysis-executions").mkdir(parents=True)
    (folder / "analysis").mkdir()
    (folder / "cases/c").mkdir(parents=True)
    (folder / "manifest.json").write_text("original manifest")
    (folder / "cases/c/journal.sqlite").write_bytes(b"original fits")
    (folder / "analysis/partial.gz").write_bytes(b"partial output")
    (folder / "analysis-executions/analysis-one-end.json").write_text(
        json.dumps(
            {
                "resource_guard": {
                    "abort_reason": "resource_measurement_unavailable:FileNotFoundError"
                }
            }
        )
    )
    for name in ("runtime.py", "driver.py"):
        (root / name).write_text(name)
    manifest = {
        "cases": [{"id": "c"}],
        "sources": {"runtime.py": diagnostic.file_hash(root / "runtime.py")},
        "caps": {"seconds": 900},
        "control": {"tolerance": 1e-10},
    }
    return folder, manifest


class DiagnosticPreservationTests(unittest.TestCase):
    def test_fit_partial_and_abort_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, manifest = fixture(root)
            with (
                patch.object(diagnostic.run, "ROOT", root),
                patch.object(diagnostic.run, "verify", return_value=manifest),
                patch.object(diagnostic, "SOURCES", ["driver.py"]),
            ):
                diagnostic.freeze(folder)
                diagnostic.verify(folder)
                for name in (
                    "cases/c/journal.sqlite",
                    "analysis/partial.gz",
                    "analysis-executions/analysis-one-end.json",
                ):
                    path = folder / name
                    saved = path.read_bytes()
                    path.write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "original fits/partial"):
                        diagnostic.verify(folder)
                    path.write_bytes(saved)
                diagnostic.verify(folder)

    def test_nonempty_wal_and_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, manifest = fixture(root)
            with (
                patch.object(diagnostic.run, "ROOT", root),
                patch.object(diagnostic.run, "verify", return_value=manifest),
                patch.object(diagnostic, "SOURCES", ["driver.py"]),
            ):
                diagnostic.freeze(folder)
                wal = folder / "cases/c/journal.sqlite-wal"
                wal.write_bytes(b"uncheckpointed")
                with self.assertRaisesRegex(ValueError, "nonempty WAL"):
                    diagnostic.verify(folder)
                wal.write_bytes(b"")
                (root / "driver.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "diagnostic source"):
                    diagnostic.verify(folder)

    def test_original_abort_is_required_before_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, manifest = fixture(root)
            (folder / "analysis-executions/analysis-one-end.json").write_text(
                json.dumps({"resource_guard": {"abort_reason": None}})
            )
            with patch.object(diagnostic.run, "verify", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "retained original"):
                    diagnostic.freeze(folder)
            self.assertFalse((folder / "diagnostic-freeze.json").exists())


if __name__ == "__main__":
    unittest.main()
