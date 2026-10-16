"""Ensure exploratory refits cannot replace the source evidence or mislabel binaries."""

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_lmm_random_slope_fix as diagnostic  # noqa: E402


class DiagnosticIntegrityTests(unittest.TestCase):
    def test_audit_rejects_changed_candidate_before_writing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "original", root / "diagnostic"
            source.mkdir()
            destination.mkdir()
            (destination / "candidate.json").write_text("changed candidate")
            diagnostic.pilot.shared.write_json(
                destination / "summary.json", {"candidate_sha256": "expected hash"}
            )
            with patch.object(diagnostic.pilot, "verify_run", return_value={}):
                with self.assertRaisesRegex(ValueError, "diagnostic artifact changed"):
                    diagnostic.audit(source, destination)
            self.assertFalse((destination / "audit.json").exists())

    def test_output_overlap_and_overwrite_rejected_before_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "study"
            source.mkdir()
            for output in (source, source / "diagnostic", source.parent):
                with self.assertRaises(ValueError):
                    diagnostic.prepare(source, output, None, {})
            existing = Path(directory) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                diagnostic.prepare(source, existing, None, {})

    def test_same_native_or_mismatched_wheel_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "study"
            source.mkdir()
            destination = Path(directory) / "new"
            wheel = Path(directory) / "candidate.whl"
            native = b"new candidate"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("lme_python/native.pyd", native)
            old_sha = "old"
            with (
                patch.object(
                    diagnostic.pilot,
                    "verify_run",
                    return_value={"freeze": {"environment": {"native_sha256": old_sha}}},
                ),
                patch.object(diagnostic.pilot, "load", return_value={"artifacts": {}}),
            ):
                for sha in (old_sha, "wrong"):
                    with self.assertRaises(ValueError):
                        diagnostic.prepare(source, destination, wheel, {"native_sha256": sha})
                    self.assertFalse(destination.exists())
                self.assertNotEqual(hashlib.sha256(native).hexdigest(), old_sha)


if __name__ == "__main__":
    unittest.main()
