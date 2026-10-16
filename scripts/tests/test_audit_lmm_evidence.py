"""Independent negative controls for a read-only scientific evidence audit."""

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "audit_lmm_evidence.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class EvidenceAuditTests(unittest.TestCase):
    def test_checksum_rejects_changed_and_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"abc")
            expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(audit.hash_check(path, expected), expected)
            path.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit.hash_check(path, expected)
            with self.assertRaisesRegex(ValueError, "missing artifact"):
                audit.hash_check(Path(directory) / "absent", expected)

    def test_record_stream_rejects_duplicate_missing_extra_and_wrong_case(self):
        valid = [{"case": "c", "replicate": i} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl.gz"
            for index, rows in enumerate(
                [
                    valid,
                    valid[:2],
                    [valid[0], valid[0], valid[2]],
                    valid + [valid[2]],
                    [{"case": "other", "replicate": 0}] + valid[1:],
                    [{"case": "c", "replicate": False}] + valid[1:],
                ]
            ):
                with gzip.open(path, "wt") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
                if index == 0:
                    self.assertEqual(list(audit.records(path, "c", 3)), valid)
                else:
                    with self.assertRaises(ValueError):
                        list(audit.records(path, "c", 3))

    def test_artifact_paths_stay_relative(self):
        for name in ["../file", "C:/file", "/file", "a\\b"]:
            with self.assertRaises(ValueError):
                audit.safe_relative(name)
        self.assertEqual(audit.safe_relative("cases/a.json"), Path("cases/a.json"))

    def test_failure_envelopes_keep_all_attempts(self):
        bounds = {
            key: [[i / 32, (16 + i) / 32] for i in range(11)]
            for key in ["pointwise", "simultaneous"]
        }
        row = {"attempted": 10, "failed": 6}
        for metric in ["rejection", "coverage"]:
            row[metric] = {
                "evaluable": 4,
                "count": 1,
                "rate": 0.25,
                "failure_envelope95": [1 / 32, 23 / 32],
                "failure_envelope_simultaneous95": [1 / 32, 23 / 32],
            }
        audit.verify_counts(row, [10, 4, 1, 1], bounds)
        row["rejection"]["failure_envelope95"] = [1 / 32, 17 / 32]
        with self.assertRaisesRegex(ValueError, "failure envelope"):
            audit.verify_counts(row, [10, 4, 1, 1], bounds)

    def test_pointwise_interval_is_not_substituted_for_global_decision(self):
        self.assertEqual(audit.classification([0.04, 0.06]), "supported")
        self.assertEqual(audit.classification([0.039, 0.061]), "inconclusive")
        self.assertEqual(audit.classification([0.01, 0.03]), "contradicted")


if __name__ == "__main__":
    unittest.main()
