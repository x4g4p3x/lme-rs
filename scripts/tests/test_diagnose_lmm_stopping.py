"""Independent dense likelihood and evidence-preservation controls."""

import json
import math
import struct
import sys
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_lmm_stopping as diagnostic  # noqa: E402


class StoppingDiagnosticTests(unittest.TestCase):
    def test_balanced_projection_matches_independent_dense_gls(self):
        import numpy as np

        # Non-centered x exercises the covariance transformation, not just the
        # symmetric fixture on which the projection formula was developed.
        for shift in (0.0, 0.125, 100.0):
            x = np.tile(np.array([-1.0, 0.0, 1.0, 2.0]) + shift, 3)
            y = np.array([1.1, -0.2, 2.4, 3.0, 0.4, 1.0, 1.8, 2.9, 2.2, 1.1, 0.1, 4.2])
            case = {"sizes": [4] * 3, "x": x.tolist(), "group": np.repeat(range(3), 4).tolist()}
            theta = [0.7, -0.3, 0.4]
            lam = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
            z = np.zeros((12, 6))
            fixed = np.column_stack([np.ones(12), x])
            for g in range(3):
                z[4 * g : 4 * g + 4, 2 * g : 2 * g + 2] = fixed[4 * g : 4 * g + 4]
            v = np.eye(12) + z @ np.kron(np.eye(3), lam @ lam.T) @ z.T
            information = fixed.T @ np.linalg.solve(v, fixed)
            beta = np.linalg.solve(information, fixed.T @ np.linalg.solve(v, y))
            residual = y - fixed @ beta
            rss = residual @ np.linalg.solve(v, residual)
            expected = 10 * (math.log(2 * math.pi * rss / 10) + 1)
            expected += np.linalg.slogdet(v)[1] + np.linalg.slogdet(information)[1]
            with localcontext() as context:
                context.prec = 60
                actual = diagnostic.BalancedLikelihood(case, y.tolist()).criterion(theta)
            # Dense uncentered solves at shift100 lose more precision.
            self.assertAlmostEqual(float(actual), expected, delta=2e-8)

    def test_reference_precision_and_interior_optimum(self):
        raw = json.loads(diagnostic.FIXTURE.read_bytes())["fixture"]
        values = []
        for precision in (60, 80):
            with localcontext() as context:
                context.prec = precision
                likelihood = diagnostic.BalancedLikelihood(raw["case"], raw["y"])
                theta = likelihood.interior_optimum()
                best = likelihood.criterion(theta)
                values.append(best)
                self.assertGreater(likelihood.criterion(raw["candidate"]["theta"]), best)
                for axis in range(3):
                    for step in (Decimal("-0.000001"), Decimal("0.000001")):
                        point = theta.copy()
                        point[axis] += step
                        self.assertGreater(likelihood.criterion(point), best)
        self.assertLess(abs(values[0] - values[1]), Decimal("1e-50"))

    def test_prepare_preserves_bits_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.json"
            diagnostic.prepare(output)
            fixture = json.loads(diagnostic.FIXTURE.read_bytes())["fixture"]
            actual = json.loads(output.read_bytes())
            for key, values in (("x", fixture["case"]["x"]), ("y", fixture["y"])):
                expected = [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]
                self.assertEqual(actual[key + "_bits"], expected)
            self.assertEqual(actual["fixture_sha256"], diagnostic.digest(diagnostic.FIXTURE))
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                diagnostic.prepare(output)
            self.assertEqual(output.read_bytes(), before)

    def test_bad_design_and_source_overlap_rejected(self):
        case = {"sizes": [3, 3], "x": [-1.0, 0.0, 1.0] * 2, "group": [0, 1, 0, 1, 0, 1]}
        with self.assertRaisesRegex(ValueError, "ordered contiguous"):
            diagnostic.BalancedLikelihood(case, [1.0] * 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "separate from original"):
                diagnostic.freeze_sensitivity(root, root / "inside")
            self.assertEqual(list(root.iterdir()), [])

    def test_changed_runtime_helper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            helper.write_text("original")
            declaration = {"runtime_sources": {"helper.py": diagnostic.digest(helper)}}
            with patch.object(diagnostic, "ROOT", root):
                diagnostic.verify_runtime(declaration)
                helper.write_text("changed")
                with self.assertRaisesRegex(ValueError, "runtime helper changed"):
                    diagnostic.verify_runtime(declaration)

    def test_changed_published_journal_and_active_wal_rejected_before_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            public = root / "scientific"
            public.mkdir()
            manifest = {"cases": [{"id": str(i)} for i in range(94)], "sources": {}}
            diagnostic.write_new(source / "manifest.json", manifest)
            diagnostic.write_new(public / "RANDOM_SLOPE_REPAIR_PILOT_V1_FREEZE.json", manifest)
            inventory = []
            for case in manifest["cases"]:
                journal = source / "cases" / case["id"] / "journal.sqlite"
                journal.parent.mkdir(parents=True)
                journal.write_bytes(b"retained")
                inventory.append({"case": case["id"], "journal_sha256": diagnostic.digest(journal)})
            diagnostic.write_new(
                public / "RANDOM_SLOPE_REPAIR_PILOT_V1.json",
                {
                    "manifest_sha256": diagnostic.digest(source / "manifest.json"),
                    "journal_inventory": inventory,
                },
            )
            with patch.object(diagnostic, "ROOT", root):
                journal.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "published evidence"):
                    diagnostic.freeze_sensitivity(source, output)
                journal.write_bytes(b"retained")
                journal.with_name("journal.sqlite-wal").write_bytes(b"uncheckpointed")
                with self.assertRaisesRegex(ValueError, "active WAL"):
                    diagnostic.freeze_sensitivity(source, output)
                self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
