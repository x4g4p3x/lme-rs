"""Independent controls for pilot designs, analysis and evidence integrity."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "lmm_calibration", Path(__file__).resolve().parents[1] / "lmm_calibration.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class CalibrationTests(unittest.TestCase):
    def test_factorial_scope_and_design_normalization(self):
        cases = study.cases()
        self.assertEqual(len(cases), 80)
        self.assertEqual(len({c["id"] for c in cases}), 80)
        self.assertEqual(sum(c["primary"] for c in cases), 24)
        for case in cases:
            self.assertEqual(len(case["group"]), sum(case["sizes"]))
            if case["stratum"] == "balanced" or case["location"] == "within":
                self.assertAlmostEqual(sum(case["x"]), 0)
                self.assertAlmostEqual(sum(x * x for x in case["x"]), len(case["x"]))

    def test_inputs_independent_of_case_order_and_namespace(self):
        cases = study.cases()
        forward = {c["id"]: study.response(c, 7) for c in cases}
        backward = {c["id"]: study.response(c, 7) for c in reversed(cases)}
        self.assertEqual(forward, backward)
        self.assertNotEqual(study.response(cases[0], 7), study.response(cases[0], 8))
        self.assertNotEqual(study.normals("pilot", 4), study.normals("confirmatory", 4))

    def test_known_variance_gls_matches_balanced_information(self):
        for case in study.cases():
            if case["stratum"] != "balanced":
                continue
            y = [2 + 3 * x for x in case["x"]]
            beta, se = study.known_variance(case, y)
            g, m, rho = case["groups"], case["sizes"][0], case["rho"]
            variance = (
                (1 - rho) / (g * m) if case["location"] == "within" else (rho + (1 - rho) / m) / g
            )
            self.assertAlmostEqual(beta, 3, places=12)
            self.assertAlmostEqual(se**2, variance, places=12)

    def test_gls_matches_independent_group_projection_including_imbalance(self):
        for case in study.cases():
            y = study.response(case, 7)
            beta, se = study.known_variance(case, y)
            if case["location"] == "within":
                information = sum(x * x for x in case["x"])
                expected_beta = sum(x * v for x, v in zip(case["x"], y)) / information
                expected_variance = (1 - case["rho"]) / information
            else:
                means, x, weights = [], [], []
                start = 0
                for n in case["sizes"]:
                    means.append(sum(y[start : start + n]) / n)
                    x.append(case["x"][start])
                    weights.append(1 / (case["rho"] + (1 - case["rho"]) / n))
                    start += n
                center = sum(w * v for w, v in zip(weights, x)) / sum(weights)
                information = sum(w * (v - center) ** 2 for w, v in zip(weights, x))
                expected_beta = (
                    sum(w * (v - center) * mean for w, v, mean in zip(weights, x, means))
                    / information
                )
                expected_variance = 1 / information
            self.assertAlmostEqual(beta, expected_beta, places=12)
            self.assertAlmostEqual(se**2, expected_variance, places=12)

    def test_null_coverage_rejection_complement_and_alternative(self):
        record = {"beta": 0.3, "se": 0.1, "critical": 1.959963984540054, "p": 0.0027}
        outcome = study.outcome(record, 0)
        self.assertTrue(outcome["reject"])
        self.assertFalse(outcome["cover"])
        self.assertTrue(study.outcome(record, 0.3)["cover"])
        with self.assertRaisesRegex(ValueError, "disagreement"):
            study.outcome({**record, "p": 0.8}, 0)
        for invalid in [math.nan, math.inf, "nan"]:
            with self.assertRaises(ValueError):
                study.outcome({**record, "se": invalid}, 0)

    def test_failures_widen_identification_bounds(self):
        bounds = {name: [[0, 1]] * 11 for name in ["pointwise", "simultaneous"]}
        result = study.binary_measure([True, False, False, False], 10, bounds)
        self.assertEqual(result["unconditional_identification_bounds"], [0.1, 0.7])
        self.assertEqual(result["rate"], 0.25)
        empty = study.binary_measure([], 10, bounds)
        self.assertIsNone(empty["rate"])
        self.assertEqual(empty["unconditional_identification_bounds"], [0, 1])

    def test_exceptions_are_not_nonconvergence_or_known_nonsingularity(self):
        bounds = {name: [[0, 1]] * 5 for name in ["pointwise", "simultaneous"]}
        records = [
            {"converged": True, "singular": False},
            {"converged": False, "singular": True, "fit_error": "nonconvergence"},
            {"converged": False, "fit_error": "linear algebra exception"},
            {},
        ]
        counts = study.diagnostic_counts(records, bounds)
        self.assertEqual(
            counts["statuses"],
            {"converged": 1, "nonconverged": 1, "exception": 1, "unknown_convergence": 1},
        )
        self.assertEqual(counts["singular_unknown"], 2)
        self.assertEqual(counts["singular_rate"]["rate"], 0.5)
        self.assertEqual(
            counts["singular_rate"]["unconditional_identification_bounds"], [0.25, 0.75]
        )

    def test_immutable_inputs_and_confirmation_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            with self.assertRaises(ValueError):
                study.generate(path, 101)
            study.generate(path, 1)
            study.verify_inputs(path)
            with self.assertRaises(FileExistsError):
                study.generate(path, 1)
            with (path / "inputs.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                study.verify_inputs(path)

    def test_missing_paired_attempt_cannot_produce_a_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            study.generate(path, 1)
            (path / "paired.jsonl").write_text("")
            (path / "binomial-bounds.json").write_text(
                json.dumps({"pointwise": [], "simultaneous": []})
            )
            with self.assertRaisesRegex(ValueError, "missing paired attempts"):
                study.analyse(path)

    def test_reanalysis_preserves_original_outcomes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            paired = path / "paired.jsonl"
            paired.write_text(
                json.dumps(
                    {
                        "case": "fixture",
                        "replicate": 0,
                        "candidate": {"converged": False, "fit_error": "exception"},
                        "reference": {"converged": True, "singular": False},
                    }
                )
                + "\n"
            )
            bounds = path / "binomial-bounds.json"
            bounds.write_text(
                json.dumps({name: [[0, 1]] * 2 for name in ["pointwise", "simultaneous"]})
            )
            original = {
                "manifest": {"namespace": "pilot", "protocol": study.PROTOCOL, "sources": {}},
                "artifacts": {f.name: study.digest(f) for f in [paired, bounds]},
                "summaries": [{"fixture_outcome": 17}],
                "fit_summaries": [
                    {"case": "fixture", "engine": e, "attempted": 1, "diagnostics": {}}
                    for e in ["candidate", "reference"]
                ],
            }
            original_path = path / "summary.json"
            study.write_json(original_path, original)
            original_hash = study.digest(original_path)
            study.reanalyse(path)
            corrected = json.loads((path / "summary-reanalysis.json").read_text())
            self.assertEqual(study.digest(original_path), original_hash)
            self.assertEqual(corrected["summaries"], original["summaries"])
            self.assertEqual(corrected["fit_summaries"][0]["singular_unknown"], 1)
            self.assertEqual(corrected["analysis_revision"]["fits_repeated"], 0)
            with self.assertRaises(FileExistsError):
                study.reanalyse(path)
            paired.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "original evidence changed"):
                study.reanalyse(path)


if __name__ == "__main__":
    unittest.main()
