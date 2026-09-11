"""Regression tests for benchmark claims; no Rust or Julia runtime required."""

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_evidence import assess_timing, check_result, fit_agreement
from build_benchmark_site import transform_fair
from run_fair_rust_julia_benchmark import (
    FAIR_CASES,
    check_optimizer_backend,
    compare_case,
    parse_args,
)

CASE = FAIR_CASES["sleepstudy_reml"]


def result(implementation, seconds=1.0, repeats=10):
    return {
        "implementation": implementation,
        "case": CASE.name,
        "model": CASE.model,
        "formula": CASE.formula,
        "reml": True,
        "n_obs": 180,
        "repeats": repeats,
        "warmups": 2,
        "cold_fit": {
            "samples_seconds": [seconds] * repeats,
            "summary": {"median_seconds": seconds},
        },
        "fit_checks": [
            {"objective": 100.0, "coefficients": [250.0, 10.0], "converged": True}
            for _ in range(repeats)
        ],
    }


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_wrong_or_unidentified_rust_backend_is_rejected(self):
        for features, expected, wrong in [
            ("", "argmin", "basin"),
            ("basin", "basin", "argmin"),
        ]:
            check_optimizer_backend(
                {"implementation": "rust", "optimizer_backend": expected}, features
            )
            for backend in [wrong, None]:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    check_optimizer_backend(
                        {"implementation": "rust", "optimizer_backend": backend},
                        features,
                    )

    def test_rust_backend_selection_does_not_reject_julia(self):
        check_optimizer_backend(result("julia"), "basin")
        check_optimizer_backend(
            {"implementation": "rust", "optimizer_backend": "basin"},
            "perf-diagnostics, basin",
        )

    def test_invalid_measurement_settings_are_rejected(self):
        for args in (
            ["--repeats", "0"],
            ["--warmups", "-1"],
            ["--threads", "0"],
            ["--target-ratio", "nan"],
        ):
            with patch.object(sys, "argv", ["benchmark", *args]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()

    def test_dashboard_excludes_unverified_ratios_but_preserves_reason(self):
        a, b = result("rust", 0.1), result("julia")
        a["fit_checks"][0]["converged"] = False
        payload = {
            "results": [a, b],
            "comparisons": [{"case": CASE.name, "metrics": compare_case(CASE, a, b, 1.0)}],
        }
        report = transform_fair(payload, label="test", source_path="test.json")
        self.assertEqual(report["summary"]["cold_fit_cases"], 0)
        self.assertIsNone(report["summary"]["geometric_mean_rust_over_julia_cold_fit"])
        metric = report["cases"][0]["cold_fit"]
        self.assertEqual(metric["rust_over_julia_median"], 0.1)
        self.assertIn("convergence", metric["fit_agreement"]["reasons"][0])

    def test_faster_but_wrong_fit_has_no_speed_claim(self):
        a, b = result("rust", 0.1), result("julia")
        a["fit_checks"][3]["objective"] = 200.0
        row = compare_case(CASE, a, b, 1.0)[0]
        self.assertEqual(row["rust_over_julia_median"], 0.1)
        self.assertFalse(row["eligible_for_speed_claim"])
        self.assertIsNone(row["meets_target"])

    def test_one_nonconverged_repeat_invalidates_comparison(self):
        a, b = result("rust", 0.1), result("julia")
        a["fit_checks"][-1]["converged"] = False
        self.assertEqual(fit_agreement(CASE, a, b)["status"], "unverified")

    def test_coefficients_and_observations_must_match(self):
        for key, value in [("coefficients", [250.0, 40.0]), ("objective", None)]:
            a, b = result("rust"), result("julia")
            a["fit_checks"][0][key] = value
            self.assertEqual(fit_agreement(CASE, a, b)["status"], "unverified")
        a["n_obs"] = 179
        self.assertIn("observation counts differ", fit_agreement(CASE, a, b)["reasons"])

    def test_smoke_run_cannot_claim_a_win(self):
        a, b = result("rust", 0.1, repeats=3), result("julia", repeats=3)
        row = compare_case(CASE, a, b, 1.0)[0]
        self.assertEqual(row["sampling_status"], "smoke_only")
        self.assertIsNone(row["ratio_interval_95"])
        self.assertIsNone(row["meets_target"])

    def test_unwarmed_run_cannot_claim_a_win(self):
        a, b = result("rust", 0.1), result("julia")
        a["warmups"] = 0
        self.assertFalse(compare_case(CASE, a, b, 1.0)[0]["eligible_for_speed_claim"])

    def test_separated_timings_and_tie(self):
        a, b = result("rust", 0.5), result("julia")
        row = compare_case(CASE, a, b, 1.0)[0]
        self.assertEqual(row["ratio_interval_95"], [0.5, 0.5])
        self.assertEqual(row["faster_implementation"], "rust")
        self.assertTrue(row["meets_target"])
        a = result("rust")
        self.assertEqual(compare_case(CASE, a, b, 1.0)[0]["faster_implementation"], "inconclusive")

    def test_overlapping_samples_are_inconclusive(self):
        a, b = result("rust"), result("julia")
        a["cold_fit"]["samples_seconds"] = [0.7, 1.3] * 5
        b["cold_fit"]["samples_seconds"] = [0.8, 1.2] * 5
        row = assess_timing(a, b, {"status": "passed"}, 1.0)
        self.assertEqual(row["faster_implementation"], "inconclusive")
        self.assertIsNone(row["meets_target"])

    def test_glmm_not_implicitly_equivalent(self):
        a, b = result("rust"), result("julia")
        self.assertEqual(
            fit_agreement(FAIR_CASES["cbpp_binomial_ml"], a, b)["status"], "unverified"
        )

    def test_prepared_vs_cold_is_diagnostic_only(self):
        a, b = result("rust", 0.5), result("julia")
        a["fit_prepared"] = copy.deepcopy(a["cold_fit"])
        row = compare_case(CASE, a, b, 1.0)[1]
        self.assertEqual(row["faster_implementation"], "not_comparable")
        self.assertIsNone(row["meets_target"])

    def test_recompute_summary_and_reject_invalid_samples(self):
        a = result("rust")
        a["cold_fit"]["summary"]["median_seconds"] = 0.01
        check_result(a, CASE, "rust", 10)
        self.assertEqual(a["cold_fit"]["summary"]["median_seconds"], 1.0)
        for value in [0, -1, float("nan"), float("inf")]:
            a = result("rust")
            a["cold_fit"]["samples_seconds"][0] = value
            with self.assertRaises(ValueError):
                check_result(a, CASE, "rust", 10)

    def test_wrong_contract_or_missing_repeat_rejected(self):
        for key, value in [("reml", False), ("formula", "y ~ 1"), ("fit_checks", [])]:
            a = result("rust")
            a[key] = value
            with self.assertRaises(ValueError):
                check_result(a, CASE, "rust", 10)
        a = result("rust")
        a["cold_fit"]["samples_seconds"].pop()
        with self.assertRaises(ValueError):
            check_result(a, CASE, "rust", 10)


if __name__ == "__main__":
    unittest.main()
