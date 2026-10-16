"""Prospective controls, seed separation and analytic diagnostics for v2."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_lmm_random_slope_controls as controls  # noqa: E402
import lmm_random_slope as v1  # noqa: E402
import lmm_random_slope_v2 as v2  # noqa: E402
import report_lmm_random_slope_v2 as reporting  # noqa: E402
from lmm_random_slope_optimum import optimum  # noqa: E402


class V2Tests(unittest.TestCase):
    def test_pair_disagreements_are_not_determined_by_marginal_counts(self):
        a = [(True, False), (False, True), None]
        self.assertEqual(reporting.pair_counts(a, a), {"evaluable": 2, "reject": 0, "cover": 0})
        self.assertEqual(
            reporting.pair_counts(a, list(reversed(a[:2])) + [None]),
            {"evaluable": 2, "reject": 2, "cover": 2},
        )

    def test_balanced_report_rejects_missing_metrics_or_unknown_rank(self):
        checks = [
            {"metric": name, "difference": 0, "limit": 1, "passed": True}
            for name in ["criterion", "sigma2", "covariance_00", "covariance_01", "covariance_11"]
        ]
        row = {"evaluable": True, "rank_match": True, "checks": checks}
        reporting.validate_balanced(row)
        for bad in [
            dict(row, checks=checks[:-1]),
            dict(row, rank_match=None),
            dict(row, checks=[dict(checks[0], passed=False)] + checks[1:]),
        ]:
            with self.assertRaises(ValueError):
                reporting.validate_balanced(bad)

    def test_disjoint_seeds_order_independence_and_unchanged_design_gates(self):
        self.assertEqual(v2.registry(), v1.registry())
        self.assertEqual(v2.GATES, v1.GATES)
        self.assertNotEqual(v2.PROTOCOL, v1.PROTOCOL)
        self.assertEqual(
            v2.CONTROL, {"tolerance": 1e-12, "max_iterations": 5000, "require_convergence": False}
        )
        for case in v2.registry():
            for replicate in (0, 1):
                pilot = v2.response(case, replicate, "pilot")
                self.assertEqual(pilot, v2.response(case, replicate, "pilot"))
                self.assertNotEqual(pilot, v1.response(case, replicate, "pilot"))
                self.assertNotEqual(pilot, v2.response(case, replicate, "smoke"))
        case = v2.registry()[0]
        expected = [v2.response(case, i, "pilot") for i in range(3)]
        self.assertEqual(
            expected, list(reversed([v2.response(case, i, "pilot") for i in reversed(range(3))]))
        )

    def test_optimum_precision_rank_and_unevaluable_flags_are_distinct(self):
        case = {"sizes": [4] * 4, "x": [-1.0, -1.0, 1.0, 1.0] * 4}
        # Two identical slope effects +/-1 and orthogonal within residuals.
        y = [
            2 + a + b * x + e
            for a, b in [(1, 1), (-1, -1), (1, -1), (-1, 1)]
            for x, e in [(-1, 1), (-1, -1), (1, -1), (1, 1)]
        ]
        expected = optimum(case["x"][:4], [y[i : i + 4] for i in range(0, 16, 4)])
        fit = {
            "converged": True,
            "criterion": expected["criterion"],
            "sigma2": expected["sigma2"],
            "covariance": expected["covariance"],
        }
        result = v2.balanced_checks(case, y, fit)
        self.assertTrue(result["evaluable"])
        self.assertTrue(result["rank_match"])
        self.assertTrue(all(c["passed"] for c in result["checks"]))
        bad = copy.deepcopy(fit)
        bad["criterion"] += 1
        bad["covariance"][0][1] += 0.1
        bad["covariance"][1][0] += 0.1
        result = v2.balanced_checks(case, y, bad)
        self.assertEqual(
            {c["metric"] for c in result["checks"] if not c["passed"]},
            {"criterion", "covariance_01"},
        )
        self.assertFalse(v2.balanced_checks(case, y, {})["evaluable"])
        self.assertIsNone(v2.balanced_checks({"sizes": [3, 4]}, [], {}))
        invalid = copy.deepcopy(fit)
        invalid["criterion"] = float("nan")
        self.assertFalse(v2.balanced_checks(case, y, invalid)["evaluable"])

    def test_rank_miss_can_occur_inside_entrywise_precision_gates(self):
        case = {"sizes": [4] * 4, "x": [-1.0, -1.0, 1.0, 1.0] * 4}
        y = [
            2 + a + b * x + e
            for a, b in [(0.25, 0), (-0.25, 0), (0, 0.5), (0, -0.5)]
            for x, e in [(-1, 1), (-1, -1), (1, -1), (1, 1)]
        ]
        expected = optimum(case["x"][:4], [y[i : i + 4] for i in range(0, 16, 4)])
        self.assertEqual(expected["binding_directions"], 2)
        fit = {
            "converged": True,
            "criterion": expected["criterion"],
            "sigma2": expected["sigma2"],
            "covariance": [[1e-7, 0], [0, 1e-7]],
        }
        result = v2.balanced_checks(case, y, fit)
        self.assertTrue(result["evaluable"])
        self.assertFalse(result["rank_match"])
        self.assertTrue(all(c["passed"] for c in result["checks"]))

    def test_selected_diagnostic_cannot_overwrite_or_nest_in_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original"
            source.mkdir()
            for destination in (source, source / "diagnostic", source.parent):
                with self.assertRaises(ValueError):
                    controls.prepare(source, destination)
            destination = Path(directory) / "existing"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                controls.prepare(source, destination)


if __name__ == "__main__":
    unittest.main()
