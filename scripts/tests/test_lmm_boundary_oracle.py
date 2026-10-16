"""Closed-form optimizer checks against exact dense GLS, not a fitting engine."""

import importlib.util
import math
import unittest
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


oracle = module("lmm_boundary_oracle")
dense = module("scientific_validation")


class BoundaryOracleTests(unittest.TestCase):
    def test_clipping_formula_misses_reml_boundary_optimum(self):
        case = oracle.fixtures()[0]
        result = oracle.oracle(
            case["group_x"], [[Q(str(v)) for v in row] for row in case["observations"]]
        )
        self.assertAlmostEqual(float(result["sigma2"]), 5 / 3)
        self.assertAlmostEqual(float(result["slope_variance"]), 1 / 24)
        self.assertAlmostEqual(float(result["group_means_variance"]), 1 / 40)
        clipped = max(result["msb"], result["msw"])
        self.assertAlmostEqual(float(clipped / (result["m"] * result["sxx"])), 1 / 20)
        gap = oracle.criterion(result, clipped, clipped) - result["criterion"]
        self.assertAlmostEqual(gap, 6 * math.log(6 / 5) - 1, places=12)
        self.assertGreater(gap, 0)

    def test_closed_form_matches_independent_dense_gls(self):
        for case in oracle.fixtures():
            xg = list(map(Q, case["group_x"]))
            yg = [[Q(str(v)) for v in row] for row in case["observations"]]
            result = oracle.oracle(xg, yg)
            x = [[Q(1), v] for v in xg for _ in range(2)]
            y = [v for row in yg for v in row]
            variance = [
                [
                    (result["tau2"] if i // 2 == j // 2 else Q(0))
                    + (result["sigma2"] if i == j else Q(0))
                    for j in range(8)
                ]
                for i in range(8)
            ]
            beta, covariance, criterion = dense.gls_oracle(x, y, variance, True)
            self.assertEqual(beta, [result["beta0"], result["beta1"]])
            self.assertEqual(covariance[1][1], result["slope_variance"])
            self.assertAlmostEqual(criterion, result["criterion"], places=12)

    def test_boundary_and_adjacent_interior_are_distinct(self):
        fixtures = {c["id"]: c["expected"] for c in oracle.fixtures()}
        self.assertEqual(fixtures["exact_boundary"]["tau2"], 0)
        self.assertEqual(fixtures["near_boundary_below"]["tau2"], 0)
        self.assertAlmostEqual(fixtures["near_boundary_above"]["tau2"], 0.002001)
        for name in ["strict_boundary", "near_boundary_below"]:
            value = fixtures[name]
            self.assertLess(value["msb"], value["v"])
            self.assertLess(value["v"], value["msw"])

    def test_invalid_designs_are_not_silently_fitted(self):
        for x, y in [
            ([1, 2], [[1, 2], [2, 3]]),
            ([1, 1, 1], [[1, 2]] * 3),
            ([1, 2, 3], [[1, 2], [2, 3], [3, 4, 5]]),
            ([1, 2, 3], [[1, 1], [2, 2], [3, 3]]),
        ]:
            with self.assertRaises(ValueError):
                oracle.oracle(x, y)


if __name__ == "__main__":
    unittest.main()
