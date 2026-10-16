"""Rational dense GLS controls for the post-pilot global optimum derivation."""

import importlib.util
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lmm_random_slope_optimum as oracle  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "dense_optimum_control", SCRIPTS / "scientific_validation.py"
)
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


class BalancedOptimumTests(unittest.TestCase):
    def test_all_three_active_sets_with_rotated_rational_covariance(self):
        x = list(map(Q, [-1, -1, 1, 1]))
        rotation = [[Q(3, 5), Q(-4, 5)], [Q(4, 5), Q(3, 5)]]
        for h, k, sigma2, binding in (
            (Q(1), Q(2), Q(2), 0),
            (Q(1, 2), Q(2), Q(18, 11), 1),
            (Q(1, 4), Q(1, 2), Q(37, 28), 2),
        ):
            offsets = [
                [h * rotation[j][0] for j in range(2)],
                [-h * rotation[j][0] for j in range(2)],
                [k * rotation[j][1] for j in range(2)],
                [-k * rotation[j][1] for j in range(2)],
            ]
            y = [
                [2 + a + (Q(4, 5) + b) * v + e for v, e in zip(x, map(Q, [1, -1, -1, 1]))]
                for a, b in offsets
            ]
            actual = oracle.optimum(list(map(float, x)), [[float(v) for v in row] for row in y])
            self.assertEqual(actual["binding_directions"], binding)
            self.assertAlmostEqual(actual["sigma2"], float(sigma2), places=13)
            lam = [8 * h * h / 3, 8 * k * k / 3]
            covariance = [
                [
                    sum(
                        rotation[i][r] * max(Q(0), lam[r] - sigma2) * rotation[j][r]
                        for r in range(2)
                    )
                    / 4
                    for j in range(2)
                ]
                for i in range(2)
            ]
            design = [[Q(1), v] for _ in range(4) for v in x]
            variance = [
                [
                    (sigma2 if i == j else 0)
                    + (
                        sum(
                            design[i][a] * covariance[a][b] * design[j][b]
                            for a in range(2)
                            for b in range(2)
                        )
                        if i // 4 == j // 4
                        else 0
                    )
                    for j in range(16)
                ]
                for i in range(16)
            ]
            beta, fixed_cov, criterion = dense.gls_oracle(
                design, [v for row in y for v in row], variance, True
            )
            for i in range(2):
                self.assertAlmostEqual(actual["coefficients"][i], float(beta[i]), places=13)
                for j in range(2):
                    self.assertAlmostEqual(
                        actual["covariance"][i][j], float(covariance[i][j]), places=13
                    )
                    self.assertAlmostEqual(
                        actual["fixed_covariance"][i][j], float(fixed_cov[i][j]), places=13
                    )
            self.assertAlmostEqual(actual["criterion"], criterion, places=12)

    def test_nonidentifiable_or_degenerate_designs_are_rejected(self):
        for x, y in (
            ([-1, 1], [[1, 2]] * 4),
            ([-1, 0, 1], [[1, 2, 3]] * 4),
            ([0, 0, 0], [[1, 4, 2]] * 4),
        ):
            with self.assertRaises(ValueError):
                oracle.optimum(x, y)


if __name__ == "__main__":
    unittest.main()
