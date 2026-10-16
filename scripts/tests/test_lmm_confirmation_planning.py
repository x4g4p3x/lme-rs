"""Independent rational identities for the proposed 72-claim planner."""

import copy
import json
import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose_lmm_random_slope_reference as reference  # noqa: E402
import plan_lmm_random_slope_confirmation as plan  # noqa: E402


def mass(n, k, p):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


class ReferencePolicyTests(unittest.TestCase):
    def test_selection_ignores_inference_and_ineligible_lower_objective(self):
        values = [
            {
                "variant": "baseline",
                "eligible": True,
                "fit": {"criterion": 100, "methods": {"p": 0.5}},
            },
            {
                "variant": "tight_positive",
                "eligible": True,
                "fit": {"criterion": 99, "methods": {"error": "bad inference"}},
            },
            {"variant": "tight_negative", "eligible": False, "fit": {"criterion": 98}},
        ]
        self.assertEqual(reference.select_best(values)["variant"], "tight_positive")
        self.assertIsNone(reference.select_best([values[-1]]))

    def test_exact_ties_use_fixed_bank_order_even_if_rows_are_shuffled(self):
        values = [
            {"variant": v["id"], "eligible": True, "fit": {"criterion": 100}}
            for v in reference.VARIANTS
        ]
        self.assertEqual(reference.select_best(list(reversed(values)))["variant"], "baseline")


@unittest.skipIf(plan.scipy is None, "optional SciPy is not installed")
class PlanningTests(unittest.TestCase):
    def test_claim_inventory_cannot_drop_cases_or_methods(self):
        config = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "scientific/RANDOM_SLOPE_CONFIRMATION_PLANNING_INPUTS.json"
            ).read_text()
        )
        plan.validate_inputs(config)
        for key, value in (
            ("family", 48),
            ("primary_methods", ["satterthwaite"]),
            ("primary_cases", config["primary_cases"][:-1]),
            ("replicate_grid", [22000, 20000]),
        ):
            bad = copy.deepcopy(config)
            bad[key] = value
            with self.assertRaises(ValueError):
                plan.validate_inputs(bad)

    def test_acceptance_counts_match_rational_binomial_tail_inversion(self):
        alpha, family = Fraction(1, 5), 2
        for n in (1, 4, 12, 40):
            for lo, hi in ((Fraction(1, 5), Fraction(4, 5)), (Fraction(0), Fraction(1))):
                tail = alpha / (2 * family)
                accepted = [
                    k
                    for k in range(n + 1)
                    if (lo == 0 or sum(mass(n, j, lo) for j in range(k, n + 1)) <= tail)
                    and (hi == 1 or sum(mass(n, j, hi) for j in range(k + 1)) <= tail)
                ]
                lower, upper = plan.acceptance_counts(
                    n, float(alpha), family, (float(lo), float(hi))
                )
                self.assertEqual(list(range(lower, upper + 1)), accepted)

    def test_power_matches_exact_missing_completion_counts(self):
        n, family = 40, 2
        lo, hi = plan.acceptance_counts(n, 0.2, family, [0.2, 0.8])
        for missing in (0, 3, 20, 40):
            for p in (Fraction(0), Fraction(1, 3), Fraction(1)):
                expected = sum(
                    mass(n - missing, k, p)
                    for k in range(n - missing + 1)
                    if lo <= k and k + missing <= hi
                )
                row = plan.power(n, missing, float(p), 0.2, family, [0.2, 0.8])
                self.assertAlmostEqual(
                    row["single_claim_decision_power"], float(expected), places=13
                )
                self.assertAlmostEqual(
                    row["joint_decision_power_lower_bound"],
                    max(0.0, 1 - family * float(1 - expected)),
                    places=13,
                )

    def test_cp_endpoints_symmetry_and_accepted_neighbors(self):
        tail = 0.05 / 144
        self.assertEqual(plan.cp_interval(0, 0, 0.05, 72), [0, 1])
        self.assertAlmostEqual(plan.cp_interval(0, 40, 0.05, 72)[1], 1 - tail ** (1 / 40))
        self.assertAlmostEqual(plan.cp_interval(40, 40, 0.05, 72)[0], tail ** (1 / 40))
        for n in (20000, 22000, 30000):
            a, b = plan.acceptance_counts(n, 0.05, 72, [0.04, 0.06])
            for k in (a - 1, a, b, b + 1):
                interval = plan.cp_interval(k, n, 0.05, 72)
                self.assertEqual(interval[0] >= 0.04 and interval[1] <= 0.06, a <= k <= b)
                opposite = plan.cp_interval(n - k, n, 0.05, 72)
                self.assertAlmostEqual(interval[0], 1 - opposite[1], places=14)

    def test_empty_intervals_and_no_observed_outcomes(self):
        self.assertGreater(*plan.acceptance_counts(5, 0.05, 72, [0.04, 0.06]))
        self.assertEqual(
            plan.power(5, 0, 0.05, 0.05, 72, [0.04, 0.06])["single_claim_decision_power"], 0
        )
        self.assertEqual(
            plan.power(30000, 30000, 0.05, 0.05, 72, [0.04, 0.06])["single_claim_decision_power"], 0
        )
        self.assertEqual(plan.power(5, 5, 0.5, 0.05, 72, [0, 1])["single_claim_decision_power"], 1)

    def test_decimal_failure_rounding_and_invalid_inputs(self):
        self.assertEqual(plan.failure_count(100, 0.07), 7)  # float product is 7.000000000000001
        self.assertEqual(plan.failure_count(22000, 0.0001), 3)
        for n, missing, p in (
            (True, 0, 0.5),
            (0, 0, 0.5),
            (4, 5, 0.5),
            (4, -1, 0.5),
            (4, 0, float("nan")),
        ):
            with self.assertRaises(ValueError):
                plan.power(n, missing, p, 0.05, 72, [0.04, 0.06])
        for alpha, family, band in (
            (0, 72, [0.04, 0.06]),
            (0.05, 0, [0.04, 0.06]),
            (0.05, 72, [0.06, 0.04]),
        ):
            with self.assertRaises(ValueError):
                plan.acceptance_counts(100, alpha, family, band)

    def test_joint_bound_is_union_bound_without_independence_assumption(self):
        row = plan.power(22000, 0, 0.05, 0.05, 72, [0.04, 0.06])
        self.assertAlmostEqual(row["joint_decision_power_lower_bound"], 0.924261523249, places=10)
        self.assertLess(
            row["joint_decision_power_lower_bound"], row["single_claim_decision_power"] ** 72
        )
        self.assertGreater(
            plan.power(30000, 30, 0.05, 0.05, 72, [0.04, 0.06])["joint_decision_power_lower_bound"],
            0.98,
        )


if __name__ == "__main__":
    unittest.main()
