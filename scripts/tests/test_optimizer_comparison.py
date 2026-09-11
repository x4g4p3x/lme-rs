"""Default-backend evidence must include every process and every measured fit."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_optimizer_comparison import ORDERS, summarize


def reports():
    runs = []
    for block, order in enumerate(ORDERS):
        for position, letter in enumerate(order):
            backend = {"A": "argmin", "B": "basin"}[letter]
            result = dict(
                case="fixture",
                optimizer_backend=backend,
                cold_fit=dict(samples_seconds=[1.0 if letter == "A" else 0.9] * 11),
                fit_checks=[
                    dict(objective=100.0, coefficients=[1.0], converged=True) for _ in range(11)
                ],
            )
            runs.append(
                dict(
                    block=block,
                    position=position,
                    backend=backend,
                    report=dict(
                        results=[result],
                        provenance={
                            field: "same-hash"
                            for field in [
                                "source_content_sha256",
                                "cargo_lock_sha256",
                                "harness_sha256",
                            ]
                        },
                    ),
                )
            )
    return runs


class OptimizerComparisonTests(unittest.TestCase):
    def test_paired_ratio_uses_independent_blocks(self):
        result = summarize(reports())[0]
        self.assertTrue(result["fit_agreement"])
        self.assertEqual(result["cold_fit"]["block_ratios"], [0.9] * 3)
        self.assertEqual(result["cold_fit"]["ratio_interval_95"], [0.9, 0.9])
        self.assertTrue(result["cold_fit"]["competitive"])

    def test_one_bad_measured_fit_disqualifies_the_case(self):
        for change in [
            dict(converged=False),
            dict(objective=101.0),
            dict(coefficients=[2.0]),
            dict(objective=float("inf")),
            dict(coefficients=[]),
        ]:
            runs = reports()
            runs[-1]["report"]["results"][0]["fit_checks"][5].update(change)
            result = summarize(runs)[0]
            self.assertFalse(result["fit_agreement"])
            self.assertFalse(result["cold_fit"]["competitive"])

    def test_missing_checks_or_wrong_backend_are_not_evidence(self):
        for key, value in [("fit_checks", []), ("optimizer_backend", "argmin")]:
            runs = reports()
            runs[1]["report"]["results"][0][key] = value
            self.assertFalse(summarize(runs)[0]["fit_agreement"])

    def test_incomplete_processes_or_coverage_are_rejected(self):
        runs = reports()
        with self.assertRaises(ValueError):
            summarize(runs[:-1])
        missing_case = copy.deepcopy(runs)
        missing_case[-1]["report"]["results"] = []
        with self.assertRaises(ValueError):
            summarize(missing_case)
        changed_source = copy.deepcopy(runs)
        changed_source[-1]["report"]["provenance"]["source_content_sha256"] = "changed"
        with self.assertRaises(ValueError):
            summarize(changed_source)


if __name__ == "__main__":
    unittest.main()
