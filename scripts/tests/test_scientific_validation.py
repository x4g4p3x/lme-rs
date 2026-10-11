"""Independent analytic controls and fail-closed evidence reporting."""

import importlib.util
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "scientific_validation", Path(__file__).resolve().parents[1] / "scientific_validation.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class ScientificValidationTests(unittest.TestCase):
    def test_exact_inverse_including_row_pivot(self):
        inverse, determinant = validation.inverse_det([[0, 2], [3, 0]])
        self.assertEqual(determinant, -6)
        self.assertEqual(inverse, [[0, Fraction(1, 3)], [Fraction(1, 2), 0]])
        with self.assertRaises(ValueError):
            validation.inverse_det([[1, 2], [2, 4]])

    def test_balanced_lmm_matches_closed_form_information(self):
        validation.self_test()

    def test_gls_intercept_translation_preserves_likelihood(self):
        x, y, sigma2, tau2, beta, covariance, criterion = validation.balanced_case(True)
        variance = [
            [(tau2 if i // 4 == j // 4 else 0) + (sigma2 if i == j else 0) for j in range(16)]
            for i in range(16)
        ]
        shifted_beta, shifted_covariance, shifted_criterion = validation.gls_oracle(
            x, [value + 1000 for value in y], variance, True
        )
        self.assertEqual(shifted_beta, [beta[0] + 1000, beta[1]])
        self.assertEqual(shifted_covariance, covariance)
        self.assertEqual(shifted_criterion, criterion)

    def test_nonfinite_candidate_does_not_pass_a_contract(self):
        checks = validation.CaseChecks({"rtol": 1e-6, "atol": 1e-6})
        for value in [float("nan"), float("inf"), -float("inf")]:
            checks.close("coefficient", value, 1)
        self.assertTrue(all(not record["passed"] for record in checks.values))
        with self.assertRaises(ValueError):
            checks.vector("coefficients", [1], [1, 2])

    def test_nist_metadata_heading_is_not_an_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.dat"
            source.write_text(
                "Data: 1 Response Variable (y)\nPredictor metadata\n\nData: y x\n1 2\n3 4\n"
            )
            self.assertEqual(validation.nist_rows(source), [[1.0, 2.0], [3.0, 4.0]])

    def test_unknown_case_cannot_fall_back_to_a_balanced_fit(self):
        with self.assertRaisesRegex(ValueError, "undeclared numerical case"):
            validation.numerical_case("typo", None, None, None, None)


if __name__ == "__main__":
    unittest.main()
