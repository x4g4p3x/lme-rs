"""Independent identities, finite differences and evidence-integrity negatives."""

import copy
import json
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_lmm_satterthwaite_dense as dense  # noqa: E402


def finite_differences(function, point, step):
    np = dense.np
    n = len(point)
    score, hessian = np.zeros(n), np.zeros((n, n))
    base = function(point)
    directions = np.eye(n) * step
    for i, direction in enumerate(directions):
        plus, minus = function(point + direction), function(point - direction)
        score[i] = (plus - minus) / (2 * step)
        hessian[i, i] = (plus - 2 * base + minus) / step**2
        for j in range(i):
            other = directions[j]
            mixed = (
                function(point + direction + other)
                - function(point + direction - other)
                - function(point - direction + other)
                + function(point - direction - other)
            ) / (4 * step**2)
            hessian[i, j] = hessian[j, i] = mixed
    return score, hessian


@unittest.skipUnless(dense.np is not None, "NumPy is optional; dense numerical controls skipped")
class DenseDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.np = dense.np
        self.x = [-1, 0, 1] * 3
        self.y = [1, 2, 5, 2, 0, 1, 2, 4, 6]
        self.model = dense.DenseReml(self.x, self.y, [0] * 3 + [1] * 3 + [2] * 3)

    def test_physical_score_hessian_and_covariance_gradient_against_finite_differences(self):
        point = self.np.array([1.2, 0.7, 0.12, 0.35])
        actual = self.model.evaluate(point)
        score, hessian = finite_differences(
            lambda eta: self.model.state(eta)["criterion"], point, 0.0003
        )
        self.np.testing.assert_allclose(actual["score"], score, atol=1e-6, rtol=1e-6)
        self.np.testing.assert_allclose(actual["hessian"], hessian, atol=5e-6, rtol=2e-5)
        gradient, _ = finite_differences(
            lambda eta: self.model.state(eta)["covariance"][1, 1], point, 0.0001
        )
        self.np.testing.assert_allclose(actual["variance_gradient"], gradient, atol=1e-8, rtol=1e-6)

    def test_nonstationary_transformation_requires_score_times_second_derivatives(self):
        point = self.np.array([0.8, 0.15, 0.45, 1.1])
        base = 0.9

        def physical(parameters):
            # Construct L L' directly, without using the analytic transformation.
            a, b, c, relative = parameters
            factor = self.np.array([[a, 0.0], [b, c]])
            covariance = base * relative * (factor @ factor.T)
            return [base * relative, covariance[0, 0], covariance[0, 1], covariance[1, 1]]

        actual = dense.relative_derivatives(self.model, point, base)
        score, hessian = finite_differences(
            lambda rho: self.model.state(physical(rho))["criterion"], point, 0.0001
        )
        self.np.testing.assert_allclose(actual["relative_score"], score, atol=2e-6, rtol=1e-6)
        self.np.testing.assert_allclose(actual["relative_hessian"], hessian, atol=2e-5, rtol=1e-5)
        gradient, _ = finite_differences(
            lambda rho: self.model.state(physical(rho))["covariance"][1, 1], point, 0.0001
        )
        self.np.testing.assert_allclose(
            actual["relative_variance_gradient"], gradient, atol=1e-8, rtol=1e-6
        )
        self.assertGreater(self.np.linalg.norm(actual["transformation_curvature"]), 0.1)
        wrong = actual["relative_hessian"] - actual["transformation_curvature"]
        self.assertGreater(self.np.linalg.norm(wrong - hessian), 0.1)

    def test_zero_random_covariance_reduces_to_exact_ols_identities(self):
        s = Q(3, 2)
        n = len(self.y)
        x = list(map(Q, self.x))
        y = list(map(Q, self.y))
        mean = sum(y) / n
        slope = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)
        rss = sum((b - mean - slope * a) ** 2 for a, b in zip(x, y))
        actual = self.model.evaluate([float(s), 0, 0, 0])
        self.assertAlmostEqual(actual["beta"][1], float(slope), places=13)
        self.assertAlmostEqual(actual["covariance"][1, 1], float(s / 6), places=13)
        self.assertAlmostEqual(actual["variance_gradient"][0], 1 / 6, places=13)
        self.assertAlmostEqual(actual["score"][0], float((n - 2) / s - rss / s**2), places=12)
        self.assertAlmostEqual(
            actual["hessian"][0, 0], float(-(n - 2) / s**2 + 2 * rss / s**3), places=12
        )

    def test_invalid_design_or_nonpositive_variance_is_rejected_without_jitter(self):
        with self.assertRaises(ValueError):
            dense.DenseReml([1, 1, 1], [1, 2, 3], [0, 0, 1])
        with self.assertRaises(ValueError):
            self.model.state([0, 0, 0, 0])
        with self.assertRaises(self.np.linalg.LinAlgError):
            self.model.state([1, -10, 0, 0])


class EvidenceTests(unittest.TestCase):
    def test_optional_dependency_error_is_explicit(self):
        with patch.object(dense, "np", None):
            with self.assertRaisesRegex(RuntimeError, "NumPy is optional"):
                dense.require_numpy()

    def test_input_hash_and_duplicate_candidate_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = [{"case": {"id": "case"}, "replicate": 2}]
            candidate = [{"case": "case", "replicate": 2, "control": {"tolerance": 1e-12}}]
            (root / "inputs.json").write_text(json.dumps(inputs))
            controls = {
                "inputs_sha256": dense.digest(root / "inputs.json"),
                "selected": [["case", 2]],
                "controls": [{"tolerance": 1e-12}],
            }
            (root / "controls.json").write_text(json.dumps(controls))
            (root / "candidate.json").write_text(json.dumps(candidate))
            dense.read_evidence(root)
            duplicated = candidate + copy.deepcopy(candidate)
            (root / "candidate.json").write_text(json.dumps(duplicated))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                dense.read_evidence(root)
            (root / "inputs.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "hash"):
                dense.read_evidence(root)


if __name__ == "__main__":
    unittest.main()
