"""Independent exact controls and integrity negatives for the slope pilot."""

import copy
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lmm_random_slope as slope  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "dense_slope_oracle", SCRIPTS / "scientific_validation.py"
)
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


class RandomSlopeTests(unittest.TestCase):
    def test_freeze_input_design_tampering_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.txt").write_text("original")
            frozen = root / "freeze"
            (frozen / "frozen-sources").mkdir(parents=True)
            shutil.copyfile(root / "source.txt", frozen / "frozen-sources/source.txt")
            (frozen / "wheel.whl").write_bytes(b"test wheel")
            (frozen / "requirements.txt").write_text("test==1\n")
            slope.shared.write_json(
                frozen / "freeze.json",
                {
                    "protocol": slope.PROTOCOL,
                    "control": slope.CONTROL,
                    "gates": slope.GATES,
                    "registry_sha256": slope.shared.json_digest(slope.registry()),
                    "sources": {"source.txt": slope.shared.digest(root / "source.txt")},
                    "wheel": {
                        "name": "wheel.whl",
                        "sha256": slope.shared.digest(frozen / "wheel.whl"),
                    },
                    "requirements_sha256": slope.shared.digest(frozen / "requirements.txt"),
                },
            )
            run = root / "smoke"
            with patch.object(slope, "ROOT", root), patch.object(slope, "SOURCES", ["source.txt"]):
                slope.generate(run, frozen, "smoke")
                slope.verify_run(run)
                with self.assertRaises(FileExistsError):
                    slope.generate(run, frozen, "smoke")
                case = slope.load(run / "designs.json")[0]
                path = run / "cases" / case["id"] / "inputs.jsonl.gz"
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "input hash"):
                    slope.verify_run(run)
                path.write_bytes(original)
                (root / "source.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "active source"):
                    slope.verify_freeze(frozen)
                (root / "source.txt").write_text("original")
                (frozen / "wheel.whl").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "wheel changed"):
                    slope.verify_freeze(frozen)
                designs = slope.load(run / "designs.json")
                designs[0]["x"][0] += 0.1
                (run / "designs.json").write_text(json.dumps(designs))
                manifest = slope.load(run / "manifest.json")
                manifest["designs_sha256"] = slope.shared.digest(run / "designs.json")
                (run / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "registry/budget"):
                    slope.verify_run(run)

    def test_native_panics_are_retained_but_interrupts_propagate(self):
        class Panic(BaseException):
            pass

        for error in (Panic("synthetic native panic"), ValueError("fit failed")):
            fake = SimpleNamespace(FitControl=lambda **kwargs: kwargs)
            with patch.object(fake, "lmer", side_effect=error, create=True):
                result = slope.candidate_fit(fake, None)
            self.assertFalse(result["converged"])
            self.assertIn(str(error), result["fit_error"])
        for error in (KeyboardInterrupt(), SystemExit()):
            fake = SimpleNamespace(FitControl=lambda **kwargs: kwargs)
            with patch.object(fake, "lmer", side_effect=error, create=True):
                with self.assertRaises(type(error)):
                    slope.candidate_fit(fake, None)

    def test_enrichment_cannot_change_retained_candidate_values(self):
        raw = {
            "beta": 0.1,
            "converged": True,
            "methods": {"normal": {"beta": 0.1, "se": 0.2, "p": 0.6, "df": None, "f": 0.25}},
        }
        enriched = copy.deepcopy(raw)
        enriched["methods"]["normal"].update(critical=1.96, distribution_p=0.6)
        result = slope.verified_enrichment(raw, enriched)
        self.assertEqual(result["beta"], raw["beta"])
        for section, field in ((enriched, "beta"), (enriched["methods"]["normal"], "se")):
            original = section[field]
            section[field] += 0.001
            with self.assertRaises(ValueError):
                slope.verified_enrichment(raw, enriched)
            section[field] = original
        wrapper = copy.deepcopy(raw)
        wrapper["methods"]["normal"] = {
            "error": "bad distribution",
            "raw": raw["methods"]["normal"],
        }
        self.assertIn("error", slope.verified_enrichment(raw, wrapper)["methods"]["normal"])

    def test_registry_counts_ids_covariances_and_normalization(self):
        cases = slope.registry()
        self.assertEqual(len(cases), 94)
        self.assertEqual(len({c["id"] for c in cases}), 94)
        self.assertEqual(sum(c["primary_future"] for c in cases), 36)
        self.assertEqual(
            Counter(c["stratum"] for c in cases),
            {
                "interior": 72,
                "zero_slope": 12,
                "near_correlation": 4,
                "exact_correlation": 4,
                "imbalanced": 2,
            },
        )
        for case in cases:
            start = 0
            for n in case["sizes"]:
                x = case["x"][start : start + n]
                self.assertAlmostEqual(sum(x), 0, places=12)
                self.assertAlmostEqual(sum(v * v for v in x), n, places=12)
                start += n
            sigma = case["sigma"]
            self.assertGreaterEqual(sigma[0][0] * sigma[1][1] - sigma[0][1] ** 2, -1e-16)
            if case["v1"] == 0:
                self.assertIsNone(case["correlation"])
                self.assertEqual(sigma[0][1], 0)

    def test_seed_separation_and_worker_order(self):
        cases = slope.registry()
        original = {c["id"]: slope.response(c, 0, "pilot") for c in cases}
        shuffled = {
            c["id"]: slope.response(c, 0, "pilot") for worker in range(4) for c in cases[worker::4]
        }
        self.assertEqual(original, shuffled)
        for namespace in ("smoke", "confirmation", "analysis"):
            self.assertNotEqual(original[cases[0]["id"]], slope.response(cases[0], 0, namespace))
        self.assertNotEqual(original[cases[0]["id"]], slope.response(cases[0], 1, "pilot"))

    def test_block_gls_matches_exact_dense_identity_even_at_rank_zero_and_one(self):
        x = list(map(Q, [-1, 0, 1, -2, -1, 0, 1, 2]))
        y = list(map(Q, [1, 4, 2, -1, 0, 4, 3, 8]))
        sizes = [3, 5]
        for sigma in (
            [[Q(0), Q(0)], [Q(0), Q(0)]],
            [[Q(1, 2), Q(1, 2)], [Q(1, 2), Q(1, 2)]],
            [[Q(1, 2), Q(-1, 2)], [Q(-1, 2), Q(1, 2)]],
            [[Q(1, 2), Q(1, 5)], [Q(1, 5), Q(1, 4)]],
        ):
            beta, covariance = slope.block_gls(x, y, sizes, sigma)
            design = [[Q(1), v] for v in x]
            group = [0] * 3 + [1] * 5
            variance = [
                [
                    Q(i == j)
                    + (
                        sum(
                            design[i][a] * sigma[a][b] * design[j][b]
                            for a in range(2)
                            for b in range(2)
                        )
                        if group[i] == group[j]
                        else 0
                    )
                    for j in range(8)
                ]
                for i in range(8)
            ]
            expected_beta, expected_cov, _ = dense.gls_oracle(design, y, variance, True)
            self.assertEqual(beta, expected_beta)
            self.assertEqual(covariance, expected_cov)

    def test_balanced_closed_form_and_random_intercept_reduction(self):
        for case in slope.registry():
            y = slope.response(case, 0, "smoke")
            beta, cov = slope.block_gls(case["x"], y, case["sizes"], case["sigma"])
            self.assertTrue(all(math.isfinite(v) for v in beta))
            if len(set(case["sizes"])) == 1:
                m = case["sizes"][0]
                self.assertAlmostEqual(cov[1][1], (case["v1"] + 1 / m) / case["groups"], places=13)
            if case["v1"] == 0:
                # Centered x is orthogonal to the random-intercept span.
                self.assertAlmostEqual(
                    beta[1],
                    sum(a * b for a, b in zip(case["x"], y)) / sum(v * v for v in case["x"]),
                    places=12,
                )

    def test_covariance_mapping_and_invalid_unknown_boundary_states(self):
        self.assertEqual(slope.candidate_covariance([2, -1, 3], 0.5), [[2, -1], [-1, 5]])
        for matrix, state, rank in [
            ([[0, 0], [0, 0]], "singular", 0),
            ([[1, 1], [1, 1]], "singular", 1),
            ([[1, -1], [-1, 1]], "singular", 1),
            ([[1, 0], [0, 1]], "full_rank", 2),
            ([[1, 0], [0, -1e-13]], "singular", 1),
            ([[1, 0], [0, -1e-5]], "invalid", None),
            ([[1, 0.1], [0.2, 1]], "invalid", None),
            ([[None, 0], [0, 1]], "invalid", None),
        ]:
            result = slope.covariance_diagnostic(matrix, 1)
            self.assertEqual((result["state"], result["rank"]), (state, rank))
        self.assertIsNone(slope.covariance_diagnostic([[1, 0], [0, 0]], 1)["correlation"])
        self.assertEqual(slope.covariance_diagnostic([[1, 0], [0, 1]], 0)["state"], "invalid")

    def test_exact_binomial_endpoints_symmetry_and_failure_envelopes(self):
        self.assertAlmostEqual(slope.cp(0, 20)[1], 1 - 0.025 ** (1 / 20), places=14)
        self.assertEqual(slope.cp(0, 0), [0, 1])
        for n in (1, 2, 20):
            for k in range(n + 1):
                self.assertAlmostEqual(slope.cp(k, n)[0], 1 - slope.cp(n - k, n)[1], places=13)
        result = slope.binary(1, 4, 20)
        self.assertEqual(result["failure_envelope95"], [slope.cp(1, 20)[0], slope.cp(17, 20)[1]])
        self.assertEqual(slope.binary(0, 0, 20)["failure_envelope95"], [0, 1])

    def test_ordered_records_reject_missing_duplicate_extra_and_boolean_indices(self):
        rows = [{"case": "a", "replicate": i} for i in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            for content in (
                rows[:1],
                rows + [rows[1]],
                [rows[0], rows[0]],
                [{"case": "a", "replicate": False}, rows[1]],
                [{"case": "b", "replicate": 0}, rows[1]],
            ):
                path.write_text("".join(json.dumps(r) + "\n" for r in content))
                with self.assertRaises(ValueError):
                    slope.ordered(path, "a", 2, False)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(slope.ordered(path, "a", 2, False), rows)


if __name__ == "__main__":
    unittest.main()
