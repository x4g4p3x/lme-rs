"""Independent identities and failure gates for the optional comparison harness."""

import copy
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("threadpoolctl")
pytest.importorskip("pingouin")

SPEC = importlib.util.spec_from_file_location(
    "workflow_benchmark",
    Path(__file__).parents[2] / "scripts/run_model_workflow_comparison.py",
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
bench.dependencies()


@pytest.mark.parametrize("reml", [False, True])
def test_gaussian_reference_agrees_with_balanced_anova_identity(reml):
    # Balanced random-intercept model with exactly known between/within sums of squares.
    group_mean = np.array([-1.5, -0.5, 0.5, 1.5])
    within = np.array([-0.3, 0.0, 0.3])
    data = pd.DataFrame(
        {"id": np.repeat(list("abcd"), 3), "y": 7 + np.repeat(group_mean, 3) + np.tile(within, 4)}
    )
    result = bench.gaussian_reference(data, np.ones((12, 1)), reml)
    sigma_e = np.sum(np.tile(within, 4) ** 2) / (12 - 4)
    variance_of_means = np.sum(group_mean**2) / (3 if reml else 4)
    sigma_u = variance_of_means - sigma_e / 3
    np.testing.assert_allclose(result["variance"], [sigma_u, sigma_e], atol=1e-7)
    np.testing.assert_allclose(result["beta"], [7], atol=1e-12)
    covariance = sigma_e * np.eye(12) + sigma_u * np.kron(np.eye(4), np.ones((3, 3)))
    residual = data.y.to_numpy() - 7
    objective = 12 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
    objective += residual @ np.linalg.solve(covariance, residual)
    if reml:
        objective += np.log(np.ones(12) @ np.linalg.solve(covariance, np.ones(12))) - np.log(
            2 * np.pi
        )
    assert result["objective"] == pytest.approx(objective, abs=1e-9)


def test_reference_evaluates_exact_zero_variance_boundary():
    data = pd.DataFrame({"id": np.repeat(list("abcd"), 3), "y": np.tile([-1.0, 0.0, 1.0], 4)})
    result = bench.gaussian_reference(data, np.ones((12, 1)), False)
    assert result["variance"][0] == 0
    assert result["variance"][1] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(result["blup"], np.zeros(4))


def test_design_alignment_rejects_changed_coding():
    x = np.column_stack((np.ones(4), [-1, 1, -1, 1]))
    assert bench.column_permutation(x, x[:, ::-1]) == [1, 0]
    with pytest.raises(ValueError, match="coding differs"):
        bench.column_permutation(x, np.column_stack((np.ones(4), [0, 1, 0, 1])))


def test_numeric_gate_rejects_sign_scale_nonfinite_and_changed_p_values():
    expected = {"beta": [1.0, -0.4], "variance": [1.5, 0.25], "p": [0.049]}
    for key, invalid in (
        ("beta", [-1.0, 0.4]),
        ("variance", [6.0, 1.0]),
        ("p", [0.051]),
        ("p", [float("nan")]),
    ):
        actual = copy.deepcopy(expected)
        actual[key] = invalid
        assert not bench.difference(actual, expected, [key])[key]["pass"]
    # A tiny numerical error still fails if it crosses the reporting threshold.
    assert not bench.difference({"p": [0.050001]}, {"p": [0.049999]}, ["p"])["p"]["pass"]


def test_incomplete_data_reference_is_invariant_to_row_order_and_response_units():
    data, columns = bench.dataset("incomplete_decomposed")
    original = bench.gaussian_reference(data, bench.canonical_design(data, columns), False)
    changed = data.sample(frac=1, random_state=9).reset_index(drop=True)
    changed["y"] *= -3.0
    result = bench.gaussian_reference(changed, bench.canonical_design(changed, columns), False)
    np.testing.assert_allclose(result["beta"], original["beta"] * -3, atol=1e-7)
    np.testing.assert_allclose(result["variance"], np.asarray(original["variance"]) * 9, rtol=1e-6)
    assert result["objective"] == pytest.approx(
        original["objective"] + 2 * len(data) * np.log(3), abs=1e-8
    )


def test_compatible_small_model_preserves_native_contrasts():
    bench.LME_TOLERANCE = 1e-10
    data, columns = bench.dataset("balanced_none")
    outputs = []
    for backend in ("statsmodels", "lme"):
        fit = bench.fit_model(backend, data, columns, True)
        value = bench.snapshot(backend, fit, data, columns)
        inference = bench.inference(value, data, columns)
        bench.compare_native_inference(backend, fit, data, columns, inference)
        if backend == "statsmodels":
            native = fit.wald_test_terms(scalar=True).table.iloc[1:]["pvalue"].to_numpy()
            np.testing.assert_allclose(native, inference["p"][: len(native)], atol=1e-12)
        outputs.append(value | inference)
    checks = bench.difference(
        outputs[0], outputs[1], ["objective", "beta", "variance", "se", "ci", "p"]
    )
    assert all(value["pass"] for value in checks.values())


def test_timing_records_keep_failed_fits_and_warnings():
    def broken_fit():
        warnings.warn("optimizer reached boundary", RuntimeWarning, stacklevel=1)
        raise ValueError("nonconverged")

    records = bench.measured(broken_fit, lambda _: None, warmups=1, repeats=2)
    assert len(records) == 3
    assert records[0]["warmup"]
    for row in records:
        assert row["error"] == "ValueError: nonconverged"
        assert row["warnings"] == ["optimizer reached boundary"]
        assert row["checks"] is None
        assert row["seconds"] > 0


def test_comparison_rejects_mixed_extension_builds():
    reference = {
        k: "same"
        for k in (
            "revision",
            "harness_sha256",
            "extension_sha256",
            "versions",
            "threads",
            "lme_tolerance",
        )
    }
    other = dict(reference, extension_sha256="different")
    with pytest.raises(ValueError, match="extension_sha256"):
        bench.summarize([{"metadata": reference}, {"metadata": other}], [])


def test_nested_main_effect_and_interaction_lrt_match_independent_likelihood():
    data, _ = bench.dataset("incomplete_decomposed")
    terms = ["a*b", "a + b", "b", "a"]
    objectives = {"statsmodels": [], "lme": [], "oracle": []}
    for term in terms:
        fixed = term + " + xb + xw"
        # Construct the two factor pieces separately so covariate names stay intact.
        reference_formula = (
            "y ~ " + term.replace("a", "C(a, Sum)").replace("b", "C(b, Sum)") + " + xb + xw"
        )
        x = bench.patsy.dmatrix(reference_formula.split("~", 1)[1], data)
        reference = bench.gaussian_reference(data, x, False)
        sm_fit = bench.smf.mixedlm(reference_formula, data, groups=data.id).fit(reml=False)
        factors = {
            name: bench.lme.FactorSpec(list(data[name].cat.categories), "sum")
            for name in ("a", "b")
            if name in term
        }
        rust_fit = bench.lme.lmer(
            "y ~ " + fixed + " + (1|id)",
            data,
            reml=False,
            factors=factors,
            control=bench.lme.FitControl(tolerance=1e-10),
        )
        assert sm_fit.converged and rust_fit.converged
        objectives["statsmodels"].append(-2 * sm_fit.llf)
        objectives["lme"].append(rust_fit.deviance)
        objectives["oracle"].append(reference["objective"])
    for backend in ("statsmodels", "lme"):
        for full, null in ((0, 1), (1, 2), (1, 3)):
            actual = objectives[backend][null] - objectives[backend][full]
            expected = objectives["oracle"][null] - objectives["oracle"][full]
            assert actual >= 0
            assert actual == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize("case", ["balanced_none", "incomplete_decomposed", "incomplete_baseline"])
def test_marginal_factor_comparisons_preserve_averaging_and_correction_families(case):
    """Averaged factor contrasts and conditional simple effects are different hypotheses."""
    bench.LME_TOLERANCE = 1e-10
    data, columns = bench.dataset(case)
    reference = bench.fit_model("statsmodels", data, columns, True)
    candidate = bench.fit_model("lme", data, columns, True)
    grid = bench.grid_for(data, columns)
    design = np.asarray(
        bench.patsy.build_design_matrices([reference.model.data.design_info], grid)[0]
    )
    for factor in ("a", "b"):
        levels = data[factor].cat.categories
        vectors = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                vectors.append(
                    design[grid[factor] == levels[j]].mean(axis=0)
                    - design[grid[factor] == levels[i]].mean(axis=0)
                )
        test = reference.t_test(np.asarray(vectors))
        adjusted = bench.pg.multicomp(np.ravel(test.pvalue), method="holm")[1]
        pairs = candidate.emmeans_grid_pairs(
            [factor], data, at=dict.fromkeys(columns, 0.0), weights="equal", adjust="holm"
        )
        observed = {
            "prediction": pairs.estimate,
            "se": pairs.std_error,
            "ci": np.column_stack(
                (
                    np.asarray(pairs.estimate) - 1.959963984540054 * np.asarray(pairs.std_error),
                    np.asarray(pairs.estimate) + 1.959963984540054 * np.asarray(pairs.std_error),
                )
            ),
            "p": pairs.p_adjust,
        }
        expected = {
            "prediction": np.ravel(test.effect),
            "se": np.ravel(test.sd),
            "ci": test.conf_int(),
            "p": adjusted,
        }
        checks = bench.difference(observed, expected, list(observed))
        assert all(value["pass"] for value in checks.values()), checks
        reference_sd = np.sqrt(float(np.asarray(reference.cov_re)[0, 0]) + reference.scale)
        candidate_sd = np.sqrt(candidate.var_corr[0][3] + candidate.sigma2)
        np.testing.assert_allclose(
            np.asarray(pairs.estimate) / candidate_sd,
            np.ravel(test.effect) / reference_sd,
            atol=2e-4,
            rtol=2e-4,
        )
