"""Model-based analysis contracts and independent R reference outputs."""

import json
from pathlib import Path

import lme_python as lme
import numpy as np
import polars as pl
import pytest


@pytest.fixture
def reference():
    return json.loads(
        (Path(__file__).parents[2] / "tests/data/modeling_reference.json").read_text()
    )


def specs():
    return {
        "a": lme.FactorSpec(["C", "A", "B"], "sum"),
        "b": lme.FactorSpec(["late", "early"], "sum"),
    }


def test_classical_inference_and_adjusted_comparisons(reference):
    r = reference["ols"]
    data = pl.DataFrame(r["data"])
    fit = lme.lm("y ~ a*b + x", data, factors=specs())
    np.testing.assert_allclose(fit.coefficients, r["coefficients"], atol=1e-10)
    ci = fit.confint()
    np.testing.assert_allclose(np.column_stack([ci.lower, ci.upper]), r["ci"], atol=1e-9)
    assert fit.fixed_term_assign.count("a") == 2
    assert fit.fixed_term_assign.count("a:b") == 2
    assert fit.categorical_levels["a"] == ["C", "A", "B"]
    for kind, expected in zip(["I", "II", "III"], r["anova"], strict=True):
        table = fit.anova(ddf_method="residual", anova_type=kind)
        for row in expected["rows"]:
            i = table.terms.index(row["term"])
            assert table.f_value[i] == pytest.approx(row["f"], rel=1e-9)
            assert table.sum_sq[i] == pytest.approx(row["ss"], rel=1e-9)
        assert table.residual_df == r["residual_df"]
    means = fit.emmeans_grid(["a"], data, by=["b"], at={"x": 0.75}, ddf_method="residual")
    for i, cell in enumerate(means.cells):
        row = next(row for row in r["means"] if row["a"] == cell["a"] and row["b"] == cell["b"])
        assert means.estimate[i] == pytest.approx(row["emmean"])
    pairs = fit.emmeans_grid_pairs(
        ["a"], data, by=["b"], at={"x": 0.75}, adjust="holm", ddf_method="residual"
    )
    np.testing.assert_allclose(pairs.p_adjust, [row["p.value"] for row in r["pairs"]], atol=1e-10)
    assert pairs.groups == [{"b": row["b"]} for row in r["pairs"]]


def test_factor_prediction_and_guards(reference):
    data = pl.DataFrame(reference["ols"]["data"])
    fit = lme.lm("y ~ a*b + x", data.to_pandas(), factors=specs())
    grid = pl.DataFrame({"a": ["B", "C", "A"], "b": ["late"] * 3, "x": [0.75] * 3})
    design = np.array(fit.design_matrix(grid))
    np.testing.assert_array_equal(design[:, 1:3], [[-1, -1], [1, 0], [0, 1]])
    np.testing.assert_allclose(design @ fit.coefficients, fit.predict(grid))
    with pytest.raises(ValueError):
        lme.FactorSpec(["C", "C"], "sum")
    with pytest.raises(ValueError):
        lme.lm("y ~ a", data, factors={"missing": lme.FactorSpec(["C"])})
    with pytest.raises(ValueError):
        fit.predict(grid.with_columns(pl.lit("new").alias("a")))
    with pytest.raises(ValueError):
        fit.emmeans_grid(["a"], data, by=["a"])
    with pytest.raises(ValueError):
        fit.emmeans_grid(["a"], data, at={"x": float("nan")})
    saturated = lme.lm([1.0, 2.0], [[1.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        saturated.test_contrast([[0.0, 1.0]], ddf_method="residual")


def test_prepared_null_bootstrap_incomplete_subjects(reference):
    data = pl.DataFrame(reference["lmm"]["data"])
    factors = {"a": specs()["a"]}
    full = lme.prepare_lmer("y ~ a*x + (1|id)", data, factors=factors)
    null = lme.prepare_lmer("y ~ a + x + (1|id)", data, factors=factors)
    fit = lme.lmer("y ~ a*x + (1|id)", data, factors=factors)
    fit.with_satterthwaite(data)
    assert fit.anova().num_df == [2.0, 1.0, 2.0]
    a = lme.bootstrap_lrt(full, null, 9, seed=127, n_jobs=1)
    b = lme.bootstrap_lrt(full, null, 9, seed=127, n_jobs=2)
    assert a.valid == a.requested == 9
    assert a.statistics == b.statistics
    assert a.errors == [None] * 9
    assert a.p_value == (1 + a.exceedances) / (1 + a.valid)
    expected = reference["lmm"]["null_deviance"] - reference["lmm"]["full_deviance"]
    assert a.observed == pytest.approx(expected, abs=1e-5)
    with pytest.raises(ValueError):
        lme.bootstrap_lrt(null, full, 9, seed=127)


def test_classical_contrast_rank_is_invariant_to_units(reference):
    data = pl.DataFrame(reference["ols"]["data"])
    fit = lme.lm("y ~ a*b + x", data, factors=specs())
    matrix = np.eye(7)[1:3]
    expected = fit.test_contrast(matrix.tolist(), ddf_method="residual")
    matrix[1] *= 1e-100
    actual = fit.test_contrast(matrix.tolist(), ddf_method="residual")
    assert actual.num_df == expected.num_df == 2
    assert actual.f_value == pytest.approx(expected.f_value)
