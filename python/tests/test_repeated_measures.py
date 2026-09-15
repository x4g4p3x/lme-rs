"""Numeric contracts for the portable repeated-measures integration recipe."""

import importlib.util
from itertools import product
from pathlib import Path
from random import Random

import lme_python as lme
import polars as pl
import pytest

SPEC = importlib.util.spec_from_file_location(
    "repeated_measures_example",
    Path(__file__).resolve().parents[1] / "examples" / "repeated_measures.py",
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def test_balanced_cell_means_and_declared_order():
    data, between, within, interactions = recipe.encode(recipe.sample_data())
    formula = "dv ~ " + " + ".join(between + within + interactions) + " + (1 | subject_id)"
    fit = lme.lmer(formula, data, control=lme.FitControl(require_convergence=True))
    rows = [{"between": b, "within": w} for b, w in product(recipe.BETWEEN, recipe.WITHIN)]
    grid, _, _, _ = recipe.encode(pl.DataFrame(rows))
    # For a balanced complete factorial with a random intercept and full cell
    # fixed effects, fitted population cell means equal observed cell means.
    observed = [
        data.filter((pl.col("between") == row["between"]) & (pl.col("within") == row["within"]))[
            "dv"
        ].mean()
        for row in rows
    ]
    assert fit.predict(grid) == pytest.approx(observed, abs=1e-7)
    assert grid["within"].to_list()[:3] == ["Day 1", "Day 2", "Day 10"]
    assert grid["w0"].to_list()[:3] == [1, 0, -1]
    assert grid["w1"].to_list()[:3] == [0, 1, -1]


def test_null_bootstrap_reproducible_and_resamples_animals():
    data, between, within, _ = recipe.encode(recipe.sample_data())
    full = "dv ~ " + " + ".join(between + within) + " + (1 | subject_id)"
    null = "dv ~ " + " + ".join(within) + " + (1 | subject_id)"
    first = recipe.bootstrap_lrt(data, full, null, replicates=5, seed=9)
    assert first == recipe.bootstrap_lrt(data, full, null, replicates=5, seed=9)
    assert 0 < first["p"] <= 1
    assert first["valid"] == 5
    fit = lme.lmer(null, data, reml=False)
    variance = next(row[3] for row in fit.var_corr if row[0] == "subject_id" and row[1] == row[2])
    mean = fit.predict(data)
    rng = Random(19)
    draws = [recipe.simulate_null(fit, data, rng) for _ in range(2000)]
    # Same-animal covariance must include the random-intercept variance. A
    # conditional response bootstrap with fixed BLUPs would miss this variance.
    first_column = [row[0] - mean[0] for row in draws]
    second_column = [row[1] - mean[1] for row in draws]
    avg_a = sum(first_column) / len(draws)
    avg_b = sum(second_column) / len(draws)
    covariance = sum((a - avg_a) * (b - avg_b) for a, b in zip(first_column, second_column)) / len(
        draws
    )
    assert variance > 1
    assert covariance == pytest.approx(variance, rel=0.15)


def test_sum_coding_rejects_unknown_levels():
    with pytest.raises(ValueError, match="unknown level"):
        recipe.sum_code(pl.DataFrame({"phase": ["unexpected"]}), "phase", ["a", "b"], "w")
