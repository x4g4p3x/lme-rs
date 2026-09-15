"""Prediction grids and contrasts must use exactly the fitted column encoding."""

import lme_python as lme
import polars as pl
import pytest


def test_design_matrix_retains_training_encoding_and_excludes_offset():
    df = pl.read_csv("../tests/data/sleepstudy.csv").with_columns(
        pl.when(pl.col("Days") < 5).then(pl.lit("Early")).otherwise(pl.lit("Late")).alias("phase"),
        pl.lit(7.0).alias("off"),
    )
    fit = lme.lmer("Reaction ~ Days * phase + offset(off) + (1 | Subject)", df)
    # No response/group columns; only one of the training factor levels.
    grid = pl.DataFrame({"Days": [5.0, 7.0], "phase": ["Late", "Late"], "off": [2.0, 3.0]})
    matrix = fit.design_matrix(grid)
    assert all(len(row) == len(fit.fixed_names) for row in matrix)
    predicted = [sum(x * b for x, b in zip(row, fit.coefficients)) for row in matrix]
    assert [v + off for v, off in zip(predicted, grid["off"])] == pytest.approx(fit.predict(grid))
    with pytest.raises(ValueError):
        fit.design_matrix(grid.with_columns(pl.lit("Unseen").alias("phase")))


def test_design_matrix_preserves_explicit_sum_coding():
    df = pl.read_csv("../tests/data/sleepstudy.csv").with_columns(
        pl.when(pl.col("Days") < 5).then(1.0).otherwise(-1.0).alias("phase_sum")
    )
    fit = lme.lmer("Reaction ~ phase_sum + (1 | Subject)", df)
    grid = pl.DataFrame({"phase_sum": [1.0, -1.0]})
    matrix = fit.design_matrix(grid)
    assert matrix == [[1.0, 1.0], [1.0, -1.0]]
    contrast = [b - a for a, b in zip(*matrix)]
    difference = sum(w * b for w, b in zip(contrast, fit.coefficients))
    means = fit.predict(grid)
    assert difference == pytest.approx(means[1] - means[0])
    assert difference == pytest.approx(-2 * fit.coefficients[1])
