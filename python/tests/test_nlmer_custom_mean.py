"""Custom nonlinear mean via ``nlmer_with_mean``."""

import math

import lme_python
import polars as pl


def _exponential_mean(x: float, params: list[float]) -> tuple[float, list[float]]:
    a, b = params
    mu = a * math.exp(-b * x)
    da = math.exp(-b * x)
    db = -x * a * math.exp(-b * x)
    return mu, [da, db]


def test_nlmer_with_mean_exponential():
    n = 30
    x = [i * 0.2 for i in range(n)]
    y = [3.0 * math.exp(-0.5 * xi) + 0.1 for xi in x]
    g = ["1"] * 15 + ["2"] * 15
    df = pl.DataFrame({"y": y, "x": x, "g": g})

    fit = lme_python.nlmer_with_mean(
        "y ~ x ~ a | g",
        data=df,
        mean_fn=_exponential_mean,
        param_names=["a", "b"],
        start={"a": 2.0, "b": 0.4},
        reml=False,
    )
    assert fit.converged
    assert fit.deviance is not None and math.isfinite(fit.deviance)
    pred = fit.predict(df)
    assert len(pred) == n


def test_nlmer_with_mean_rejects_bad_grad_length():
    df = pl.DataFrame({"y": [1.0], "x": [0.0], "g": ["1"]})

    def bad_mean(x: float, params: list[float]) -> tuple[float, list[float]]:
        return 1.0, [1.0]

    try:
        lme_python.nlmer_with_mean(
            "y ~ x ~ a | g",
            data=df,
            mean_fn=bad_mean,
            param_names=["a", "b"],
        )
    except ValueError as e:
        assert "grad length" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched grad length")
