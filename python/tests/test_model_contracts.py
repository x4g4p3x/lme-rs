"""Cross-feature regression contracts for Python model workflows."""

from pathlib import Path

import lme_python as lme
import polars as pl
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def data():
    return pl.read_csv(DATA / "sleepstudy.csv").with_columns((pl.col("Days") * 10.0).alias("off"))


def test_prepared_response_retains_offsets_weights_and_controls():
    df = data()
    formula = "Reaction ~ Days + offset(off) + (Days | Subject)"
    weights = [1.0 + i % 3 for i in range(df.height)]
    prepared = lme.prepare_lmer(formula, df, weights=weights)
    response = (df["Reaction"] + 7.0).to_list()
    fit = prepared.fit(y=response)
    cold = lme.lmer_weighted(
        formula, df.with_columns(pl.Series("Reaction", response)), weights=weights
    )
    assert fit.coefficients == pytest.approx(cold.coefficients, abs=1e-7)
    assert fit.predict_conditional(df) == pytest.approx(fit.fitted, abs=1e-7)
    assert lme.fit_prepared(prepared, y=response).coefficients == pytest.approx(fit.coefficients)
    limited = lme.FitControl(max_iterations=1)
    assert not prepared.fit(control=limited).converged
    assert not lme.lmer_weighted(formula, df, weights=weights, control=limited).converged
    with pytest.raises(RuntimeError, match="converge"):
        prepared.fit(control=lme.FitControl(max_iterations=1, require_convergence=True))
    with pytest.raises(ValueError):
        prepared.fit(y=[1.0])


def test_ipc_projects_columns_and_retains_expression_sources(monkeypatch):
    df = data().with_columns(pl.lit("unused").alias("unused"))
    original = pl.DataFrame.write_ipc
    observed = []

    def record(frame, *args, **kwargs):
        observed.append(frame.columns)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_ipc", record)
    lme.lmer("Reaction ~ I(Days * Days) + offset(off) + (1 | Subject)", df)
    assert set(observed[-1]) == {"Reaction", "Days", "off", "Subject"}
    lme.lm(
        "Reaction ~ .",
        df.drop("Subject", "unused").with_columns((pl.col("Days") ** 2).alias("off")),
    )
    assert "off" in observed[-1]


def test_glmm_prepared_replacement_and_limits():
    df = pl.DataFrame(
        {
            "y": [i % 4 for i in range(60)],
            "x": [i % 5 for i in range(60)],
            "g": [str(i // 10) for i in range(60)],
        }
    )
    formula = "y ~ x + (1 | g)"
    prepared = lme.prepare_glmer(formula, df, "poisson", n_agq=3)
    response = [float(i % 3) for i in range(60)]
    fit = prepared.fit(y=response)
    cold = lme.glmer(formula, df.with_columns(pl.Series("y", response)), "poisson", n_agq=3)
    assert fit.coefficients == pytest.approx(cold.coefficients, abs=1e-7)
    assert fit.diagnostics["requested_n_agq"] == 3
    assert lme.fit_prepared_glmer(prepared, y=response).coefficients == pytest.approx(
        fit.coefficients
    )
    with pytest.raises(RuntimeError):
        lme.glmer(
            formula,
            df,
            "poisson",
            control=lme.FitControl(max_inner_iterations=1, require_convergence=True),
        )


def test_nonlinear_iteration_limits_are_honored():
    df = pl.read_csv(DATA / "orange.csv")
    fit = lme.nlmer(
        "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
        df,
        start={"Asym": 200.0, "xmid": 725.0, "scal": 350.0},
        max_inner=1,
        max_outer_iters=1,
    )
    assert not fit.converged
    assert fit.diagnostics["inner_iterations"] == 1
    with pytest.raises(ValueError):
        lme.nlmer("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree", df, max_inner=0)


def test_prepared_native_fit_releases_interpreter():
    import sys
    import threading

    n = 50000
    df = pl.DataFrame(
        {
            "y": [float(i % 13) + (i % 5) * 0.2 for i in range(n)],
            "x": [float(i % 5) for i in range(n)],
            "g": [str(i // 20) for i in range(n)],
        }
    )
    prepared = lme.prepare_lmer("y ~ x + (x | g)", df)
    ready = threading.Event()
    ran = threading.Event()
    worker = threading.Thread(target=lambda: (ready.wait(), ran.set()), daemon=True)
    worker.start()
    previous = sys.getswitchinterval()
    try:
        # Disable normal Python bytecode time slicing for this short critical region.
        # The waiting Python thread can then run only when native fitting detaches.
        sys.setswitchinterval(100.0)
        ready.set()
        prepared.fit()
        assert ran.is_set()
    finally:
        sys.setswitchinterval(previous)
        ready.set()
        worker.join(timeout=5)


def test_zero_sized_simulation_batches_are_rejected():
    fit = lme.lmer("Reaction ~ Days + (1 | Subject)", data())
    with pytest.raises(ValueError, match="batch_size"):
        fit.simulate_batches(10, 0)
