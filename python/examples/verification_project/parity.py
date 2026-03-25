"""
Numerical and API checks for `lme_python` against the same fixtures and tolerances
used in the Rust integration tests (`tests/test_numerical_parity.rs`, `tests/test_glmm.rs`).

This module is imported by `run.py` and `test_parity.py`; it raises AssertionError on failure.
"""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl

import lme_python

from paths import repo_root, tests_data


def _close(name: str, got: float, expected: float, tol: float) -> None:
    d = abs(got - expected)
    if d > tol:
        raise AssertionError(f"{name}: got {got}, expected ~{expected} (|Δ|={d} > tol {tol})")


def verify_sleepstudy_reml() -> None:
    """R / lme4 reference scalars for REML sleepstudy (see `tests/test_numerical_parity.rs`)."""
    df = pl.read_csv(tests_data("sleepstudy.csv"))
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    assert fit.converged
    assert fit.num_obs == 180
    assert len(fit.coefficients) == 2
    _close("beta_0", fit.coefficients[0], 251.4051, 0.05)
    _close("beta_Days", fit.coefficients[1], 10.4673, 0.05)
    assert fit.sigma2 is not None
    _close("sigma2", fit.sigma2, 654.94, 1.0)
    dev = fit.deviance
    assert dev is not None
    if abs(dev - 1743.6283) > 25.0:
        raise AssertionError(f"REML deviance: expected ~1743.6, got {dev}")
    s = fit.summary()
    if "1743" not in s:
        raise AssertionError("summary() should contain REML criterion ~1743")
    preds = fit.predict(df)
    assert len(preds) == 180
    _close("pred_pop_row0", preds[0], 251.405105, 1e-3)


def verify_sleepstudy_nested_lrt() -> None:
    """Likelihood ratio tests on ML fits (nested models, same pattern as `model_comparison.py`)."""
    df = pl.read_csv(tests_data("sleepstudy.csv"))
    m0 = lme_python.lmer("Reaction ~ 1 + (1 | Subject)", data=df, reml=False)
    m1 = lme_python.lmer("Reaction ~ Days + (1 | Subject)", data=df, reml=False)
    m2 = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=False)
    r01 = lme_python.anova(m0, m1)
    chi01 = r01[4]
    p01 = r01[6]
    if chi01 <= 0:
        raise AssertionError(f"LRT M0 vs M1: expected positive chi^2, got {chi01}")
    # p can be 0.0 when χ² is huge (numerical underflow); still a valid LRT.
    if p01 < 0.0 or p01 > 1.0:
        raise AssertionError(f"LRT M0 vs M1: p-value in [0,1], got {p01}")
    r12 = lme_python.anova(m1, m2)
    if r12[4] < 0:
        raise AssertionError("LRT M1 vs M2: chi^2 should be non-negative")
    # AIC: more complex model should not be worse than intercept-only by huge margin
    if m2.aic >= m0.aic + 500:
        raise AssertionError("Full model AIC unexpectedly worse than null")


def verify_cbpp_binomial_json() -> None:
    """Match `tests/data/glmm_binomial.json` R-exported beta and theta (see `tests/test_glmm.rs`)."""
    path = repo_root() / "tests" / "data" / "glmm_binomial.json"
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    beta_r = data["outputs"]["beta"]
    theta_r = data["outputs"]["theta"]
    df = pl.read_csv(tests_data("cbpp_binary.csv"))
    fit = lme_python.glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        data=df,
        family_name="binomial",
        n_agq=1,
    )
    assert fit.converged
    assert len(fit.coefficients) == len(beta_r)
    for i, (got, want) in enumerate(zip(fit.coefficients, beta_r)):
        _close(f"cbpp_beta[{i}]", got, want, 0.05)
    th = fit.theta
    assert th is not None and len(th) == len(theta_r)
    for i, (got, want) in enumerate(zip(th, theta_r)):
        _close(f"cbpp_theta[{i}]", got, want, 0.05)
    dv = fit.deviance
    assert dv is not None and not math.isnan(dv)


def verify_grouseticks_poisson() -> None:
    """Poisson GLMM: finite coefficients, positive mean predictions on response scale."""
    df = pl.read_csv(tests_data("grouseticks.csv"))
    fit = lme_python.glmer(
        "TICKS ~ YEAR + HEIGHT + (1 | BROOD)", data=df, family_name="poisson", n_agq=1
    )
    assert fit.converged
    assert len(fit.coefficients) == 3
    assert all(math.isfinite(c) for c in fit.coefficients)
    pr = fit.predict_response(df)
    assert len(pr) == df.height
    assert min(pr) >= 0.0
    plink = fit.predict(df)
    assert abs(math.exp(plink[0]) - pr[0]) < 1e-5


def verify_pastes_nested_lmm() -> None:
    """Nested batch/cask model: intercept matches lme4 ballpark (~60.05)."""
    df = pl.read_csv(tests_data("pastes.csv"))
    fit = lme_python.lmer("strength ~ 1 + (1 | batch/cask)", data=df, reml=True)
    assert fit.converged
    _close("pastes_intercept", fit.coefficients[0], 60.0533, 0.05)


def verify_penicillin_crossed() -> None:
    """Crossed RE: two theta components, positive variances on summary."""
    df = pl.read_csv(tests_data("penicillin.csv"))
    fit = lme_python.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data=df, reml=True)
    assert fit.converged
    th = fit.theta
    assert th is not None and len(th) >= 2
    assert fit.sigma2 is not None and fit.sigma2 > 0


def verify_type3_anova_pastes() -> None:
    """Type III ANOVA with Satterthwaite (categorical fixed effect)."""
    df = pl.read_csv(tests_data("pastes.csv"))
    fit = lme_python.lmer("strength ~ cask + (1 | batch)", data=df, reml=True)
    fit.with_satterthwaite(df)
    terms, num_df, den_df, f_val, p_val, method = fit.anova("satterthwaite")
    assert any("cask" in str(t).lower() for t in terms)
    assert len(num_df) >= 1
    assert all(x > 0 for x in num_df)


def verify_weighted_sleepstudy() -> None:
    """Prior weights path: σ² and β differ from unweighted REML (internal consistency)."""
    df = pl.read_csv(tests_data("sleepstudy.csv"))
    n = df.height
    w = [0.5 + (i % 5) * 0.1 for i in range(n)]
    fw = lme_python.lmer_weighted(
        "Reaction ~ Days + (Days | Subject)",
        data=df,
        reml=True,
        weights=w,
    )
    uw = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    assert fw.converged and uw.converged
    assert fw.sigma2 is not None and uw.sigma2 is not None
    if abs(fw.sigma2 - uw.sigma2) < 1e-6:
        raise AssertionError("weighted and unweighted fits should not have identical sigma2")


def verify_confint_and_simulate() -> None:
    """Wald confint shape; simulate returns correct dimensions."""
    df = pl.read_csv(tests_data("sleepstudy.csv"))
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    ci = fit.confint(0.95)
    assert len(ci) == 2
    assert ci[0][0] < fit.coefficients[0] < ci[0][1]
    sim = fit.simulate(3)
    assert len(sim) == 3
    assert len(sim[0]) == 180


def run_all_checks() -> list[tuple[str, Exception | None]]:
    """Run every check; return list of (name, error or None)."""
    checks = [
        ("sleepstudy_reml", verify_sleepstudy_reml),
        ("sleepstudy_nested_lrt", verify_sleepstudy_nested_lrt),
        ("cbpp_binomial_json", verify_cbpp_binomial_json),
        ("grouseticks_poisson", verify_grouseticks_poisson),
        ("pastes_nested_lmm", verify_pastes_nested_lmm),
        ("penicillin_crossed", verify_penicillin_crossed),
        ("type3_anova_pastes", verify_type3_anova_pastes),
        ("weighted_sleepstudy", verify_weighted_sleepstudy),
        ("confint_simulate", verify_confint_and_simulate),
    ]
    results: list[tuple[str, Exception | None]] = []
    for name, fn in checks:
        try:
            fn()
            results.append((name, None))
        except Exception as e:  # noqa: BLE001 — surface any failure to the runner
            results.append((name, e))
    return results
