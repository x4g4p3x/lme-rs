//! Fast, deterministic checks for standard regression / mixed-model identities.
//! These complement fixture-backed parity tests (`test_numerical_parity`, `test_glmm`, …):
//! they encode algebraic or probabilistic invariants that should always hold for the
//! implemented algorithms (within floating-point tolerance).

use lme_rs::anova;
use lme_rs::family::Family;
use lme_rs::glmer;
use lme_rs::lm_df;
use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

fn sleepstudy_df() -> DataFrame {
    let mut file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("csv")
}

fn cbpp_df() -> DataFrame {
    let mut file = File::open("tests/data/cbpp_binary.csv").expect("cbpp_binary.csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("csv")
}

fn grouseticks_df() -> DataFrame {
    let mut file = File::open("tests/data/grouseticks.csv").expect("grouseticks.csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("csv")
}

/// OLS with an intercept: residuals sum to zero (exact under full-rank fixed-effects design).
#[test]
fn ols_residuals_sum_to_zero_with_intercept() {
    let df = sleepstudy_df();
    let fit = lm_df("Reaction ~ Days", &df).expect("lm_df");
    let s: f64 = fit.residuals.iter().sum();
    assert!(
        s.abs() < 1e-5,
        "sum(residuals) should be ~0 for OLS with intercept, got {}",
        s
    );
}

/// Conditional decomposition on the original response scale: y = fitted + residual (exact).
#[test]
fn lmm_y_equals_fitted_plus_residual() {
    let df = sleepstudy_df();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).expect("lmer");
    let y = df
        .column("Reaction")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    let y = y.f64().unwrap();
    for (i, yv) in y.into_no_null_iter().enumerate() {
        let recon = fit.fitted[i] + fit.residuals[i];
        assert!(
            (recon - yv).abs() < 1e-5,
            "y[{i}] = {yv} but fitted + residual = {recon}"
        );
    }
}

/// Nested ML comparison: likelihood-ratio statistic is non-negative; p-value is a valid probability.
#[test]
fn lrt_nested_lmm_invariants() {
    let df = sleepstudy_df();
    let fit0 = lmer("Reaction ~ 1 + (1 | Subject)", &df, false).expect("null");
    let fit1 = lmer("Reaction ~ Days + (1 | Subject)", &df, false).expect("alt");
    let res = anova(&fit0, &fit1).expect("anova LRT");
    assert!(res.chi_sq >= 0.0, "chi_sq = {}", res.chi_sq);
    assert!(res.df >= 1, "df = {}", res.df);
    assert!(
        res.p_value >= 0.0 && res.p_value <= 1.0,
        "p_value = {}",
        res.p_value
    );
    // More complex model should not have larger deviance than simpler on same data (ML).
    assert!(
        res.deviance_0 >= res.deviance_1 - 1e-6,
        "deviance null {} < alt {} inconsistent with nesting",
        res.deviance_0,
        res.deviance_1
    );
}

/// REML LMM: residual variance and variance-component parameters on the θ scale are finite;
/// σ² is strictly positive on standard sleepstudy fit.
#[test]
fn lmm_variance_parameters_sane() {
    let df = sleepstudy_df();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).expect("lmer");
    let s2 = fit.sigma2.expect("sigma2");
    assert!(s2 > 0.0 && s2.is_finite(), "sigma2 = {}", s2);
    let th = fit.theta.as_ref().expect("theta");
    for (i, &t) in th.iter().enumerate() {
        assert!(t.is_finite(), "theta[{i}] = {}", t);
    }
}

/// Binomial GLMM: response-scale predictions lie strictly in (0, 1) for a standard fit.
#[test]
fn glmm_binomial_fitted_probabilities_in_open_unit_interval() {
    let df = cbpp_df();
    let fit = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
        1,
    )
    .expect("glmer");
    let mu = fit.predict_response(&df).expect("predict_response");
    for (i, &p) in mu.iter().enumerate() {
        assert!(
            p > 1e-12 && p < 1.0 - 1e-12,
            "mu[{i}] = {} not in (0,1)",
            p
        );
    }
}

/// Poisson GLMM: mean response-scale predictions are strictly positive.
#[test]
fn glmm_poisson_fitted_rates_positive() {
    let df = grouseticks_df();
    // Same structure as `tests/test_glmm.rs` (R reference uses YEAR dummies).
    let fit = glmer(
        "TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)",
        &df,
        Family::Poisson,
        1,
    )
    .expect("glmer");
    let mu = fit.predict_response(&df).expect("predict_response");
    for (i, &m) in mu.iter().enumerate() {
        assert!(m > 0.0 && m.is_finite(), "mu[{i}] = {}", m);
    }
}
