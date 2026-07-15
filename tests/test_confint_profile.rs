//! Profile-likelihood confidence intervals for fixed effects.

use lme_rs::family::Family;
use lme_rs::{glmer, lmer, ConfintMethod};
use polars::prelude::*;
use std::fs::File;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv");
    CsvReader::new(file).finish().expect("read sleepstudy")
}

fn load_cbpp() -> DataFrame {
    let file = File::open("tests/data/cbpp_binary.csv").expect("cbpp_binary.csv");
    CsvReader::new(file).finish().expect("read cbpp")
}

#[test]
fn test_confint_profile_sleepstudy_contains_estimate() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();
    let wald = fit.confint(0.95).unwrap();
    let profile = fit.confint_profile(0.95, &df).unwrap();
    assert_eq!(profile.names, wald.names);
    for i in 0..wald.lower.len() {
        assert!(
            profile.lower[i] < fit.coefficients[i] && fit.coefficients[i] < profile.upper[i],
            "coef {i} not inside profile CI"
        );
        let pw = profile.upper[i] - profile.lower[i];
        let ww = wald.upper[i] - wald.lower[i];
        assert!(pw.is_finite() && pw > 0.0);
        // Profile (χ² / ML) and Wald (z on REML) need not nest; require similar scale.
        assert!(
            pw > 0.5 * ww && pw < 2.5 * ww,
            "coef {i}: profile width {pw} vs Wald {ww}"
        );
    }
}

#[test]
fn test_confint_with_profile_matches_confint_profile() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let a = fit.confint_profile(0.90, &df).unwrap();
    let b = fit
        .confint_with(0.90, ConfintMethod::Profile, Some(&df))
        .unwrap();
    for i in 0..a.lower.len() {
        assert!((a.lower[i] - b.lower[i]).abs() < 1e-8);
        assert!((a.upper[i] - b.upper[i]).abs() < 1e-8);
    }
}

#[test]
fn test_confint_profile_rejects_bad_level() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    assert!(fit.confint_profile(0.0, &df).is_err());
    assert!(fit.confint_profile(1.0, &df).is_err());
}

#[test]
fn test_confint_profile_glmm_binomial_smoke() {
    let df = load_cbpp();
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let fit = glmer(formula, &df, Family::Binomial, 1).unwrap();
    let profile = fit.confint_profile(0.90, &df).unwrap();
    assert_eq!(profile.lower.len(), fit.coefficients.len());
    for i in 0..profile.lower.len() {
        assert!(profile.lower[i] < profile.upper[i]);
        assert!(
            profile.lower[i] < fit.coefficients[i] && fit.coefficients[i] < profile.upper[i],
            "coef {i} not inside profile CI"
        );
    }
    let wald = fit.confint(0.90).unwrap();
    // Width sanity: profile should be finite and not tiny vs Wald.
    for i in 0..wald.lower.len() {
        let pw = profile.upper[i] - profile.lower[i];
        let ww = wald.upper[i] - wald.lower[i];
        assert!(pw.is_finite() && pw > 0.0);
        assert!(pw > 0.25 * ww, "profile width suspiciously small vs Wald");
    }
}
