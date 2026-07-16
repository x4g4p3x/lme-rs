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
fn test_confint_profile_parms_subset() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, false).unwrap();
    let all = fit.confint_profile(0.95, &df).unwrap();
    let days_only = fit.confint_profile_parms(0.95, &df, &[1]).unwrap();
    assert_eq!(days_only.names.len(), 1);
    assert_eq!(days_only.names[0], all.names[1]);
    assert!((days_only.lower[0] - all.lower[1]).abs() < 1e-6);
    assert!((days_only.upper[0] - all.upper[1]).abs() < 1e-6);
}

#[test]
fn test_confint_profile_sleepstudy_matches_r_fixture() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, false).unwrap();
    let profile = fit.confint_profile(0.95, &df).unwrap();
    let file = File::open("tests/data/sleepstudy_confint_profile.json")
        .expect("sleepstudy_confint_profile.json");
    let raw: serde_json::Value = serde_json::from_reader(file).unwrap();
    let outs = &raw["outputs"];
    let r_lo = outs["lower"].as_array().unwrap();
    let r_hi = outs["upper"].as_array().unwrap();
    for i in 0..2 {
        let lo = r_lo[i].as_f64().unwrap();
        let hi = r_hi[i].as_f64().unwrap();
        assert!(
            (profile.lower[i] - lo).abs() < 6.0,
            "lower[{i}]: rust={} r={}",
            profile.lower[i],
            lo
        );
        assert!(
            (profile.upper[i] - hi).abs() < 6.0,
            "upper[{i}]: rust={} r={}",
            profile.upper[i],
            hi
        );
    }
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
