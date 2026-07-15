use lme_rs::family::Family;
use lme_rs::{boot_glmer, boot_lmer, glmer, lmer, BootLmerMethod};
use polars::prelude::*;
use std::fs::File;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

#[test]
fn test_boot_lmer_parametric_sleepstudy() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();

    let boot = boot_lmer(
        formula,
        &df,
        &fit,
        25,
        BootLmerMethod::Parametric,
        true,
        Some(42),
        Some(1),
    )
    .unwrap();

    assert_eq!(boot.nsim, 25);
    assert_eq!(boot.replicates.len(), 25);
    assert_eq!(boot.t0.len(), fit.coefficients.len());
    assert!(boot.prop_converged > 0.5);
    for rep in &boot.replicates {
        assert_eq!(rep.coefficients.len(), fit.coefficients.len());
    }

    let ci = boot.confint_percentile(0.95).unwrap();
    assert_eq!(ci.estimate.len(), fit.coefficients.len());
    for i in 0..ci.estimate.len() {
        assert!(ci.lower[i] <= ci.estimate[i]);
        assert!(ci.upper[i] >= ci.estimate[i]);
    }
}

#[test]
fn test_boot_lmer_residual_reproducible() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();

    let a = boot_lmer(
        formula,
        &df,
        &fit,
        15,
        BootLmerMethod::Residual,
        true,
        Some(7),
        Some(1),
    )
    .unwrap();
    let b = boot_lmer(
        formula,
        &df,
        &fit,
        15,
        BootLmerMethod::Residual,
        true,
        Some(7),
        Some(1),
    )
    .unwrap();

    for (ra, rb) in a.replicates.iter().zip(b.replicates.iter()) {
        assert_eq!(ra.coefficients, rb.coefficients);
    }
}

#[test]
fn test_boot_lmer_parallel_matches_sequential() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();

    let sequential = boot_lmer(
        formula,
        &df,
        &fit,
        12,
        BootLmerMethod::Parametric,
        true,
        Some(99),
        Some(1),
    )
    .unwrap();
    let parallel = boot_lmer(
        formula,
        &df,
        &fit,
        12,
        BootLmerMethod::Parametric,
        true,
        Some(99),
        None,
    )
    .unwrap();

    for (s, p) in sequential.replicates.iter().zip(parallel.replicates.iter()) {
        assert_eq!(s.coefficients, p.coefficients);
        assert_eq!(s.converged, p.converged);
    }
}

#[test]
fn test_fit_boot_method() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();

    let boot = fit
        .boot(
            formula,
            &df,
            10,
            BootLmerMethod::Parametric,
            true,
            Some(1),
            Some(1),
        )
        .unwrap();
    assert_eq!(boot.nsim, 10);
}

fn synthetic_binary_glmm() -> (DataFrame, &'static str) {
    // Small 0/1 binomial with one grouping factor.
    let n_groups = 8usize;
    let n_per = 6usize;
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for gi in 0..n_groups {
        for j in 0..n_per {
            let xv = (j as f64) / (n_per as f64);
            let lp = -0.5 + 0.8 * xv + 0.3 * ((gi % 3) as f64 - 1.0);
            let p = 1.0 / (1.0 + (-lp).exp());
            y.push(if p > 0.5 { 1.0 } else { 0.0 });
            x.push(xv);
            g.push(format!("g{gi}"));
        }
    }
    let df = DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("g".into(), g),
    ])
    .unwrap();
    (df, "y ~ x + (1 | g)")
}

#[test]
fn test_boot_glmer_parametric_binary() {
    let (df, formula) = synthetic_binary_glmm();
    let fit = glmer(formula, &df, Family::Binomial, 1).unwrap();
    let boot = boot_glmer(
        formula,
        &df,
        &fit,
        12,
        BootLmerMethod::Parametric,
        Some(11),
        Some(1),
    )
    .unwrap();
    assert_eq!(boot.nsim, 12);
    assert_eq!(boot.replicates.len(), 12);
    assert!(boot.prop_converged > 0.0);
    let ci = boot.confint_percentile(0.90).unwrap();
    assert_eq!(ci.estimate.len(), fit.coefficients.len());
}

#[test]
fn test_boot_glmer_rejects_residual() {
    let (df, formula) = synthetic_binary_glmm();
    let fit = glmer(formula, &df, Family::Binomial, 1).unwrap();
    let err = boot_glmer(
        formula,
        &df,
        &fit,
        5,
        BootLmerMethod::Residual,
        Some(1),
        Some(1),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("residual"), "{msg}");
}

#[test]
fn test_boot_glmer_rejects_lmm() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();
    let err = boot_glmer(
        formula,
        &df,
        &fit,
        5,
        BootLmerMethod::Parametric,
        Some(1),
        Some(1),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("GLMM") || msg.contains("family"), "{msg}");
}
