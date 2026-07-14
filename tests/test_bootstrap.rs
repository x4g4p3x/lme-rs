use lme_rs::{boot_lmer, lmer, BootLmerMethod};
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
