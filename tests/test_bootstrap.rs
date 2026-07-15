use lme_rs::family::Family;
use lme_rs::{boot_glmer, boot_lmer, glmer, lmer, BootLmerMethod};
use ndarray::Array1;
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

fn synthetic_proportion_trials_glmm() -> (DataFrame, Array1<f64>, &'static str) {
    // Proportion response + integer trial weights (cbpp-style).
    let n_groups = 6usize;
    let n_per = 4usize;
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    let mut w = Vec::new();
    for gi in 0..n_groups {
        for j in 0..n_per {
            let xv = (j as f64) / (n_per as f64);
            let n_trials = 5 + ((gi + j) % 4) as u64;
            let k = ((n_trials as f64) * (0.2 + 0.5 * xv))
                .round()
                .clamp(0.0, n_trials as f64);
            y.push(k / (n_trials as f64));
            x.push(xv);
            g.push(format!("g{gi}"));
            w.push(n_trials as f64);
        }
    }
    let df = DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("g".into(), g),
    ])
    .unwrap();
    (df, Array1::from_vec(w), "y ~ x + (1 | g)")
}

#[test]
fn test_simulate_binomial_proportion_trials() {
    use lme_rs::glmer_weighted;
    let (df, weights, formula) = synthetic_proportion_trials_glmm();
    let fit = glmer_weighted(formula, &df, Family::Binomial, 1, Some(weights)).unwrap();
    assert!(fit.weights.is_some());
    let sim = fit.simulate_with(8, Some(1), Some(42)).unwrap();
    for s in &sim.simulations {
        assert_eq!(s.len(), fit.num_obs);
        for &yi in s.iter() {
            assert!((0.0..=1.0).contains(&yi), "proportion out of range: {yi}");
        }
        // With n_i > 1, at least some draws should be non-binary proportions.
    }
    let any_fraction = sim.simulations.iter().any(|s| {
        s.iter()
            .any(|&yi| (yi - 0.0).abs() > 1e-12 && (yi - 1.0).abs() > 1e-12)
    });
    assert!(
        any_fraction,
        "expected some non-0/1 proportions under multi-trial binomial"
    );
}

#[test]
fn test_boot_glmer_parametric_proportion_trials() {
    use lme_rs::glmer_weighted;
    let (df, weights, formula) = synthetic_proportion_trials_glmm();
    let fit = glmer_weighted(formula, &df, Family::Binomial, 1, Some(weights)).unwrap();
    let boot = boot_glmer(
        formula,
        &df,
        &fit,
        10,
        BootLmerMethod::Parametric,
        Some(13),
        Some(1),
    )
    .unwrap();
    assert_eq!(boot.nsim, 10);
    assert_eq!(boot.replicates.len(), 10);
    assert!(boot.prop_converged > 0.0);
}

#[test]
fn test_binomial_trials_reject_non_integer_counts() {
    use lme_rs::glmer_weighted;
    let df = DataFrame::new(vec![
        Column::new("y".into(), vec![0.3_f64, 0.5]),
        Column::new("x".into(), vec![0.0_f64, 1.0]),
        Column::new("g".into(), vec!["a".to_string(), "b".to_string()]),
    ])
    .unwrap();
    // 0.3 * 10 = 3 ok; 0.5 * 3 = 1.5 not near-integer
    let err = glmer_weighted(
        "y ~ x + (1 | g)",
        &df,
        Family::Binomial,
        1,
        Some(Array1::from_vec(vec![10.0, 3.0])),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("near-integer") || msg.contains("trials"),
        "{msg}"
    );
}
