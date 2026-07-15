use lme_rs::{cv_grouped, fit_prepared, lmer, prepare_lmer, refit_lmer};
use polars::prelude::*;
use std::fs::File;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

#[test]
fn test_cv_grouped_sleepstudy() {
    let df = load_sleepstudy();
    let result = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        5,
        true,
        Some(42),
        Some(1),
    )
    .unwrap();

    assert_eq!(result.n_splits, 5);
    assert_eq!(result.group_col, "Subject");
    assert!(result.folds.len() == 5);
    assert!(result.all_converged);
    assert!(result.rmse.is_finite() && result.rmse > 0.0);
    assert!(result.mae.is_finite() && result.mae > 0.0);
    assert_eq!(result.oof_predictions.len(), df.height());
    assert_eq!(result.test_fold.len(), df.height());

    for &f in result.test_fold.iter() {
        assert!((0..5).contains(&f));
    }
    for fold in &result.folds {
        assert!(fold.rmse.is_finite());
        assert!(fold.n_test_obs > 0);
        assert!(fold.n_train_obs > 0);
    }
}

#[test]
fn test_cv_grouped_reproducible_with_seed() {
    let df = load_sleepstudy();
    let a = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        3,
        true,
        Some(7),
        Some(1),
    )
    .unwrap();
    let b = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        3,
        true,
        Some(7),
        Some(1),
    )
    .unwrap();
    assert_eq!(a.oof_predictions, b.oof_predictions);
    assert_eq!(a.test_fold, b.test_fold);
}

#[test]
fn test_cv_grouped_parallel_matches_sequential() {
    let df = load_sleepstudy();
    let sequential = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        5,
        true,
        Some(99),
        Some(1),
    )
    .unwrap();
    let parallel = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        5,
        true,
        Some(99),
        Some(4),
    )
    .unwrap();

    assert_eq!(sequential.oof_predictions, parallel.oof_predictions);
    assert_eq!(sequential.test_fold, parallel.test_fold);
    assert_eq!(sequential.rmse, parallel.rmse);
    assert_eq!(sequential.mae, parallel.mae);
    assert_eq!(sequential.folds.len(), parallel.folds.len());
    for (a, b) in sequential.folds.iter().zip(parallel.folds.iter()) {
        assert_eq!(a.fold, b.fold);
        assert!((a.rmse - b.rmse).abs() < 1e-10);
        assert_eq!(a.converged, b.converged);
    }
}

#[test]
fn test_cv_grouped_too_many_splits() {
    let df = load_sleepstudy();
    let err = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        100,
        true,
        None,
        None,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("n_splits"),
        "expected n_splits error, got: {err}"
    );
}

#[test]
fn test_cv_grouped_invalid_n_jobs() {
    let df = load_sleepstudy();
    let err = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        3,
        true,
        Some(1),
        Some(0),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("n_jobs"),
        "expected n_jobs error, got: {err}"
    );
}

#[test]
fn test_prepare_lmer_fit_prepared_refit() {
    let df = load_sleepstudy();
    let prep = prepare_lmer("Reaction ~ Days + (1 | Subject)", &df).unwrap();
    let fit_reml = fit_prepared(&prep, true).unwrap();
    let fit_ml = fit_prepared(&prep, false).unwrap();
    let fit_refit = refit_lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();

    assert!(fit_reml.reml.is_some());
    assert!(fit_ml.reml.is_none());
    assert!(
        (fit_reml.coefficients[0] - fit_refit.coefficients[0]).abs() < 1e-10,
        "refit_lmer should match fit_prepared"
    );

    let cold = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    assert!(
        (fit_reml.coefficients[0] - cold.coefficients[0]).abs() < 1e-8,
        "fit_prepared should match lmer"
    );
}

fn synthetic_binary_cv() -> DataFrame {
    let n_groups = 12usize;
    let n_per = 5usize;
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for gi in 0..n_groups {
        for j in 0..n_per {
            let xv = (j as f64) / (n_per as f64);
            let lp = -0.4 + 1.0 * xv + 0.25 * ((gi % 4) as f64 - 1.5);
            let p = 1.0 / (1.0 + (-lp).exp());
            y.push(if p > 0.5 { 1.0 } else { 0.0 });
            x.push(xv);
            g.push(format!("g{gi}"));
        }
    }
    DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("g".into(), g),
    ])
    .unwrap()
}

#[test]
fn test_cv_grouped_glmer_binomial() {
    use lme_rs::cv_grouped_glmer;
    use lme_rs::family::{Family, Link};

    let df = synthetic_binary_cv();
    let result = cv_grouped_glmer(
        "y ~ x + (1 | g)",
        &df,
        "g",
        4,
        Family::Binomial,
        Link::Logit,
        1,
        None,
        Some(7),
        Some(1),
    )
    .unwrap();

    assert_eq!(result.n_splits, 4);
    assert!(result.rmse.is_finite());
    assert!(result.mae.is_finite());
    assert!(result.mean_log_loss.is_some());
    let ll = result.mean_log_loss.unwrap();
    assert!(ll.is_finite() && ll > 0.0);
    assert_eq!(result.oof_predictions.len(), df.height());
    for &p in result.oof_predictions.iter() {
        assert!(
            (0.0..=1.0).contains(&p),
            "response-scale pred out of range: {p}"
        );
    }
}
