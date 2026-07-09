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
    )
    .unwrap();
    let b = cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        &df,
        "Subject",
        3,
        true,
        Some(7),
    )
    .unwrap();
    assert_eq!(a.oof_predictions, b.oof_predictions);
    assert_eq!(a.test_fold, b.test_fold);
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
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("n_splits"),
        "expected n_splits error, got: {err}"
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
