//! Edge case tests for lm() and lmer() behavior.

use lme_rs::{lm, lmer};
use ndarray::array;
use polars::prelude::*;

#[test]
fn test_lm_perfect_fit_zero_residuals() {
    // When X has as many columns as rows, residuals should be ~0 (exact fit)
    let y = array![1.0, 2.0];
    let x = array![[1.0, 1.0], [1.0, 2.0],];
    let fit = lm(&y, &x).unwrap();

    for &r in fit.residuals.iter() {
        assert!(
            r.abs() < 1e-12,
            "Residual should be ~0 for perfect fit, got {}",
            r
        );
    }
    // sigma2 should be None (no degrees of freedom for estimation)
    assert!(fit.sigma2.is_none());
}

#[test]
fn test_lm_display_formatting() {
    let y = array![1.0, 2.0, 3.0, 4.0];
    let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
    let fit = lm(&y, &x).unwrap();
    let summary = format!("{}", fit);

    assert!(
        summary.contains("Scaled residuals:"),
        "lm display should have scaled residuals"
    );
    assert!(
        summary.contains("Fixed effects:"),
        "lm display should have fixed effects"
    );
}

#[test]
fn test_lm_single_predictor() {
    // Simple regression y = 2 + 3*x
    let y = array![5.0, 8.0, 11.0, 14.0];
    let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
    let fit = lm(&y, &x).unwrap();

    assert!(
        (fit.coefficients[0] - 2.0).abs() < 1e-10,
        "Intercept should be 2.0"
    );
    assert!(
        (fit.coefficients[1] - 3.0).abs() < 1e-10,
        "Slope should be 3.0"
    );

    // Fitted values should exactly match y
    for i in 0..4 {
        assert!(
            (fit.fitted[i] - y[i]).abs() < 1e-10,
            "Fitted[{}] mismatch",
            i
        );
    }
}

fn load_sleepstudy() -> DataFrame {
    let mut file =
        std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("Failed to read CSV")
}

#[test]
fn test_intercept_only_predict_is_constant() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ 1 + (1 | Subject)", &df, true).unwrap();

    // With no fixed-effect slope, predictions should be constant (= intercept)
    let new_days = Series::new("Days".into(), &[0.0, 5.0, 10.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();

    let preds = fit.predict(&newdata).unwrap();

    // All population-level predictions should be the same (intercept only)
    assert!(
        (preds[0] - preds[1]).abs() < 1e-10,
        "Intercept-only model should give constant predictions"
    );
    assert!(
        (preds[0] - preds[2]).abs() < 1e-10,
        "Intercept-only model should give constant predictions"
    );

    // The prediction should match the fitted intercept
    assert!(
        (preds[0] - fit.coefficients[0]).abs() < 1e-10,
        "Population pred should equal intercept: {} vs {}",
        preds[0],
        fit.coefficients[0]
    );
}

#[test]
fn test_conditional_predict_intercept_only() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ 1 + (1 | Subject)", &df, true).unwrap();

    // Conditional predictions for a known subject should differ from population
    let new_days = Series::new("Days".into(), &[0.0]);
    let new_subject = Series::new("Subject".into(), &["308"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();

    let pop = fit.predict(&newdata).unwrap();
    let cond = fit.predict_conditional(&newdata, true).unwrap();

    // Population = intercept, conditional = intercept + b_308
    // These should differ since Subject 308 has a non-zero random effect
    let diff = (pop[0] - cond[0]).abs();
    assert!(
        diff > 0.1,
        "Conditional should differ from population for known subject, diff = {}",
        diff
    );
}

#[test]
fn test_fitted_plus_residuals_equals_y() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let y_series = df
        .column("Reaction")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    let y_f64 = y_series.f64().unwrap();

    for i in 0..fit.num_obs {
        let y_i = y_f64.get(i).unwrap();
        let reconstructed = fit.fitted[i] + fit.residuals[i];
        assert!(
            (reconstructed - y_i).abs() < 1e-8,
            "y[{}] = {} but fitted + residuals = {}",
            i,
            y_i,
            reconstructed
        );
    }
}

#[test]
fn test_multiple_subjects_conditional() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Predict for two different known subjects at same Day value
    let new_days = Series::new("Days".into(), &[5.0, 5.0]);
    let new_subject = Series::new("Subject".into(), &["308", "309"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();

    let pop = fit.predict(&newdata).unwrap();
    let cond = fit.predict_conditional(&newdata, true).unwrap();

    // Population predictions should be identical (same Day)
    assert!(
        (pop[0] - pop[1]).abs() < 1e-10,
        "Population preds for same Day should be equal"
    );

    // Conditional predictions should differ (different subjects have different REs)
    assert!(
        (cond[0] - cond[1]).abs() > 0.01,
        "Conditional preds for different subjects should differ: {} vs {}",
        cond[0],
        cond[1]
    );
}
