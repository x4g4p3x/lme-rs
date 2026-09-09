//! Additional public-API regressions from the September 2026 bug hunt.

use lme_rs::family::{Family, Link};
use lme_rs::{boot_lmer, cv_grouped, cv_grouped_glmer, lm_df, lmer, BootLmerMethod};
use polars::prelude::*;

fn sleepstudy() -> DataFrame {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
        .unwrap()
        .finish()
        .unwrap()
}

#[test]
fn predictor_named_intercept_is_not_dropped() {
    let mut df = sleepstudy();
    df.rename("Days", "intercept".into()).unwrap();
    for (formula, reference) in [
        ("Reaction ~ intercept", "Reaction ~ Days"),
        ("Reaction ~ 0 + intercept", "Reaction ~ 0 + Days"),
        (
            "Reaction ~ intercept + I(intercept^2)",
            "Reaction ~ Days + I(Days^2)",
        ),
    ] {
        let got = lm_df(formula, &df).unwrap();
        let expected = lm_df(reference, &sleepstudy()).unwrap();
        assert_eq!(got.coefficients.len(), expected.coefficients.len());
        assert!((&got.coefficients - &expected.coefficients)
            .iter()
            .all(|v| v.abs() < 1e-8));
        let predicted = got.predict(&df).unwrap();
        assert!((&predicted - &expected.fitted)
            .iter()
            .all(|v| v.abs() < 1e-8));
    }
    let got = lmer("Reaction ~ intercept + (intercept | Subject)", &df, true).unwrap();
    let expected = lmer("Reaction ~ Days + (Days | Subject)", &sleepstudy(), true).unwrap();
    assert_eq!(got.coefficients.len(), expected.coefficients.len());
    assert!((&got.coefficients - &expected.coefficients)
        .iter()
        .all(|v| v.abs() < 1e-6));
    let predicted = got.predict_conditional(&df, false).unwrap();
    assert!((&predicted - &expected.fitted)
        .iter()
        .all(|v| v.abs() < 1e-6));
}

#[test]
fn response_named_intercept_is_not_confused_with_constant_term() {
    let df = sleepstudy();
    let mut renamed = df.clone();
    renamed.rename("Reaction", "intercept".into()).unwrap();
    let a = lm_df("Reaction ~ Days", &df).unwrap();
    let b = lm_df("intercept ~ Days", &renamed).unwrap();
    assert_eq!(a.coefficients, b.coefficients);
    assert_eq!(a.fitted, b.predict(&renamed).unwrap());
}

fn missing_cv_group_data() -> DataFrame {
    let mut df = sleepstudy();
    let mut split: Vec<Option<String>> = (0..df.height())
        .map(|i| Some(format!("g{}", i / 10)))
        .collect();
    split[3] = None;
    df.with_column(Column::new("split".into(), split)).unwrap();
    df
}

#[test]
fn cv_lmm_rejects_missing_split_labels_without_panicking() {
    let df = missing_cv_group_data();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cv_grouped(
            "Reaction ~ Days + (1 | Subject)",
            &df,
            "split",
            3,
            true,
            Some(42),
            Some(1),
        )
    }))
    .expect("missing split labels must return an error, not panic");
    assert!(
        result.is_err(),
        "missing split labels must not produce partial CV results"
    );
}

#[test]
fn cv_glmm_rejects_missing_split_labels_without_panicking() {
    let mut df = missing_cv_group_data();
    df.with_column(Column::new(
        "y".into(),
        (0..df.height()).map(|i| (i % 3) as f64).collect::<Vec<_>>(),
    ))
    .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cv_grouped_glmer(
            "y ~ Days + (1 | Subject)",
            &df,
            "split",
            3,
            Family::Poisson,
            Link::Log,
            1,
            None,
            Some(42),
            Some(1),
        )
    }))
    .expect("missing split labels must return an error, not panic");
    assert!(
        result.is_err(),
        "missing split labels must not produce partial CV results"
    );
}

#[test]
fn wald_intervals_reject_nan_confidence() {
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &sleepstudy(), true).unwrap();
    assert!(fit.confint(f64::NAN).is_err());
}

#[test]
fn bootstrap_intervals_reject_nan_confidence() {
    let df = sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer(formula, &df, true).unwrap();
    let boot = boot_lmer(
        formula,
        &df,
        &fit,
        4,
        BootLmerMethod::Parametric,
        true,
        Some(42),
        Some(1),
    )
    .unwrap();
    assert!(boot.confint_percentile(f64::NAN).is_err());
}

#[test]
fn fixed_profile_intervals_reject_nan_confidence() {
    let df = sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    assert!(fit.confint_profile(f64::NAN, &df).is_err());
}

#[test]
fn variance_profile_intervals_reject_nan_confidence() {
    let df = sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    assert!(fit.confint_profile_vc(f64::NAN, &df).is_err());
}

#[test]
fn robust_inference_rejects_wrong_row_count_without_panicking() {
    let df = sleepstudy();
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    for shorter in [true, false] {
        let other = if shorter {
            df.slice(0, df.height() - 1)
        } else {
            df.vstack(&df.slice(0, 1)).unwrap()
        };
        let mut copy = fit.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copy.with_robust_se(&other, Some("Subject")).map(|_| ())
        }))
        .expect("wrong row count must return an error, not panic");
        assert!(result.is_err());
    }
}

#[test]
fn robust_inference_rejects_missing_cluster_labels() {
    let df = missing_cv_group_data();
    let mut fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    assert!(fit.with_robust_se(&df, Some("split")).is_err());
}

#[test]
fn empty_group_labels_remain_valid_for_cv_and_robust_inference() {
    let mut df = sleepstudy();
    let labels: Vec<String> = (0..df.height())
        .map(|i| {
            if i < 10 {
                String::new()
            } else {
                format!("g{}", i / 10)
            }
        })
        .collect();
    df.with_column(Column::new("split".into(), labels)).unwrap();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let cv = cv_grouped(formula, &df, "split", 3, true, Some(42), Some(1)).unwrap();
    assert!(cv.oof_predictions.iter().all(|v| v.is_finite()));
    assert!(cv.test_fold.iter().all(|&f| f >= 0));
    let mut fit = lmer(formula, &df, true).unwrap();
    fit.with_robust_se(&df, Some("split")).unwrap();
    assert!(fit
        .robust
        .as_ref()
        .unwrap()
        .robust_se
        .iter()
        .all(|v| v.is_finite()));
}
