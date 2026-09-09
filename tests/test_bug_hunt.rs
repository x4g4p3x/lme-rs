//! Reproductions discovered during the 2026-09-09 bug hunt.

use lme_rs::family::{Family, Link};
use lme_rs::{cv_grouped, cv_grouped_glmer, lmer, prepare_lmer_weighted, FitControl};
use ndarray::Array1;
use polars::prelude::*;

fn assert_kr_equivalent(
    a: &mut lme_rs::LmeFit,
    a_data: &DataFrame,
    b: &mut lme_rs::LmeFit,
    b_data: &DataFrame,
) {
    a.with_kenward_roger(a_data).unwrap();
    b.with_kenward_roger(b_data).unwrap();
    let akr = a.kenward_roger.as_ref().unwrap();
    let bkr = b.kenward_roger.as_ref().unwrap();
    assert!(
        (&akr.dfs - &bkr.dfs).iter().all(|v| v.abs() < 0.02),
        "KR df: {:?} versus {:?}",
        akr.dfs,
        bkr.dfs
    );
    assert!((&akr.p_values - &bkr.p_values)
        .iter()
        .all(|v| v.abs() < 1e-4));
    let contrast = ndarray::Array2::eye(a.coefficients.len());
    let at = a
        .test_contrast(&contrast, lme_rs::anova::DdfMethod::KenwardRoger)
        .unwrap();
    let bt = b
        .test_contrast(&contrast, lme_rs::anova::DdfMethod::KenwardRoger)
        .unwrap();
    assert!(
        (at.f_value - bt.f_value).abs() < 0.01,
        "KR F: {} versus {}",
        at.f_value,
        bt.f_value
    );
    assert!((at.den_df - bt.den_df).abs() < 0.02);
    assert!((at.p_value - bt.p_value).abs() < 1e-4);
}

fn assert_balanced_cv(cv: &lme_rs::CvGroupedResult, groups: usize) {
    assert_eq!(cv.folds.len(), cv.n_splits);
    let sizes: Vec<usize> = cv.folds.iter().map(|f| f.n_test_groups).collect();
    assert_eq!(sizes.iter().sum::<usize>(), groups);
    assert!(sizes.iter().all(|&n| n > 0));
    assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
    assert!(cv.oof_predictions.iter().all(|v| v.is_finite()));
    for fold in 0..cv.n_splits {
        assert_eq!(
            cv.test_fold.iter().filter(|&&f| f == fold as i32).count(),
            cv.folds[fold].n_test_obs
        );
    }
}

fn sleepstudy() -> DataFrame {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
        .unwrap()
        .finish()
        .unwrap()
}

#[test]
fn cv_lmm_honors_requested_fold_count() {
    let df = sleepstudy();
    for n_splits in [2, 5, 10, 13, 18] {
        let cv = cv_grouped(
            "Reaction ~ Days + (1 | Subject)",
            &df,
            "Subject",
            n_splits,
            true,
            Some(42),
            Some(1),
        )
        .unwrap();
        assert_balanced_cv(&cv, 18);
    }
}

#[test]
fn cv_glmm_honors_requested_fold_count() {
    let mut df = sleepstudy();
    df.with_column(Column::new(
        "y".into(),
        (0..df.height()).map(|i| (i % 3) as f32).collect::<Vec<_>>(),
    ))
    .unwrap();
    for n_splits in [2, 5, 10, 13, 18] {
        let cv = cv_grouped_glmer(
            "y ~ Days + (1 | Subject)",
            &df,
            "Subject",
            n_splits,
            Family::Poisson,
            Link::Log,
            1,
            None,
            Some(42),
            Some(1),
        )
        .unwrap();
        assert_balanced_cv(&cv, 18);
    }
}

#[test]
fn nested_group_predictions_reproduce_training_fitted_values() {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for (ai, bi, intercept) in [
        ("a_b", "c", 0.0),
        ("a", "b_c", 10.0),
        ("d", "e", -5.0),
        ("f", "g", 20.0),
        ("", "a_b_c", 30.0),
        ("a\\", "b_c", -8.0),
        ("a\\_b", "c", 8.0),
    ] {
        for i in 0..8 {
            a.push(ai);
            b.push(bi);
            x.push(i as f64);
            y.push(intercept + 0.5 * i as f64 + if i % 2 == 0 { 0.2 } else { -0.2 });
        }
    }
    let mut df = df!("y" => y, "x" => x, "a" => a, "b" => b).unwrap();
    df.with_column(Column::new("c".into(), vec!["tail_\\"; df.height()]))
        .unwrap();
    for formula in ["y ~ x + (1 | a:b)", "y ~ x + (1 | a:b:c)"] {
        let fit = lmer(formula, &df, true).unwrap();
        // Prediction must retain identities when levels are encountered in another order.
        let prediction = fit.predict_conditional(&df.reverse(), false).unwrap();
        let fitted_reversed = Array1::from_iter(fit.fitted.iter().rev().copied());
        let delta = (&prediction - &fitted_reversed)
            .mapv(f64::abs)
            .fold(0.0_f64, |a, &b| a.max(b));
        let block = &fit.re_blocks.as_ref().unwrap()[0];
        assert_eq!(block.m, block.group_map.len());
        assert!(
            delta < 1e-8,
            "training prediction differs from fitted by {delta}"
        );
        // This tuple would alias the training tuple ("", "a_b_c") without escaping.
        let novel = df!("x" => [0.0], "a" => ["_a"], "b" => ["b_c"], "c" => ["tail_\\"]).unwrap();
        assert!(fit.predict_conditional(&novel, false).is_err());
        assert_eq!(
            fit.predict_conditional(&novel, true).unwrap(),
            fit.predict(&novel).unwrap()
        );
    }
}

#[test]
fn inference_is_invariant_to_explicit_offset() {
    let df = sleepstudy();
    let mut shifted = df.clone();
    let offset: Vec<f64> = (0..df.height()).map(|i| (i % 7) as f64 * 50.0).collect();
    let y: Vec<f64> = df
        .column("Reaction")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .zip(&offset)
        .map(|(y, o)| y + o)
        .collect();
    shifted
        .with_column(Column::new("Reaction".into(), y))
        .unwrap();
    shifted
        .with_column(Column::new("off".into(), offset))
        .unwrap();
    let mut plain = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let mut with_offset = lmer(
        "Reaction ~ Days + offset(off) + (1 | Subject)",
        &shifted,
        true,
    )
    .unwrap();
    assert!((&plain.coefficients - &with_offset.coefficients)
        .iter()
        .all(|v| v.abs() < 1e-6));
    assert!((plain.sigma2.unwrap() - with_offset.sigma2.unwrap()).abs() < 1e-5);
    plain.with_satterthwaite(&df).unwrap();
    with_offset.with_satterthwaite(&shifted).unwrap();
    let a = &plain.satterthwaite.as_ref().unwrap().dfs;
    let b = &with_offset.satterthwaite.as_ref().unwrap().dfs;
    assert!(
        (a - b).iter().all(|v| v.abs() < 0.01),
        "equivalent fits must have the same degrees of freedom"
    );
    assert_kr_equivalent(&mut plain, &df, &mut with_offset, &shifted);
}

#[test]
fn inference_is_invariant_to_uniform_weight_scaling() {
    let df = sleepstudy();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let control = FitControl {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut a = prepare_lmer_weighted(formula, &df, Some(Array1::ones(df.height())))
        .unwrap()
        .fit(None, true, &control)
        .unwrap();
    let mut b = prepare_lmer_weighted(formula, &df, Some(Array1::from_elem(df.height(), 9.0)))
        .unwrap()
        .fit(None, true, &control)
        .unwrap();
    assert!((&a.coefficients - &b.coefficients)
        .iter()
        .all(|v| v.abs() < 1e-5));
    assert!((a.beta_se.as_ref().unwrap() - b.beta_se.as_ref().unwrap())
        .iter()
        .all(|v| v.abs() < 1e-4));
    a.with_satterthwaite(&df).unwrap();
    b.with_satterthwaite(&df).unwrap();
    let adf = &a.satterthwaite.as_ref().unwrap().dfs;
    let bdf = &b.satterthwaite.as_ref().unwrap().dfs;
    assert!(
        (adf - bdf).iter().all(|v| v.abs() < 0.01),
        "uniform precision scaling must preserve inference"
    );
    assert_kr_equivalent(&mut a, &df, &mut b, &df);
}

#[test]
fn cv_accepts_numeric_response_types() {
    for dtype in [DataType::Float32, DataType::Int32, DataType::UInt32] {
        let mut df = sleepstudy();
        let y = df.column("Reaction").unwrap().cast(&dtype).unwrap();
        df.with_column(y).unwrap();
        lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
        let cv = cv_grouped(
            "Reaction ~ Days + (1 | Subject)",
            &df,
            "Subject",
            3,
            true,
            Some(42),
            Some(1),
        )
        .unwrap();
        let as_f64 = df
            .column("Reaction")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        df.with_column(as_f64).unwrap();
        let reference = cv_grouped(
            "Reaction ~ Days + (1 | Subject)",
            &df,
            "Subject",
            3,
            true,
            Some(42),
            Some(1),
        )
        .unwrap();
        assert_eq!(cv.oof_predictions, reference.oof_predictions);
        assert_eq!(cv.rmse, reference.rmse);
    }
}

#[test]
fn cv_rejects_missing_and_nonfinite_responses() {
    for bad in [None, Some(f64::NAN), Some(f64::INFINITY)] {
        let mut df = sleepstudy();
        let mut y: Vec<Option<f64>> = df
            .column("Reaction")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect();
        y[0] = bad;
        df.with_column(Column::new("Reaction".into(), y)).unwrap();
        let result = cv_grouped(
            "Reaction ~ Days + (1 | Subject)",
            &df,
            "Subject",
            3,
            true,
            Some(42),
            Some(1),
        );
        assert!(matches!(result, Err(lme_rs::LmeError::InvalidInput { .. })));
    }
}

#[test]
fn weighted_inference_matches_explicitly_whitened_model() {
    let df = sleepstudy();
    let weights = Array1::from_iter((0..df.height()).map(|i| 1.0 + (i % 4) as f64));
    let sqrt_w = weights.mapv(f64::sqrt);
    let reaction = df.column("Reaction").unwrap().f64().unwrap();
    let days = df.column("Days").unwrap().cast(&DataType::Float64).unwrap();
    let y: Vec<f64> = reaction
        .into_no_null_iter()
        .zip(&sqrt_w)
        .map(|(y, w)| y * w)
        .collect();
    let x: Vec<f64> = days
        .f64()
        .unwrap()
        .into_no_null_iter()
        .zip(&sqrt_w)
        .map(|(x, w)| x * w)
        .collect();
    let whitened = DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("one_w".into(), sqrt_w.to_vec()),
        df.column("Subject").unwrap().clone(),
    ])
    .unwrap();
    let control = FitControl {
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut weighted = prepare_lmer_weighted("Reaction ~ Days + (1 | Subject)", &df, Some(weights))
        .unwrap()
        .fit(None, true, &control)
        .unwrap();
    let mut explicit =
        prepare_lmer_weighted("y ~ 0 + one_w + x + (0 + one_w | Subject)", &whitened, None)
            .unwrap()
            .fit(None, true, &control)
            .unwrap();
    assert!((&weighted.coefficients - &explicit.coefficients)
        .iter()
        .all(|v| v.abs() < 1e-6));
    assert!(
        (weighted.beta_se.as_ref().unwrap() - explicit.beta_se.as_ref().unwrap())
            .iter()
            .all(|v| v.abs() < 1e-5)
    );
    weighted.with_satterthwaite(&df).unwrap();
    explicit.with_satterthwaite(&whitened).unwrap();
    assert!((&weighted.satterthwaite.as_ref().unwrap().dfs
        - &explicit.satterthwaite.as_ref().unwrap().dfs)
        .iter()
        .all(|v| v.abs() < 0.01));
    assert_kr_equivalent(&mut weighted, &df, &mut explicit, &whitened);
}
