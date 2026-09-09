//! Numerical invariance and unavailable-inference regressions.

use lme_rs::anova::DdfMethod;
use lme_rs::mcp::{McpAdjust, McpType};
use lme_rs::{lm_df, lmer, prepare_lmer_weighted, FitControl};
use ndarray::{array, Array1};
use polars::prelude::*;

fn sleepstudy() -> DataFrame {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
        .unwrap()
        .finish()
        .unwrap()
}

#[test]
fn robust_covariance_is_invariant_to_uniform_precision_scaling() {
    let df = sleepstudy();
    let control = FitControl {
        tolerance: 1e-12,
        ..Default::default()
    };
    let fit = |w| {
        prepare_lmer_weighted(
            "Reaction ~ Days + (1 | Subject)",
            &df,
            Some(Array1::from_elem(df.height(), w)),
        )
        .unwrap()
        .fit(None, true, &control)
        .unwrap()
    };
    let mut a = fit(1.0);
    // Express the same fitted likelihood in precision units nine times larger.
    // This avoids optimizer noise while testing the exact scale invariance.
    let mut b = a.clone();
    b.weights = Some(Array1::from_elem(df.height(), 9.0));
    b.sigma2 = b.sigma2.map(|s| s * 9.0);
    b.theta = b.theta.map(|theta| theta / 3.0);
    b.v_beta_unscaled = b.v_beta_unscaled.map(|v| v / 9.0);
    for cluster in [None, Some("Subject")] {
        a.with_robust_se(&df, cluster).unwrap();
        b.with_robust_se(&df, cluster).unwrap();
        let va = &a.robust.as_ref().unwrap().v_beta_robust;
        let vb = &b.robust.as_ref().unwrap().v_beta_robust;
        assert!(
            va.iter().zip(vb).all(|(a, b)| (a - b).abs() < 1e-4),
            "precision scaling changed robust covariance: {va:?} vs {vb:?}"
        );
    }
}

#[test]
fn robust_covariance_matches_explicit_whitening() {
    let df = sleepstudy();
    let weights = Array1::from_iter((0..df.height()).map(|i| 1.0 + (i % 4) as f64));
    let sqrt_w = weights.mapv(f64::sqrt);
    let days = df.column("Days").unwrap().cast(&DataType::Float64).unwrap();
    let whitened = DataFrame::new(vec![
        Column::new(
            "y".into(),
            df.column("Reaction")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .zip(&sqrt_w)
                .map(|(y, w)| y * w)
                .collect::<Vec<_>>(),
        ),
        Column::new(
            "x".into(),
            days.f64()
                .unwrap()
                .into_no_null_iter()
                .zip(&sqrt_w)
                .map(|(x, w)| x * w)
                .collect::<Vec<_>>(),
        ),
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
    for cluster in [None, Some("Subject")] {
        weighted.with_robust_se(&df, cluster).unwrap();
        explicit.with_robust_se(&whitened, cluster).unwrap();
        let a = &weighted.robust.as_ref().unwrap().v_beta_robust;
        let b = &explicit.robust.as_ref().unwrap().v_beta_robust;
        assert!(
            a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-4),
            "weighted and whitened covariance differ: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn saturated_ols_preserves_predictions_without_inventing_uncertainty() {
    let df = df!("y" => [2.0, 5.0], "x" => [0.0, 1.0]).unwrap();
    let fit = lm_df("y ~ x", &df).unwrap();
    assert!(fit.sigma2.is_none());
    assert!((&fit.predict(&df).unwrap() - &array![2.0, 5.0])
        .iter()
        .all(|v| v.abs() < 1e-12));
    assert!(
        fit.beta_se.is_none(),
        "zero residual df cannot estimate standard errors"
    );
    assert!(fit.beta_t.is_none());
    assert!(fit.confint(0.95).is_err());
}

#[test]
fn contrast_rejects_nonfinite_weights_and_null_values() {
    let df = sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    fit.with_satterthwaite(&df).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    for method in [DdfMethod::Satterthwaite, DdfMethod::KenwardRoger] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                fit.test_contrast(&array![[0.0, value]], method).is_err(),
                "nonfinite contrast must be rejected for {method:?}"
            );
            assert!(fit
                .test_contrast_vs(&array![[0.0, 1.0]], &array![0.0, value], method)
                .is_err());
        }
    }
}

#[test]
fn kr_equivalent_single_coefficient_hypotheses_agree() {
    let df = sleepstudy().slice(0, 137);
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    let method = DdfMethod::KenwardRoger;
    let a = fit.test_contrast(&array![[0.0, 1.0]], method).unwrap();
    assert!((a.den_df - fit.kenward_roger.as_ref().unwrap().dfs[1]).abs() < 1e-10);
    assert!((a.p_value - fit.kenward_roger.as_ref().unwrap().p_values[1]).abs() < 1e-12);
    let b = fit.test_contrast(&array![[0.0, 2.0]], method).unwrap();
    let c = fit
        .test_contrast_vs(&array![[0.0, 1.0]], &array![0.0, 0.0], method)
        .unwrap();
    for other in [b, c] {
        assert!(
            (a.f_value - other.f_value).abs() < 1e-8,
            "equivalent KR F values differ: {a:?} vs {other:?}"
        );
        assert!(
            (a.den_df - other.den_df).abs() < 1e-8,
            "equivalent KR denominator df differ: {a:?} vs {other:?}"
        );
        assert!((a.p_value - other.p_value).abs() < 1e-10);
    }
}

#[test]
fn kr_unbalanced_model_matches_pbkrtest_reference() {
    // R 4.6.1, lme4 2.0.1, pbkrtest 0.5.5; lmer with bobyqa/rhoend=1e-10, then
    // pbkrtest::KRmodcomp(m, L)$stats. Retain 3..10 days per subject.
    let all = sleepstudy();
    let mask = BooleanChunked::from_iter_values(
        "".into(),
        (0..all.height()).map(|i| i % 10 < 3 + (i / 10) % 8),
    );
    let df = all.filter(&mask).unwrap();
    let mut fit = prepare_lmer_weighted("Reaction ~ Days + (Days | Subject)", &df, None)
        .unwrap()
        .fit(
            None,
            true,
            &FitControl {
                tolerance: 1e-10,
                max_iterations: 10000,
                require_convergence: true,
                ..Default::default()
            },
        )
        .unwrap();
    fit.with_kenward_roger(&df).unwrap();
    let ci = fit.confint(0.95).unwrap();
    use statrs::distribution::{ContinuousCDF, StudentsT};
    for (j, variance) in [60.77981382064677_f64, 3.83638564771022]
        .into_iter()
        .enumerate()
    {
        let df = fit.kenward_roger.as_ref().unwrap().dfs[j];
        let critical = StudentsT::new(0.0, 1.0, df).unwrap().inverse_cdf(0.975);
        let se = (ci.upper[j] - ci.lower[j]) / (2.0 * critical);
        assert!((se / variance.sqrt() - 1.0).abs() < 1e-4);
    }
    for (l, f, df, p) in [
        (
            array![[0.0, 1.0]],
            25.7645262195361,
            13.4821291779379,
            0.000189900632423985,
        ),
        (
            array![[1.0, 1.0]],
            1395.41169753153,
            16.9938459099754,
            9.36462730427476e-18,
        ),
        (
            array![[1.0, 0.0], [0.0, 1.0]],
            775.973442894987,
            14.2437573628656,
            2.90187288597778e-15,
        ),
    ] {
        let result = fit.test_contrast(&l, DdfMethod::KenwardRoger).unwrap();
        assert!(
            (result.f_value / f - 1.0).abs() < 1e-4,
            "F mismatch: {result:?}"
        );
        assert!((result.den_df - df).abs() < 1e-3, "df mismatch: {result:?}");
        assert!(
            (result.p_value / p - 1.0).abs() < 1e-3,
            "p mismatch: {result:?}"
        );
    }
}

#[test]
fn kr_marginal_means_and_multiple_comparisons_use_adjusted_uncertainty() {
    let all = sleepstudy();
    let mask = BooleanChunked::from_iter_values(
        "".into(),
        (0..all.height()).map(|i| i % 10 < 3 + (i / 10) % 8),
    );
    let mut df = all.filter(&mask).unwrap();
    let days = df.column("Days").unwrap().cast(&DataType::Float64).unwrap();
    df.with_column(Column::new(
        "phase".into(),
        days.f64()
            .unwrap()
            .into_no_null_iter()
            .map(|d| if d < 3.0 { "early" } else { "late" })
            .collect::<Vec<_>>(),
    ))
    .unwrap();
    let mut fit = lmer("Reaction ~ phase + (Days | Subject)", &df, true).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    let method = DdfMethod::KenwardRoger;
    let means = fit.emmeans("phase", &df, 0.95, Some(method)).unwrap();
    for i in 0..means.levels.len() {
        let l = means.linfct.slice(ndarray::s![i..i + 1, ..]).to_owned();
        let test = fit.test_contrast(&l, method).unwrap();
        let t = means.estimate[i] / means.std_error[i];
        assert!((t * t / test.f_value - 1.0).abs() < 1e-10);
    }
    let pairs = fit
        .emmeans_pairs("phase", &df, McpAdjust::None, Some(method))
        .unwrap();
    let glht = fit
        .glht("phase", McpType::Tukey, McpAdjust::None, Some(method))
        .unwrap();
    let l = (&means.linfct.row(1) - &means.linfct.row(0)).insert_axis(ndarray::Axis(0));
    let test = fit.test_contrast(&l, method).unwrap();
    for t in [pairs.statistic_values[0], glht.statistic_values[0]] {
        assert!((t * t / test.f_value - 1.0).abs() < 1e-10);
    }
    assert!((pairs.p_value[0] - test.p_value).abs() < 1e-12);
    assert!((glht.p_value[0] - test.p_value).abs() < 1e-12);
}

#[test]
fn saturated_ols_rejects_marginal_mean_uncertainty() {
    let df = df!("y" => [2.0, 5.0], "group" => ["a", "b"]).unwrap();
    let fit = lm_df("y ~ group", &df).unwrap();
    assert!(fit.emmeans("group", &df, 0.95, None).is_err());
}
