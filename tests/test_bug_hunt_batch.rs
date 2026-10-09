//! Public regressions for the ten-defect October batch.

use lme_rs::family::{CloglogLink, Family, GlmLink, LogLink};
use lme_rs::{lm_df, BootLmerMethod, BootLmerResult, BootReplicate};
use ndarray::{array, Array1};
use polars::prelude::*;

fn data() -> DataFrame {
    df!("y" => [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1],
        "x" => [1., 2., 3., 4., 5., 6., 7., 8.],
        "a" => ["A", "B", "A", "B", "A", "B", "A", "B"],
        "id" => [1, 1, 2, 2, 3, 3, 4, 4])
    .unwrap()
}

#[test]
fn sparse_contrast_repeated_coefficients_add() {
    let got = lme_rs::contrast_matrix(2, &[vec![(0, 1.), (0, 2.), (1, -1.)]]);
    assert_eq!(got, array![[3., -1.]]);
    let names = vec!["x".into(), "a".into()];
    let row = lme_rs::ContrastRowSpec {
        label: "sum",
        weights: &[("x", 1.), ("x", 2.), ("a", -1.)],
    };
    assert_eq!(
        lme_rs::contrast_matrix_from_names(&names, &[row]).unwrap(),
        got
    );
    assert_eq!(
        lme_rs::contrast_matrix(2, &[vec![(0, 2.), (0, -2.)]]),
        array![[0., 0.]]
    );
    let invalid = lme_rs::ContrastRowSpec {
        label: "unknown",
        weights: &[("missing", 1.)],
    };
    assert!(lme_rs::contrast_matrix_from_names(&names, &[invalid]).is_err());
}

#[test]
fn dot_prediction_uses_training_sources_and_order() {
    let training = data().select(["y", "x", "a"]).unwrap();
    let fit = lm_df("y ~ .", &training).unwrap();
    let expected = fit.predict(&training).unwrap();
    let mut newdata = training.select(["a", "x"]).unwrap();
    newdata
        .with_column(Column::new("unused".into(), vec![None::<f64>; 8]))
        .unwrap();
    assert_eq!(fit.predict(&newdata).unwrap(), expected);
    assert!(fit.predict(&newdata.select(["a"]).unwrap()).is_err());
    let one = df!("a" => ["B"], "x" => [2.]).unwrap();
    assert!((fit.predict(&one).unwrap()[0] - expected[1]).abs() < 1e-14);
}

#[test]
fn dot_marginal_means_ignore_new_metadata() {
    let training = data().select(["y", "x", "a"]).unwrap();
    let fit = lm_df("y ~ .", &training).unwrap();
    let expected = fit
        .emmeans("a", &training, 0.95, Some(lme_rs::DdfMethod::Residual))
        .unwrap();
    let mut wide = training.select(["a", "x"]).unwrap();
    wide.with_column(Column::new("unused".into(), vec![None::<f64>; 8]))
        .unwrap();
    let actual = fit
        .emmeans("a", &wide, 0.95, Some(lme_rs::DdfMethod::Residual))
        .unwrap();
    assert_eq!(actual.estimate, expected.estimate);
    assert_eq!(actual.std_error, expected.std_error);
    let grid = lme_rs::ReferenceGrid {
        terms: vec!["a".into()],
        at: [("unused".into(), 2.)].into(),
        ..Default::default()
    };
    assert!(fit.emmeans_with_grid(&wide, &grid, 0.95, None).is_err());
}

#[test]
fn prediction_rejects_unrepresentable_linear_mean() {
    let training = df!("x" => [0., 1., 2., 3.], "y" => [0.1, 2.2, 3.8, 6.1]).unwrap();
    let fit = lm_df("y ~ x", &training).unwrap();
    assert!(fit.predict(&df!("x" => [f64::MAX]).unwrap()).is_err());
    assert!(fit.predict(&df!("x" => [1e100]).unwrap()).unwrap()[0].is_finite());
}

#[test]
fn simulation_rejects_invalid_distribution_means() {
    // Public fit state can be restored/edited; distribution validation must not
    // silently turn negative means into tiny positive Poisson intensities.
    let mut fit = lm_df("y ~ x", &data()).unwrap();
    for (family, invalid) in [
        (Family::Poisson, -1.),
        (Family::Poisson, f64::NAN),
        (Family::Binomial, -0.1),
        (Family::Binomial, 1.1),
        (Family::Gamma, 0.),
        (Family::Gamma, f64::NAN),
        (Family::Gaussian, f64::INFINITY),
    ] {
        fit.family = Some(family);
        fit.fitted.fill(0.5);
        fit.fitted[0] = invalid;
        for workers in [1, 2] {
            assert!(
                fit.simulate_with(2, Some(workers), Some(17)).is_err(),
                "{family:?}: {invalid}"
            );
        }
    }
}

#[test]
fn simulation_preserves_zero_intensity_and_probability_endpoints() {
    let mut fit = lm_df("y ~ x", &data()).unwrap();
    fit.family = Some(Family::Poisson);
    fit.fitted.fill(0.);
    assert!(fit
        .simulate_with(3, Some(2), Some(17))
        .unwrap()
        .simulations
        .iter()
        .flatten()
        .all(|&v| v == 0.));
    fit.family = Some(Family::Binomial);
    fit.weights = Some(Array1::from_elem(8, 1e17));
    for p in [0., 1.] {
        fit.fitted.fill(p);
        assert!(fit
            .simulate_with(3, Some(2), Some(17))
            .unwrap()
            .simulations
            .iter()
            .flatten()
            .all(|&v| v == p));
    }
}

#[test]
fn binomial_trials_reject_saturating_integer_casts() {
    let df = data();
    let mut df = df.select(["x", "id"]).unwrap();
    df.with_column(Column::new("y".into(), [0., 1., 0., 1., 0., 1., 0., 1.]))
        .unwrap();
    let weights = Array1::from_elem(8, 2_f64.powi(64));
    assert!(lme_rs::prepare_glmer_weighted_with_link(
        "y ~ x + (1 | id)",
        &df,
        Family::Binomial,
        lme_rs::family::Link::Logit,
        1,
        Some(weights)
    )
    .is_err());
    // The largest representable f64 below 2^64 is still within u64's domain.
    for weight in [2_f64.powi(64) - 2048., 1. - 1e-10, 0.75] {
        assert!(lme_rs::prepare_glmer_weighted_with_link(
            "y ~ x + (1 | id)",
            &df,
            Family::Binomial,
            lme_rs::family::Link::Logit,
            1,
            Some(Array1::from_elem(8, weight))
        )
        .is_ok());
    }
    let mut fit = lm_df("y ~ x", &data()).unwrap();
    fit.family = Some(Family::Binomial);
    fit.fitted.fill(0.5);
    fit.weights = Some(Array1::from_elem(8, 2_f64.powi(64)));
    assert!(fit.simulate_with(1, Some(1), Some(17)).is_err());
    fit.family = Some(Family::Gaussian);
    assert!(fit.simulate_with(1, Some(1), Some(17)).is_ok());
}

fn boot(values: &[f64]) -> BootLmerResult {
    BootLmerResult {
        method: BootLmerMethod::Parametric,
        nsim: values.len(),
        fixed_names: vec!["x".into()],
        t0: array![0.],
        t0_theta: Some(array![1.]),
        t0_sigma2: Some(1.),
        replicates: values
            .iter()
            .enumerate()
            .map(|(index, &v)| BootReplicate {
                index,
                coefficients: array![v],
                theta: Some(array![1.]),
                sigma2: Some(1.),
                converged: true,
            })
            .collect(),
        prop_converged: 1.,
    }
}

#[test]
fn bootstrap_summary_rejects_nonfinite_draws() {
    assert!(boot(&[f64::NAN, 1.]).confint_percentile(0.95).is_err());
    let good = boot(&[-1., 1.]).confint_percentile(0.95).unwrap();
    assert!((good.lower[0] + 0.95).abs() < 1e-14);
    assert!((good.upper[0] - 0.95).abs() < 1e-14);
    let mut invalid = boot(&[-1., 1.]);
    invalid.t0[0] = f64::INFINITY;
    assert!(invalid.confint_percentile(0.95).is_err());
    for sigma2 in [-1., f64::NAN, f64::INFINITY] {
        let mut invalid = boot(&[-1., 1.]);
        invalid.t0_sigma2 = Some(sigma2);
        assert!(invalid.confint_percentile_vc(0.95).is_err());
        let mut invalid = boot(&[-1., 1.]);
        invalid.replicates[0].sigma2 = Some(sigma2);
        assert!(invalid.confint_percentile_vc(0.95).is_err());
    }
    let mut skipped = boot(&[f64::NAN, 1.]);
    skipped.replicates[0].converged = false;
    assert_eq!(skipped.confint_percentile(0.95).unwrap().lower[0], 1.);
    let mut zero = boot(&[-1., 1.]);
    zero.t0_sigma2 = Some(0.);
    assert_eq!(
        zero.confint_percentile_vc(0.95).unwrap().estimate,
        array![0., 0.]
    );
}

#[test]
fn grouped_cv_accepts_separate_split_columns() {
    let mut df = data();
    df.with_column(Column::new("unused".into(), [1, 2, 1, 2, 1, 2, 1, 2]))
        .unwrap();
    assert!(lme_rs::cv_grouped(
        "y ~ x + (1 | id)",
        &df,
        "unused",
        2,
        true,
        Some(17),
        Some(1)
    )
    .is_ok());
    assert!(lme_rs::cv_grouped("y ~ x + (1 | id)", &df, "id", 2, true, Some(17), Some(1)).is_ok());
    df.with_column(Column::new("y".into(), [1., 1., 2., 2., 3., 3., 4., 4.]))
        .unwrap();
    assert!(lme_rs::cv_grouped_glmer(
        "y ~ x + (1 | id)",
        &df,
        "unused",
        2,
        Family::Poisson,
        lme_rs::family::Link::Log,
        1,
        None,
        Some(17),
        Some(1)
    )
    .is_ok());
}

#[test]
fn natural_spline_prediction_extends_linearly_beyond_training_boundaries() {
    let training = df!("x" => [1., 2., 3., 4., 5., 6., 7., 8.],
        "y" => [5., 8., 11., 14., 17., 20., 23., 26.])
    .unwrap();
    let x = [-4., 0., 1., 8., 9., 12.];
    for formula in ["y ~ ns(x, 3)", "y ~ 0 + ns(x, 4, intercept = TRUE)"] {
        let fit = lm_df(formula, &training).unwrap();
        let predictions = fit.predict(&df!("x" => x).unwrap()).unwrap();
        // Every natural-spline space contains affine functions; its tails are
        // linear and must reproduce this line across and beyond both boundaries.
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                (predictions[i] - (2. + 3. * xi)).abs() < 1e-11,
                "{formula}, x={xi}, prediction={}",
                predictions[i]
            );
        }
    }
}

#[test]
fn cv_accepts_transformed_and_nested_split_sources() {
    let df = data();
    // x is a formula source even though the generated fixed term is log(x).
    assert!(lme_rs::cv_grouped(
        "y ~ log(x) + (1 | id)",
        &df,
        "x",
        2,
        true,
        Some(17),
        Some(1)
    )
    .is_ok());
    let mut nested = df.clone();
    nested
        .with_column(Column::new("site".into(), [1, 1, 1, 1, 2, 2, 2, 2]))
        .unwrap();
    assert!(lme_rs::cv_grouped(
        "y ~ x + (1 | site:id)",
        &nested,
        "id",
        2,
        true,
        Some(17),
        Some(1)
    )
    .is_ok());
}

#[test]
fn robust_inference_rejects_mismatched_training_rows() {
    let training = data();
    let mut fit = lm_df("y ~ x", &training).unwrap();
    let mut changed = training.clone();
    changed
        .with_column(Column::new("x".into(), [8., 7., 6., 5., 4., 3., 2., 1.]))
        .unwrap();
    assert!(fit.with_robust_se(&changed, None).is_err());
    assert!(fit.with_robust_se(&training, None).is_ok());
}

#[test]
fn robust_inference_accepts_training_bases_and_unused_columns() {
    let training = data().select(["y", "x", "a"]).unwrap();
    let mut wide = training.clone();
    wide.with_column(Column::new("unused".into(), vec![None::<f64>; 8]))
        .unwrap();
    for formula in ["y ~ x", "y ~ poly(x, 2)", "y ~ ns(x, 3)", "y ~ ."] {
        let mut fit = lm_df(formula, &training).unwrap();
        assert!(fit.with_robust_se(&wide, None).is_ok(), "{formula}");
    }
}

#[test]
fn log_inverse_link_preserves_representable_range() {
    let eta = array![-100., -31., 31., 100.];
    let got = LogLink.link_inv(&eta);
    let derivative = LogLink.mu_eta(&eta);
    for i in 0..eta.len() {
        let expected = eta[i].exp();
        assert!((got[i] / expected - 1.).abs() < 1e-14);
        assert!((derivative[i] / expected - 1.).abs() < 1e-14);
        assert!((LogLink.link_fun(&array![expected])[0] - eta[i]).abs() < 1e-13);
    }
}

#[test]
fn poisson_response_prediction_uses_exact_log_inverse_or_errors() {
    let mut training = data();
    training
        .with_column(Column::new("y".into(), [1., 1., 2., 2., 3., 3., 4., 5.]))
        .unwrap();
    let fit = lme_rs::glmer("y ~ x + (1 | id)", &training, Family::Poisson, 1).unwrap();
    let b0 = fit.coefficients[0];
    let b1 = fit.coefficients[1];
    assert!(b1 > 0.);
    let large = df!("x" => [(100. - b0) / b1]).unwrap();
    let eta = fit.predict(&large).unwrap()[0];
    assert!((fit.predict_response(&large).unwrap()[0] / eta.exp() - 1.).abs() < 1e-14);
    let overflow = df!("x" => [(1000. - b0) / b1]).unwrap();
    assert!(fit.predict(&overflow).unwrap()[0].is_finite());
    assert!(fit.predict_response(&overflow).is_err());
}

#[test]
fn cloglog_link_is_accurate_for_small_probabilities() {
    let p = array![1e-12, 1e-10, 1e-6, 0.1];
    let eta = CloglogLink.link_fun(&p);
    let roundtrip = CloglogLink.link_inv(&eta);
    // Independent 80-digit Decimal evaluation of ln(-ln(1-p)), rounded to f64.
    let expected = [
        -27.63102111592805,
        -23.025850929890456,
        -13.815510057964065,
        -2.2503673273124454,
    ];
    for i in 0..p.len() {
        assert!((eta[i] - expected[i]).abs() < 1e-13);
        assert!((roundtrip[i] / p[i] - 1.).abs() < 1e-13);
    }
    assert!(CloglogLink.mu_eta(&array![1000.])[0].is_finite());
}
