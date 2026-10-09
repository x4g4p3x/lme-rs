//! Post-fit and unit-invariance regressions from the October correctness hunt.

use lme_rs::{lm, lm_df, DdfMethod, McpAdjust, ReferenceGrid};
use ndarray::array;
use polars::prelude::*;

fn data() -> DataFrame {
    df!(
        "y" => [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1],
        "a" => ["A", "B", "A", "B", "A", "B", "A", "B"],
        "x" => [1., 2., 3., 4., 5., 6., 7., 8.]
    )
    .unwrap()
}

#[test]
fn marginal_means_ignore_unused_numeric_columns_and_response() {
    let data = data();
    let fit = lm_df("y ~ a + x", &data).unwrap();
    let expected = fit
        .emmeans("a", &data, 0.95, Some(DdfMethod::Residual))
        .unwrap();
    let expected_pairs = fit
        .emmeans_pairs("a", &data, McpAdjust::Holm, Some(DdfMethod::Residual))
        .unwrap();
    for name in ["unused", "y"] {
        let mut reference = data.clone();
        reference
            .with_column(Column::new(name.into(), vec![None::<f64>; data.height()]))
            .unwrap();
        let means = fit
            .emmeans("a", &reference, 0.95, Some(DdfMethod::Residual))
            .unwrap();
        assert_eq!(means.estimate, expected.estimate);
        assert_eq!(means.std_error, expected.std_error);
        let pairs = fit
            .emmeans_pairs("a", &reference, McpAdjust::Holm, Some(DdfMethod::Residual))
            .unwrap();
        assert_eq!(pairs.estimate, expected_pairs.estimate);
        assert_eq!(pairs.p_adjust, expected_pairs.p_adjust);
    }
}

#[test]
fn marginal_means_reject_reference_values_for_random_only_covariates() {
    let mut data = data();
    data.with_column(Column::new("id".into(), [1_i32, 1, 2, 2, 3, 3, 4, 4]))
        .unwrap();
    let fit = lme_rs::lmer("y ~ a + (1 | id)", &data, true).unwrap();
    let grid = ReferenceGrid {
        terms: vec!["a".into()],
        at: [("id".into(), 100.)].into(),
        ..Default::default()
    };
    assert!(
        fit.emmeans_with_grid(&data, &grid, 0.95, None).is_err(),
        "a grouping variable must not be accepted as a fixed numeric reference value"
    );
    let expected = fit.emmeans("a", &data, 0.95, None).unwrap();
    data.with_column(Column::new("id".into(), vec![None::<i32>; data.height()]))
        .unwrap();
    let means = fit.emmeans("a", &data, 0.95, None).unwrap();
    assert_eq!(means.estimate, expected.estimate);
}

#[test]
fn marginal_mean_references_follow_transformed_interaction_and_dot_sources() {
    let data = data();
    for formula in [
        "y ~ a + log(x)",
        "y ~ a + I(x^2)",
        "y ~ a + poly(x, 2)",
        "y ~ a:x",
        "y ~ .",
    ] {
        let fit = lm_df(formula, &data).unwrap();
        let grid = ReferenceGrid {
            terms: vec!["a".into()],
            at: [("x".into(), 2.)].into(),
            ..Default::default()
        };
        let means = fit
            .emmeans_with_grid(&data, &grid, 0.95, Some(DdfMethod::Residual))
            .unwrap();
        let reference = df!("a" => ["A", "B"], "x" => [2., 2.]).unwrap();
        let expected = fit.predict(&reference).unwrap();
        assert!(
            (&means.estimate - &expected)
                .iter()
                .all(|v| v.abs() < 1e-12),
            "{formula}"
        );
        for name in ["unused", "y"] {
            let mut invalid = grid.clone();
            invalid.at = [(name.into(), 2.)].into();
            assert!(fit.emmeans_with_grid(&data, &invalid, 0.95, None).is_err());
        }
    }
}

#[test]
fn ols_is_invariant_to_predictor_units() {
    let y = array![1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1];
    let x = array![
        [1., 1.],
        [1., 2.],
        [1., 3.],
        [1., 4.],
        [1., 5.],
        [1., 6.],
        [1., 7.],
        [1., 8.]
    ];
    let expected = lm(&y, &x).unwrap();
    let expected_test = expected
        .test_contrast(&array![[0., 1.]], DdfMethod::Residual)
        .unwrap();
    for scale in [1e-100, 1e-20, 1e20, 1e100] {
        let mut rescaled = x.clone();
        rescaled.column_mut(1).mapv_inplace(|v| v * scale);
        let fit = lm(&y, &rescaled).unwrap();
        assert!((fit.coefficients[1] * scale - expected.coefficients[1]).abs() < 1e-12);
        assert!((&fit.fitted - &expected.fitted)
            .iter()
            .all(|v| v.abs() < 1e-12));
        assert!((fit.sigma2.unwrap() - expected.sigma2.unwrap()).abs() < 1e-12);
        let test = fit
            .test_contrast(&array![[0., 1.]], DdfMethod::Residual)
            .unwrap();
        assert!((test.f_value - expected_test.f_value).abs() < 1e-10);
        assert!((test.p_value - expected_test.p_value).abs() < 1e-12);
    }
}

#[test]
fn ols_still_rejects_collinear_predictors_in_different_units() {
    let y = array![1., 2., 3., 4.];
    for scale in [1e-100, 1., 1e100] {
        let x = array![
            [1., scale],
            [2., 2. * scale],
            [3., 3. * scale],
            [4., 4. * scale]
        ];
        assert!(lm(&y, &x).is_err());
    }
}

#[test]
fn ols_rejects_unrepresentable_covariance_instead_of_returning_nonfinite_inference() {
    let y = array![1., 2.1, 2.8, 4.2];
    for scale in [1e-200, 1e200] {
        let x = array![
            [1., scale],
            [1., 2. * scale],
            [1., 3. * scale],
            [1., 4. * scale]
        ];
        assert!(
            lm(&y, &x).is_err(),
            "unrepresentable covariance at scale {scale}"
        );
    }
}

#[test]
fn sequential_anova_is_invariant_to_predictor_units() {
    let data = data();
    let baseline = lm_df("y ~ x + a", &data).unwrap();
    let expected = baseline
        .anova_typed(lme_rs::AnovaType::Type1, DdfMethod::Residual)
        .unwrap();
    // Independently verify the first sequential SS by comparing nested fits.
    let intercept = lm_df("y ~ 1", &data).unwrap();
    let first = lm_df("y ~ x", &data).unwrap();
    let first_ss =
        intercept.residuals.dot(&intercept.residuals) - first.residuals.dot(&first.residuals);
    assert!((expected.sum_sq.as_ref().unwrap()[0] - first_ss).abs() < 1e-10);
    for scale in [1e-100, 1e-20, 1e20, 1e100] {
        let mut rescaled = data.clone();
        let x: Vec<_> = data
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .map(|v| v * scale)
            .collect();
        rescaled.with_column(Column::new("x".into(), x)).unwrap();
        let result = lm_df("y ~ x + a", &rescaled)
            .unwrap()
            .anova_typed(lme_rs::AnovaType::Type1, DdfMethod::Residual)
            .unwrap();
        assert!(
            (&result.f_value - &expected.f_value)
                .iter()
                .all(|v| v.abs() < 1e-9),
            "sequential F statistics changed at scale {scale}: {:?} vs {:?}",
            result.f_value,
            expected.f_value
        );
    }
}
