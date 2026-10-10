//! Regression coverage for invalid likelihood comparisons and uncertainty edges.
use lme_rs::{anova, lm_df, lmer, McpAdjust, McpType, SatterthwaiteResult};
use ndarray::array;
use polars::prelude::*;

fn sleepstudy() -> DataFrame {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
        .unwrap()
        .finish()
        .unwrap()
}

fn factor_data() -> DataFrame {
    df!("y" => [1.0, 2.0, 4.0, 3.0, 5.0, 6.0],
        "group" => ["a", "a", "a", "b", "b", "b"])
    .unwrap()
}

#[test]
fn lrt_rejects_different_reml_fixed_effects() {
    let df = sleepstudy();
    let null = lmer("Reaction ~ 1 + (1 | Subject)", &df, true).unwrap();
    let alternative = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    for (a, b) in [(&null, &alternative), (&alternative, &null)] {
        let error = anova(a, b).expect_err("different REML fixed effects are not comparable");
        assert!(error.to_string().contains("ML"));
    }
}

#[test]
fn lrt_rejects_mixed_estimation_methods() {
    let df = sleepstudy();
    let a = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let b = lmer("Reaction ~ Days + (Days | Subject)", &df, false).unwrap();
    assert!(anova(&a, &b).is_err());
    assert!(anova(&b, &a).is_err());
}

#[test]
fn lrt_checks_reml_design_values_not_only_coefficient_counts() {
    let mut df = sleepstudy();
    let scaled: Vec<f64> = df
        .column("Days")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|x| 2.0 * x)
        .collect();
    df.with_column(Column::new("ScaledDays".into(), scaled))
        .unwrap();
    let a = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let b = lmer("Reaction ~ ScaledDays + (Days | Subject)", &df, true).unwrap();
    assert_eq!(a.coefficients.len(), b.coefficients.len());
    assert!(anova(&a, &b).is_err());
}

#[test]
fn lrt_allows_same_design_reml_random_effect_comparisons() {
    let df = sleepstudy();
    let a = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let b = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    let result = anova(&a, &b).unwrap();
    assert_eq!(result.df, 2);
    assert!(result.chi_sq > 0.0 && result.p_value.is_finite());
}

#[test]
fn dispersion_glmm_information_criteria_count_estimated_scale() {
    use lme_rs::family::{Family, Link};
    let data = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/dyestuff.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    for family in [Family::Gamma, Family::Gaussian] {
        let fit = lme_rs::glmer_with_link("Yield ~ 1 + (1 | Batch)", &data, family, Link::Log, 1)
            .unwrap();
        assert!(
            fit.sigma2.is_some(),
            "{family:?} estimates a dispersion parameter"
        );
        // One intercept, one random-effect SD, and one estimated dispersion.
        let k = 3.;
        let deviance = -2. * fit.log_likelihood.unwrap();
        assert!(
            (fit.aic.unwrap() - (deviance + 2. * k)).abs() < 1e-10,
            "{family:?}: AIC must include estimated dispersion"
        );
        assert!(
            (fit.bic.unwrap() - (deviance + k * (fit.num_obs as f64).ln())).abs() < 1e-10,
            "{family:?}: BIC must include estimated dispersion"
        );
    }
}

#[test]
fn gaussian_glmm_lrt_counts_match_equivalent_ml_lmm() {
    use lme_rs::family::Family;
    let data = sleepstudy();
    let formulas = [
        "Reaction ~ Days + (1 | Subject)",
        "Reaction ~ Days + (Days | Subject)",
    ];
    let lmm: Vec<_> = formulas
        .iter()
        .map(|formula| lmer(formula, &data, false).unwrap())
        .collect();
    let glmm: Vec<_> = formulas
        .iter()
        .map(|formula| lme_rs::glmer(formula, &data, Family::Gaussian, 1).unwrap())
        .collect();
    let expected = anova(&lmm[0], &lmm[1]).unwrap();
    let actual = anova(&glmm[0], &glmm[1]).unwrap();
    assert_eq!(expected.n_params_0, 4);
    assert_eq!(expected.n_params_1, 6);
    assert_eq!(actual.n_params_0, expected.n_params_0);
    assert_eq!(actual.n_params_1, expected.n_params_1);
    assert_eq!(actual.df, expected.df);
    assert_eq!(actual.chi_sq, expected.chi_sq);
    assert_eq!(actual.p_value, expected.p_value);
}

#[test]
fn fixed_dispersion_glmm_penalties_do_not_count_an_extra_scale() {
    use lme_rs::family::Family;
    let mut data = factor_data();
    for family in [Family::Poisson, Family::Binomial] {
        if family == Family::Binomial {
            data.with_column(Column::new("y".into(), [0., 1., 0., 1., 1., 0.]))
                .unwrap();
        }
        let fit = lme_rs::glmer("y ~ 1 + (1 | group)", &data, family, 1).unwrap();
        assert!(fit.sigma2.is_none());
        // The model estimates an intercept and random-effect SD; the family
        // dispersion is fixed, so there are exactly two free parameters.
        let deviance = -2. * fit.log_likelihood.unwrap();
        assert!((fit.aic.unwrap() - (deviance + 4.)).abs() < 1e-10);
        assert!((fit.bic.unwrap() - (deviance + 2. * (fit.num_obs as f64).ln())).abs() < 1e-10);
    }
}

#[test]
fn lrt_rejects_nonfinite_deviance() {
    let df = sleepstudy();
    let mut a = lmer("Reaction ~ 1 + (1 | Subject)", &df, false).unwrap();
    let b = lmer("Reaction ~ Days + (1 | Subject)", &df, false).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        a.deviance = Some(invalid);
        assert!(
            anova(&a, &b).is_err(),
            "deviance {invalid} must not yield a p-value"
        );
        assert!(anova(&b, &a).is_err());
    }
}

#[test]
fn confidence_intervals_stay_finite_at_largest_valid_level() {
    let df = factor_data();
    let mut fit = lm_df("y ~ group", &df).unwrap();
    let level = f64::from_bits(1.0_f64.to_bits() - 1);
    for with_df in [false, true] {
        if with_df {
            fit.satterthwaite = Some(SatterthwaiteResult::univariate(
                array![20.0, 20.0],
                array![0.5, 0.5],
            ));
        }
        let ci = fit.confint(level).unwrap();
        let ordinary = fit.confint(0.95).unwrap();
        for i in 0..ci.lower.len() {
            assert!(ci.lower[i].is_finite() && ci.upper[i].is_finite(), "{ci:?}");
            assert!(ci.lower[i] < ordinary.lower[i] && ci.upper[i] > ordinary.upper[i]);
        }
    }
    let means = fit.emmeans("group", &df, level, None).unwrap();
    assert!(means
        .lower
        .iter()
        .chain(&means.upper)
        .all(|v| v.is_finite()));
}

#[test]
fn confidence_intervals_reject_unavailable_df_and_respect_fractional_df() {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let df = factor_data();
    let mut fit = lm_df("y ~ group", &df).unwrap();
    for invalid in [f64::NAN, 0.0, -1.0, f64::NEG_INFINITY] {
        fit.satterthwaite = Some(SatterthwaiteResult::univariate(
            array![invalid, 20.0],
            array![f64::NAN, 0.5],
        ));
        assert!(
            fit.confint(0.95).is_err(),
            "invalid df {invalid} must not be replaced by one"
        );
    }
    fit.satterthwaite = Some(SatterthwaiteResult::univariate(
        array![0.5, 20.0],
        array![0.5, 0.5],
    ));
    let ci = fit.confint(0.95).unwrap();
    let expected = -StudentsT::new(0.0, 1.0, 0.5).unwrap().inverse_cdf(0.025);
    let critical = (ci.upper[0] - fit.coefficients[0]) / fit.beta_se.as_ref().unwrap()[0];
    assert!((critical / expected - 1.0).abs() < 1e-10);
}

#[test]
fn marginal_mean_intervals_stay_finite_for_all_ddf_methods() {
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/pastes.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true).unwrap();
    fit.with_satterthwaite(&df).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    let level = f64::from_bits(1.0_f64.to_bits() - 1);
    for method in [
        None,
        Some(lme_rs::anova::DdfMethod::Satterthwaite),
        Some(lme_rs::anova::DdfMethod::KenwardRoger),
    ] {
        let means = fit.emmeans("cask", &df, level, method).unwrap();
        let ordinary = fit.emmeans("cask", &df, 0.95, method).unwrap();
        for i in 0..means.estimate.len() {
            assert!(
                means.lower[i].is_finite() && means.upper[i].is_finite(),
                "{method:?}"
            );
            assert!(means.lower[i] < ordinary.lower[i] && means.upper[i] > ordinary.upper[i]);
        }
    }
}

#[test]
fn adjusted_comparisons_preserve_unavailable_uncertainty() {
    let df = factor_data();
    let mut fit = lm_df("y ~ group", &df).unwrap();
    fit.satterthwaite = Some(SatterthwaiteResult::univariate(
        array![20.0, f64::NAN],
        array![0.5, f64::NAN],
    ));
    for adjust in [
        McpAdjust::None,
        McpAdjust::Bonferroni,
        McpAdjust::Holm,
        McpAdjust::Tukey,
    ] {
        let ddf = Some(lme_rs::anova::DdfMethod::Satterthwaite);
        let mcp = fit.glht("group", McpType::Tukey, adjust, ddf).unwrap();
        assert!(mcp.p_value[0].is_nan());
        assert!(mcp.p_adjust[0].is_nan(), "{adjust:?}: {mcp:?}");
        let pairs = fit.emmeans_pairs("group", &df, adjust, ddf).unwrap();
        assert!(pairs.p_value[0].is_nan() && pairs.p_adjust[0].is_nan());
    }
}
