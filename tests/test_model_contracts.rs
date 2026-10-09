//! Combinations of model features must retain their semantics after fitting.
use lme_rs::{
    boot_lmer, fit_prepared_with_response, lm_df, lmer, lmer_weighted, prepare_lmer_weighted,
    BootLmerMethod,
};
use ndarray::Array1;
use polars::prelude::*;

fn data() -> DataFrame {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
        .unwrap()
        .finish()
        .unwrap()
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < tol, "{x} != {y}");
    }
}

#[test]
fn offsets_survive_fit_prediction_and_response_replacement() {
    let mut df = data();
    let days = df.column("Days").unwrap().cast(&DataType::Float64).unwrap();
    let offset: Vec<f64> = days
        .f64()
        .unwrap()
        .into_no_null_iter()
        .map(|v| v * 10.0)
        .collect();
    df.with_column(Series::new("off".into(), &offset)).unwrap();
    for formula in [
        "Reaction ~ Days + offset(off)",
        "Reaction ~ Days + offset(off) + (1 | Subject)",
    ] {
        let fit = if formula.contains('|') {
            lmer(formula, &df, true).unwrap()
        } else {
            lm_df(formula, &df).unwrap()
        };
        let pred = if formula.contains('|') {
            fit.predict_conditional(&df, false).unwrap()
        } else {
            fit.predict(&df).unwrap()
        };
        close(
            pred.as_slice().unwrap(),
            fit.fitted.as_slice().unwrap(),
            1e-8,
        );
        let mut newdata = df.clone();
        newdata
            .with_column(Series::new(
                "off".into(),
                offset.iter().map(|x| x + 7.0).collect::<Vec<_>>(),
            ))
            .unwrap();
        let old = fit.predict(&df).unwrap();
        let new = fit.predict(&newdata).unwrap();
        close(
            (&new - &old).as_slice().unwrap(),
            &vec![7.0; df.height()],
            1e-8,
        );
    }
    let formula = "Reaction ~ Days + offset(off) + (1 | Subject)";
    let weights = Array1::from_iter((0..df.height()).map(|i| if i % 3 == 0 { 20.0 } else { 1.0 }));
    let prepared = prepare_lmer_weighted(formula, &df, Some(weights.clone())).unwrap();
    let fit = lmer_weighted(formula, &df, true, Some(weights.clone())).unwrap();
    let y = fit
        .simulate_with(1, Some(1), Some(42))
        .unwrap()
        .simulations
        .remove(0);
    let hot = fit_prepared_with_response(&prepared, Some(y.clone()), true).unwrap();
    df.with_column(Series::new("Reaction".into(), y.to_vec()))
        .unwrap();
    let cold = lmer_weighted(formula, &df, true, Some(weights)).unwrap();
    close(
        hot.coefficients.as_slice().unwrap(),
        cold.coefficients.as_slice().unwrap(),
        1e-7,
    );
    close(
        hot.predict_conditional(&df, false)
            .unwrap()
            .as_slice()
            .unwrap(),
        hot.fitted.as_slice().unwrap(),
        1e-7,
    );
}

#[test]
fn weighted_bootstrap_matches_explicit_refit_and_parallel_seeds() {
    let df = data();
    let weights = Array1::from_iter((0..df.height()).map(|i| if i % 3 == 0 { 20.0 } else { 1.0 }));
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lmer_weighted(formula, &df, true, Some(weights.clone())).unwrap();
    let simulated = fit
        .simulate_with(1, Some(1), Some(42))
        .unwrap()
        .simulations
        .remove(0);
    let mut sample = df.clone();
    sample
        .with_column(Series::new("Reaction".into(), simulated.to_vec()))
        .unwrap();
    let expected = lmer_weighted(formula, &sample, true, Some(weights)).unwrap();
    for method in [BootLmerMethod::Parametric, BootLmerMethod::Residual] {
        let serial = boot_lmer(formula, &df, &fit, 4, method, true, Some(42), Some(1)).unwrap();
        let parallel = boot_lmer(formula, &df, &fit, 4, method, true, Some(42), Some(2)).unwrap();
        if method == BootLmerMethod::Parametric {
            close(
                serial.replicates[0].coefficients.as_slice().unwrap(),
                expected.coefficients.as_slice().unwrap(),
                1e-8,
            );
        }
        for (a, b) in serial.replicates.iter().zip(&parallel.replicates) {
            close(
                a.coefficients.as_slice().unwrap(),
                b.coefficients.as_slice().unwrap(),
                1e-10,
            );
        }
    }
}

#[test]
fn gaussian_simulation_uses_observation_precision() {
    let df = data();
    let base = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let mut weighted = base.clone();
    weighted.weights = Some(Array1::from_elem(df.height(), 4.0));
    let a = base.simulate_with(3, Some(1), Some(11)).unwrap();
    let b = weighted.simulate_with(3, Some(2), Some(11)).unwrap();
    for (a, b) in a.simulations.iter().zip(b.simulations.iter()) {
        close(
            (a - &base.fitted).as_slice().unwrap(),
            ((b - &base.fitted) * 2.0).as_slice().unwrap(),
            1e-10,
        );
    }
}

#[test]
fn nested_groups_predict_without_synthetic_columns() {
    let mut df = data();
    df.with_column(Series::new(
        "site".into(),
        (0..df.height())
            .map(|i| if i < 90 { "a" } else { "b" })
            .collect::<Vec<_>>(),
    ))
    .unwrap();
    let fit = lmer("Reaction ~ Days + (1 | site/Subject)", &df, true).unwrap();
    close(
        fit.predict_conditional(&df, false)
            .unwrap()
            .as_slice()
            .unwrap(),
        fit.fitted.as_slice().unwrap(),
        1e-8,
    );
}

#[test]
fn ols_offset_matches_adjusted_response_and_qr_covariance() {
    let df = df!("y"=>[3.,5.,8.,8.,12.,14.],"x"=>[0.,1.,2.,3.,4.,5.],"off"=>[2.,0.,1.,-1.,2.,1.])
        .unwrap();
    let adjusted = df!("y"=>[1.,5.,7.,9.,10.,13.],"x"=>[0.,1.,2.,3.,4.,5.]).unwrap();
    let fit = lm_df("y ~ x + offset(off)", &df).unwrap();
    let reference = lm_df("y ~ x", &adjusted).unwrap();
    close(
        fit.coefficients.as_slice().unwrap(),
        reference.coefficients.as_slice().unwrap(),
        1e-10,
    );
    close(
        fit.beta_se.unwrap().as_slice().unwrap(),
        reference.beta_se.unwrap().as_slice().unwrap(),
        1e-10,
    );
}

#[test]
fn workspace_reuses_design_without_stale_response_products() {
    let df = data();
    let y = Array1::from_iter(
        df.column("Reaction")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter(),
    );
    for formula in [
        "Reaction ~ Days + (1 | Subject)",
        "Reaction ~ Days + (Days | Subject)",
    ] {
        let weights = Array1::from_shape_fn(df.height(), |i| 1.0 + (i % 3) as f64);
        let prepared = prepare_lmer_weighted(formula, &df, Some(weights)).unwrap();
        let mut workspace = prepared.workspace();
        for shift in [3.0, -7.0, 0.0] {
            let response = &y + shift;
            let reused = workspace
                .fit_response(response.clone(), true, &Default::default())
                .unwrap();
            let cold = prepared
                .fit(Some(response), true, &Default::default())
                .unwrap();
            close(
                reused.coefficients.as_slice().unwrap(),
                cold.coefficients.as_slice().unwrap(),
                1e-6,
            );
            close(
                reused.theta.as_ref().unwrap().as_slice().unwrap(),
                cold.theta.as_ref().unwrap().as_slice().unwrap(),
                1e-6,
            );
        }
        assert!(workspace
            .fit_response(Array1::zeros(1), true, &Default::default())
            .is_err());
    }
}

#[test]
fn iteration_limits_are_visible_and_can_be_required() {
    let prepared = lme_rs::prepare_lmer("Reaction ~ Days + (Days | Subject)", &data()).unwrap();
    let mut control = lme_rs::FitControl {
        max_iterations: 1,
        ..Default::default()
    };
    let fit = prepared.fit(None, true, &control).unwrap();
    assert_eq!(fit.converged, Some(false));
    assert_eq!(
        fit.diagnostics.as_ref().unwrap().termination,
        lme_rs::TerminationReason::IterationLimit
    );
    assert!(fit.to_string().contains("did NOT converge"));
    assert!(fit.confint(0.95).is_err());
    control.require_convergence = true;
    assert!(matches!(
        prepared.fit(None, true, &control),
        Err(lme_rs::LmeError::NonConvergence { .. })
    ));
}

fn limited_inference_fit() -> (DataFrame, lme_rs::LmeFit) {
    let mut df = data();
    df.with_column(Series::new(
        "phase".into(),
        (0..df.height())
            .map(|i| if i % 10 < 5 { "early" } else { "late" })
            .collect::<Vec<_>>(),
    ))
    .unwrap();
    let prepared = lme_rs::prepare_lmer("Reaction ~ Days + phase + (Days | Subject)", &df).unwrap();
    let fit = prepared
        .fit(
            None,
            true,
            &lme_rs::FitControl {
                max_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(fit.converged, Some(false));
    (df, fit)
}

fn assert_inference_nonconvergence<T>(result: lme_rs::Result<T>, context: &str) {
    match result {
        Err(lme_rs::LmeError::NonConvergence { .. }) => {}
        Err(error) => panic!("{context}: expected NonConvergence, got {error:?}"),
        Ok(_) => panic!("{context}: returned uncertainty for a nonconverged fit"),
    }
}

fn assert_anyhow_nonconvergence<T>(result: anyhow::Result<T>, context: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{context}: returned uncertainty for a nonconverged fit"));
    assert!(
        matches!(
            error.downcast_ref::<lme_rs::LmeError>(),
            Some(lme_rs::LmeError::NonConvergence { .. })
        ),
        "{context}: expected NonConvergence, got {error:?}"
    );
}

#[test]
fn nonconverged_fits_reject_asymptotic_factor_inference() {
    let (df, fit) = limited_inference_fit();
    // A diagnostic fit remains usable for prediction, while its uncertainty
    // must obey the same convergence contract as confint().
    assert!(fit.predict(&df).unwrap().iter().all(|x| x.is_finite()));
    let results = [
        (
            "marginal means",
            fit.emmeans("phase", &df, 0.95, None).map(|_| ()),
        ),
        (
            "marginal-mean pairs",
            fit.emmeans_pairs("phase", &df, lme_rs::McpAdjust::Holm, None)
                .map(|_| ()),
        ),
        (
            "multiple comparisons",
            fit.glht(
                "phase",
                lme_rs::McpType::Tukey,
                lme_rs::McpAdjust::Holm,
                None,
            )
            .map(|_| ()),
        ),
    ];
    for (context, result) in results {
        assert_inference_nonconvergence(result, context);
    }
}

#[test]
fn nonconverged_fits_reject_satterthwaite_calculation() {
    let (df, mut fit) = limited_inference_fit();
    let direct = lme_rs::satterthwaite::compute_satterthwaite(&fit, &df);
    let wrapped = fit.with_satterthwaite(&df).map(|_| ());
    assert_inference_nonconvergence(direct, "Satterthwaite calculation");
    assert_anyhow_nonconvergence(wrapped, "with_satterthwaite");
    assert!(fit.satterthwaite.is_none());
}

#[test]
fn nonconverged_fits_reject_kenward_roger_calculation() {
    let (df, mut fit) = limited_inference_fit();
    let direct = lme_rs::kenward_roger::compute_kenward_roger(&fit, &df);
    let wrapped = fit.with_kenward_roger(&df).map(|_| ());
    assert_inference_nonconvergence(direct, "Kenward-Roger calculation");
    assert_anyhow_nonconvergence(wrapped, "with_kenward_roger");
    assert!(fit.kenward_roger.is_none());
}

#[test]
fn nonconverged_fits_reject_robust_calculation() {
    let (df, mut fit) = limited_inference_fit();
    let direct = lme_rs::robust::compute_robust_se(&fit, &df, Some("Subject"));
    let wrapped = fit.with_robust_se(&df, Some("Subject")).map(|_| ());
    assert!(
        direct.err().is_some_and(|e| e.contains("did not converge")),
        "direct robust calculation must reject nonconvergence"
    );
    assert_anyhow_nonconvergence(wrapped, "with_robust_se");
    assert!(fit.robust.is_none());
}

#[test]
fn nonconverged_fits_cannot_use_cached_df_adjustments() {
    let (df, _) = limited_inference_fit();
    let mut fit = lmer("Reaction ~ Days + phase + (Days | Subject)", &df, true).unwrap();
    assert_eq!(fit.converged, Some(true));
    fit.with_satterthwaite(&df).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    // Cached results must never bypass the authoritative convergence state.
    fit.converged = Some(false);
    let mut one = ndarray::Array2::zeros((1, fit.coefficients.len()));
    one[[0, 1]] = 1.0;
    for method in [
        lme_rs::DdfMethod::Satterthwaite,
        lme_rs::DdfMethod::KenwardRoger,
    ] {
        let results = [
            (
                "single contrast",
                fit.test_contrast(&one, method).map(|_| ()),
            ),
            (
                "joint contrast",
                fit.test_contrast(&ndarray::Array2::eye(fit.coefficients.len()), method)
                    .map(|_| ()),
            ),
            ("ANOVA", fit.anova(method).map(|_| ())),
            (
                "linear hypothesis",
                fit.linear_hypothesis("phase", method).map(|_| ()),
            ),
            (
                "marginal means",
                fit.emmeans("phase", &df, 0.95, Some(method)).map(|_| ()),
            ),
            (
                "multiple comparisons",
                fit.glht(
                    "phase",
                    lme_rs::McpType::Tukey,
                    lme_rs::McpAdjust::Holm,
                    Some(method),
                )
                .map(|_| ()),
            ),
        ];
        for (context, result) in results {
            assert_inference_nonconvergence(result, context);
        }
    }
}

#[test]
fn nonconverged_fits_reject_likelihood_ratio_comparisons() {
    let df = data();
    let small = lmer("Reaction ~ Days + (1 | Subject)", &df, false).unwrap();
    let large = lmer("Reaction ~ Days + (Days | Subject)", &df, false).unwrap();
    assert!(lme_rs::anova(&small, &large).unwrap().p_value.is_finite());
    for (formula, other) in [
        ("Reaction ~ Days + (1 | Subject)", &large),
        ("Reaction ~ Days + (Days | Subject)", &small),
    ] {
        let limited = lme_rs::prepare_lmer(formula, &df)
            .unwrap()
            .fit(
                None,
                false,
                &lme_rs::FitControl {
                    max_iterations: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(limited.converged, Some(false));
        let forward = lme_rs::anova(&limited, other);
        let reverse = lme_rs::anova(other, &limited);
        assert_anyhow_nonconvergence(forward, "LRT model A");
        assert_anyhow_nonconvergence(reverse, "LRT model B");
    }
}

#[test]
fn nonconverged_fits_reject_profile_and_bootstrap_uncertainty() {
    let control = lme_rs::FitControl {
        max_iterations: 1,
        ..Default::default()
    };
    let df = data();
    let formula = "Reaction ~ Days + (1 | Subject)";
    let fit = lme_rs::prepare_lmer(formula, &df)
        .unwrap()
        .fit(None, false, &control)
        .unwrap();
    assert_eq!(fit.converged, Some(false));
    let fixed = fit.confint_profile_parms(0.5, &df, &[1]);
    let variance = fit.confint_profile_vc(0.5, &df);
    let bootstrap = boot_lmer(
        formula,
        &df,
        &fit,
        1,
        BootLmerMethod::Parametric,
        false,
        Some(17),
        Some(1),
    );
    assert_inference_nonconvergence(bootstrap, "LMM bootstrap");
    assert_anyhow_nonconvergence(fixed, "LMM fixed profile");
    assert_anyhow_nonconvergence(variance, "LMM variance profile");
}

#[test]
fn nonconverged_glmm_rejects_profile_and_bootstrap_uncertainty() {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("tests/data/cbpp_binary.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let fit = lme_rs::prepare_glmer(formula, &df, lme_rs::family::Family::Binomial, 1)
        .unwrap()
        .fit(
            None,
            &lme_rs::FitControl {
                max_iterations: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(fit.converged, Some(false));
    let fixed = fit.confint_profile_parms(0.5, &df, &[1]);
    let variance = fit.confint_profile_vc(0.5, &df);
    let bootstrap = lme_rs::boot_glmer(
        formula,
        &df,
        &fit,
        1,
        BootLmerMethod::Parametric,
        Some(17),
        Some(1),
    );
    assert_inference_nonconvergence(bootstrap, "GLMM bootstrap");
    assert_anyhow_nonconvergence(fixed, "GLMM fixed profile");
    assert_anyhow_nonconvergence(variance, "GLMM variance profile");
}

#[test]
fn converged_and_ols_fits_keep_factor_inference_available() {
    let (df, _) = limited_inference_fit();
    let mixed = lmer("Reaction ~ Days + phase + (Days | Subject)", &df, true).unwrap();
    assert_eq!(mixed.converged, Some(true));
    let ols = lm_df("Reaction ~ Days + phase", &df).unwrap();
    assert_eq!(ols.converged, None);
    for mut fit in [mixed, ols] {
        fit.with_robust_se(&df, Some("Subject")).unwrap();
        assert!(fit
            .confint(0.95)
            .unwrap()
            .lower
            .iter()
            .all(|x| x.is_finite()));
        assert!(fit
            .emmeans("phase", &df, 0.95, None)
            .unwrap()
            .std_error
            .iter()
            .all(|x| x.is_finite()));
        assert!(fit
            .emmeans_pairs("phase", &df, lme_rs::McpAdjust::Holm, None)
            .unwrap()
            .p_value
            .iter()
            .all(|x| x.is_finite()));
        assert!(fit
            .glht(
                "phase",
                lme_rs::McpType::Tukey,
                lme_rs::McpAdjust::Holm,
                None
            )
            .unwrap()
            .p_value
            .iter()
            .all(|x| x.is_finite()));
    }
}

#[test]
fn lmm_preparation_rejects_formulas_without_random_effects() {
    let df = df!("x" => [0., 1., 2., 3.], "y" => [0., 1., 0., 1.]).unwrap();
    for weighted in [false, true] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_lmer_weighted("y ~ x", &df, weighted.then(|| Array1::ones(df.height())))
        }))
        .expect("a formula without random effects must return an error, not panic");
        assert!(matches!(result, Err(lme_rs::LmeError::InvalidInput { .. })));
    }
    // The same fixed-only formula remains valid through the OLS API.
    let fit = lm_df("y ~ x", &df).unwrap();
    let profile = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit.confint_profile(0.95, &df)
    }))
    .expect("unsupported OLS profiling must return an error, not panic");
    assert!(profile.is_err());
}

#[test]
fn singular_random_slopes_recover_zero_variance_components() {
    // Identical groups have no between-group variation, while curvature leaves
    // positive residual variance after fitting the common linear trend.
    let x: Vec<f64> = (0..60).map(|i| (i % 5) as f64 - 2.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| 3.0 + 2.0 * v + 0.1 * v * v).collect();
    let group: Vec<String> = (0..60).map(|i| format!("g{}", i / 5)).collect();
    let df = df!("x" => x, "y" => y, "group" => group).unwrap();
    let fixed = lm_df("y ~ x", &df).unwrap();
    for reml in [false, true] {
        let fit = lmer("y ~ x + (1 + x | group)", &df, reml).unwrap();
        assert_eq!(fit.converged, Some(true));
        let theta = fit.theta.as_ref().unwrap();
        assert!(theta[0] >= 0.0 && theta[0] < 1e-3, "{theta:?}");
        assert!(theta[2] >= 0.0 && theta[2] < 1e-3, "{theta:?}");
        assert!(fit.diagnostics.as_ref().unwrap().objective.is_finite());
        close(
            fit.coefficients.as_slice().unwrap(),
            fixed.coefficients.as_slice().unwrap(),
            1e-6,
        );
    }
}

#[test]
fn execution_context_preserves_caller_pool_and_environment() {
    let before = std::env::var_os("MKL_NUM_THREADS");
    let context = lme_rs::execution::ExecutionContext::new(2).unwrap();
    context.install(|| assert_eq!(rayon::current_num_threads(), 2));
    assert_eq!(std::env::var_os("MKL_NUM_THREADS"), before);
}

#[test]
fn transformed_glmm_offset_survives_both_prediction_scales() {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("tests/data/grouseticks.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let formula = "TICKS ~ YEAR96 + YEAR97 + offset(log(HEIGHT)) + (1 | BROOD)";
    let fit = lme_rs::glmer(formula, &df, lme_rs::family::Family::Poisson, 1).unwrap();
    close(
        fit.predict_conditional_response(&df, false)
            .unwrap()
            .as_slice()
            .unwrap(),
        fit.fitted.as_slice().unwrap(),
        1e-7,
    );
    let heights = df
        .column("HEIGHT")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    let mut newdata = df.clone();
    newdata
        .with_column(Series::new(
            "HEIGHT".into(),
            heights
                .f64()
                .unwrap()
                .into_no_null_iter()
                .map(|v| v * 2.0)
                .collect::<Vec<_>>(),
        ))
        .unwrap();
    let linear = fit.predict(&df).unwrap();
    let shifted = fit.predict(&newdata).unwrap();
    close(
        (&shifted - &linear).as_slice().unwrap(),
        &vec![2.0f64.ln(); df.height()],
        1e-10,
    );
    let response = fit.predict_response(&df).unwrap();
    let changed = fit.predict_response(&newdata).unwrap();
    close(
        changed.as_slice().unwrap(),
        (&response * 2.0).as_slice().unwrap(),
        1e-8,
    );
}

#[test]
fn crossed_ml_refines_grid_before_claiming_convergence() {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("tests/data/penicillin.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let prepared = lme_rs::prepare_lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", &df).unwrap();
    let fit = prepared.fit(None, false, &Default::default()).unwrap();
    assert_eq!(fit.converged, Some(true));
    let refine = lme_rs::FitControl {
        start: fit.theta.clone(),
        tolerance: 1e-8,
        require_convergence: true,
        ..Default::default()
    };
    let check = prepared.fit(None, false, &refine).unwrap();
    assert!((fit.deviance.unwrap() - check.deviance.unwrap()).abs() < 1e-4);
}
