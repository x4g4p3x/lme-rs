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
