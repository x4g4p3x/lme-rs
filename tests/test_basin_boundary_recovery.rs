//! Covariance-boundary regression against an independently evaluated lme4 fit.
#![cfg(feature = "basin")]

use lme_rs::{prepare_lmer, prepare_lmer_weighted, FitControl};
use ndarray::Array1;
use polars::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

fn boundary_data() -> DataFrame {
    // Same recipe as the production random-slope benchmark, with seed 106.
    // No group effects are generated, so the fitted covariance is near singular.
    let mut rng = StdRng::seed_from_u64(106);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (mut y, mut x, mut group) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..80_000 {
        let g = rng.random_range(0..3_000);
        let xi = normal.sample(&mut rng);
        y.push(1.0 + 1.25 * xi + normal.sample(&mut rng));
        x.push(xi);
        group.push(format!("G{g}"));
    }
    df!("y" => y, "x" => x, "group" => group).unwrap()
}

#[test]
fn random_slopes_escape_a_covariance_boundary_orientation() {
    let data = boundary_data();
    let prepared = prepare_lmer("y ~ x + (x | group)", &data).unwrap();
    let fit = prepared.fit(None, false, &FitControl::default()).unwrap();
    assert_eq!(fit.converged, Some(true));
    // R 4.6.1, lme4 2.0.1, ML; three independent starts agree to 4e-5.
    // The previous Basin fit stopped at 226271.4878426, 3.6641 too high.
    let expected = 226267.82374543123;
    assert!(
        (fit.deviance.unwrap() - expected).abs() < 1e-4,
        "objective {:?}, theta {:?}",
        fit.deviance,
        fit.theta
    );
    for (actual, expected) in fit
        .coefficients
        .iter()
        .zip([1.0027942700740697, 1.247_051_433_626_717])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert!((fit.diagnostics.as_ref().unwrap().objective - fit.deviance.unwrap()).abs() < 1e-6);
}

#[test]
fn covariance_recovery_matches_weighted_ml_and_reml_references() {
    let data = boundary_data();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/basin_boundary_lme4.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let reml = case["reml"].as_bool().unwrap();
        let weights = case["weighted"]
            .as_bool()
            .unwrap()
            .then(|| Array1::from_iter((0..data.height()).map(|i| 0.5 + (i % 7) as f64 / 3.0)));
        // The current lme-rs weighted criterion is on the whitened-data scale.
        // Remove its parameter-independent Jacobian constant to compare with
        // lme4's original-response likelihood; this does not change the optimum.
        let log_weight_determinant = weights
            .as_ref()
            .map_or(0.0, |w| w.iter().map(|v| v.ln()).sum());
        let prepared = prepare_lmer_weighted("y ~ x + (x | group)", &data, weights).unwrap();
        let fit = prepared.fit(None, reml, &FitControl::default()).unwrap();
        let objective = fit.diagnostics.as_ref().unwrap().objective - log_weight_determinant;
        assert_eq!(fit.converged, Some(true), "{case}");
        assert!(
            (objective - case["objective"].as_f64().unwrap()).abs() < 1e-4,
            "{case}: objective {objective}, theta {:?}",
            fit.theta
        );
        for (actual, expected) in fit
            .coefficients
            .iter()
            .zip(case["coefficients"].as_array().unwrap())
        {
            assert!((actual - expected.as_f64().unwrap()).abs() < 1e-6, "{case}");
        }
    }
}

#[test]
fn covariance_recovery_respects_a_shared_iteration_budget() {
    let data = boundary_data();
    let prepared = prepare_lmer("y ~ x + (x | group)", &data).unwrap();
    // The old simplex converged after 66 iterations. Recovery must not reset
    // the caller's remaining budget when it discovers the better direction.
    let control = FitControl {
        max_iterations: 67,
        ..FitControl::default()
    };
    let fit = prepared.fit(None, false, &control).unwrap();
    assert_eq!(fit.converged, Some(false));
    assert_eq!(fit.iterations, Some(67));
    assert!(fit.deviance.unwrap() < 226271.4878426);
    let strict = FitControl {
        require_convergence: true,
        ..control
    };
    assert!(prepared.fit(None, false, &strict).is_err());
}
