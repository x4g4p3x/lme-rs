//! Regression tests for contrast representation and simulation edge cases.

use lme_rs::anova::DdfMethod;
use lme_rs::{lm_df, lmer};
use ndarray::{array, Array2};
use polars::prelude::*;

fn sleepstudy() -> DataFrame {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
        .unwrap()
        .finish()
        .unwrap()
}

fn inference_fit() -> lme_rs::LmeFit {
    let df = sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    fit.with_satterthwaite(&df).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    fit
}

#[test]
fn contrast_scaling_preserves_inference() {
    let fit = inference_fit();
    for method in [DdfMethod::Satterthwaite, DdfMethod::KenwardRoger] {
        let reference = fit.test_contrast(&array![[0.3, 1.0]], method).unwrap();
        for scale in [1e-100, 1e-9, -1e9, 1e100] {
            let got = fit
                .test_contrast(&(array![[0.3, 1.0]] * scale), method)
                .unwrap();
            assert_eq!(got.num_df, reference.num_df, "{method:?}, scale {scale}");
            assert!(
                (got.den_df - reference.den_df).abs() < 1e-6,
                "{method:?}, scale {scale}: {got:?} vs {reference:?}"
            );
            assert!((got.f_value / reference.f_value - 1.0).abs() < 1e-10);
            assert!((got.p_value - reference.p_value).abs() < 1e-10);
        }
    }
}

#[test]
fn redundant_kr_contrasts_test_the_same_hypothesis() {
    let fit = inference_fit();
    let method = DdfMethod::KenwardRoger;
    let reference = fit.test_contrast(&array![[0.3, 1.0]], method).unwrap();
    let got = fit
        .test_contrast(&array![[0.3, 1.0], [0.6, 2.0], [0.0, 0.0]], method)
        .unwrap();
    assert_eq!(got.num_df, 1.0);
    assert!((got.den_df - reference.den_df).abs() < 1e-8);
    assert!((got.f_value / reference.f_value - 1.0).abs() < 1e-10);
}

#[test]
fn zero_rank_contrasts_return_errors() {
    let fit = inference_fit();
    for method in [DdfMethod::Satterthwaite, DdfMethod::KenwardRoger] {
        for rows in [1, 2] {
            assert!(fit
                .test_contrast(&Array2::zeros((rows, 2)), method)
                .is_err());
        }
    }
}

#[test]
fn satterthwaite_retains_small_upper_tail_probabilities() {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let fit = inference_fit();
    for l in [array![[1.0, 0.0]], array![[0.3, 1.0]]] {
        let result = fit.test_contrast(&l, DdfMethod::Satterthwaite).unwrap();
        let expected = 2.0
            * StudentsT::new(0.0, 1.0, result.den_df)
                .unwrap()
                .sf(result.f_value.sqrt());
        assert!(expected > 0.0 && expected < 1e-14);
        assert!(
            (result.p_value / expected - 1.0).abs() < 1e-8,
            "Satterthwaite upper tail rounded away: {result:?}"
        );
    }
}

#[test]
fn kr_joint_test_is_invariant_to_redundancy_and_row_units() {
    let fit = inference_fit();
    let method = DdfMethod::KenwardRoger;
    let a = fit.test_contrast(&Array2::eye(2), method).unwrap();
    let b = fit
        .test_contrast_vs(
            &array![[1e-100, 0.0], [0.0, -1e100], [2.0, 3.0]],
            &array![0.0, 0.0],
            method,
        )
        .unwrap();
    assert_eq!(b.num_df, 2.0);
    assert!((a.f_value / b.f_value - 1.0).abs() < 1e-10);
    assert!((a.den_df - b.den_df).abs() < 1e-8);
    assert!((a.p_value / b.p_value - 1.0).abs() < 1e-8);
}

#[test]
fn saturated_ols_cannot_simulate_unestimated_noise() {
    let df = df!("y" => [2.0, 5.0], "x" => [0.0, 1.0]).unwrap();
    let fit = lm_df("y ~ x", &df).unwrap();
    for jobs in [1, 2] {
        assert!(fit.simulate_with(2, Some(jobs), Some(7)).is_err());
    }
}

#[test]
fn simulation_rejects_invalid_variance_but_allows_zero_gaussian_noise() {
    let mut fit = lm_df("Reaction ~ Days", &sleepstudy()).unwrap();
    for variance in [f64::NAN, f64::INFINITY, -1.0] {
        fit.sigma2 = Some(variance);
        for jobs in [1, 2] {
            assert!(fit.simulate_with(2, Some(jobs), Some(7)).is_err());
        }
    }
    fit.sigma2 = Some(0.0);
    for jobs in [1, 2] {
        let draws = fit.simulate_with(2, Some(jobs), Some(7)).unwrap();
        assert!(draws.simulations.iter().all(|draw| draw == &fit.fitted));
    }
}

#[test]
fn simulation_range_rejects_index_overflow_without_panicking() {
    let fit = lm_df("Reaction ~ Days", &sleepstudy()).unwrap();
    for jobs in [1, 2] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lme_rs::simulate::simulate_range(&fit, usize::MAX, 2, Some(jobs), Some(7))
        }))
        .expect("invalid index range must not panic");
        assert!(result.is_err());
    }
    let last = lme_rs::simulate::simulate_range(&fit, usize::MAX, 1, Some(1), Some(7)).unwrap();
    assert_eq!(last.len(), 1, "the final representable index is valid");
}

#[test]
fn gamma_simulation_respects_observation_precision() {
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("tests/data/dyestuff.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let mut a = lme_rs::glmer(
        "Yield ~ 1 + (1 | Batch)",
        &df,
        lme_rs::family::Family::Gamma,
        1,
    )
    .unwrap();
    a.weights = Some(ndarray::Array1::from_iter(
        (0..df.height()).map(|i| 1.0 + (i % 4) as f64),
    ));
    let mut b = a.clone();
    b.weights = b.weights.map(|w| w * 4.0);
    b.sigma2 = b.sigma2.map(|phi| phi * 4.0);
    for jobs in [1, 2] {
        let x = a.simulate_with(4, Some(jobs), Some(42)).unwrap();
        let y = b.simulate_with(4, Some(jobs), Some(42)).unwrap();
        assert_eq!(
            x.simulations, y.simulations,
            "Gamma variance depends on dispersion divided by observation precision"
        );
    }
}
