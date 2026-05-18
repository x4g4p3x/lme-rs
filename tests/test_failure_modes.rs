//! Convergence and failure-mode coverage: invalid inputs should surface as `Result::Err`
//! or (where documented) deterministic panics inside the sparse/cholesky path, not silent corruption.

use lme_rs::family::Family;
use lme_rs::formula::parse;
use lme_rs::glmer;
use lme_rs::glmm_math::GlmmData;
use lme_rs::lm_df;
use lme_rs::lmer;
use lme_rs::lmer_weighted;
use lme_rs::model_matrix::build_design_matrices;
use lme_rs::LmeError;
use ndarray::array;
use polars::prelude::*;

fn assert_notimplemented_contains(err: &LmeError, needle: &str) {
    match err {
        LmeError::NotImplemented { feature } => assert!(
            feature.contains(needle),
            "expected message containing {:?}, got {}",
            needle,
            feature
        ),
        e => panic!("expected NotImplemented, got {:?}", e),
    }
}

#[test]
fn lmer_weights_length_mismatch_is_rejected() {
    let df = df!(
        "y" => &[1.0_f64, 2.0, 3.0],
        "x" => &[1.0_f64, 2.0, 3.0],
        "g" => &["a", "b", "a"],
    )
    .unwrap();
    let w = ndarray::Array1::from_vec(vec![1.0, 1.0]);
    let err = lmer_weighted("y ~ x + (1 | g)", &df, true, Some(w)).unwrap_err();
    assert_notimplemented_contains(&err, "weights");
    assert_notimplemented_contains(&err, "length");
}

#[test]
fn lmer_weights_non_positive_is_rejected() {
    let df = df!(
        "y" => &[1.0_f64, 2.0],
        "x" => &[1.0_f64, 2.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let w = ndarray::Array1::from_vec(vec![1.0, 0.0]);
    let err = lmer_weighted("y ~ x + (1 | g)", &df, true, Some(w)).unwrap_err();
    assert_notimplemented_contains(&err, "strictly positive");
}

#[test]
fn lmer_weights_nan_is_rejected() {
    let df = df!(
        "y" => &[1.0_f64, 2.0],
        "x" => &[1.0_f64, 2.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let w = ndarray::Array1::from_vec(vec![1.0, f64::NAN]);
    let err = lmer_weighted("y ~ x + (1 | g)", &df, true, Some(w)).unwrap_err();
    assert_notimplemented_contains(&err, "non-finite");
}

#[test]
fn glmer_binomial_response_outside_unit_interval_is_rejected() {
    let df = df!(
        "y" => &[0.0_f64, 1.5],
        "x" => &[0.0_f64, 1.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let err = glmer("y ~ x + (1 | g)", &df, Family::Binomial, 1).unwrap_err();
    assert_notimplemented_contains(&err, "binomial");
}

#[test]
fn glmer_poisson_negative_response_is_rejected() {
    let df = df!(
        "y" => &[1.0_f64, -0.5],
        "x" => &[1.0_f64, 2.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let err = glmer("y ~ x + (1 | g)", &df, Family::Poisson, 1).unwrap_err();
    assert_notimplemented_contains(&err, "Poisson");
}

#[test]
fn glmer_gamma_non_positive_response_is_rejected() {
    let df = df!(
        "y" => &[2.0_f64, 0.0],
        "x" => &[1.0_f64, 2.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let err = glmer("y ~ x + (1 | g)", &df, Family::Gamma, 1).unwrap_err();
    assert_notimplemented_contains(&err, "Gamma");
}

#[test]
fn lm_df_rank_deficient_fixed_effects_returns_error() {
    let df = df!(
        "y" => &[1.0_f64, 2.0, 3.0],
        "x" => &[1.0_f64, 2.0, 3.0],
        "z" => &[1.0_f64, 2.0, 3.0],
    )
    .unwrap();
    let err = lm_df("y ~ x + z", &df).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("inversion") || msg.contains("lm failed") || msg.contains("Cholesky"),
        "unexpected lm_df error: {}",
        msg
    );
}

#[test]
fn lmer_rank_deficient_fixed_effects_currently_panics() {
    // `math::LmmData::evaluate` uses `.expect` on the downdated Cholesky; collinear fixed columns
    // are not turned into a structured error yet.
    let outcome = std::panic::catch_unwind(|| {
        let df = df!(
            "y" => &[1.0_f64, 2.0, 3.0],
            "x" => &[1.0_f64, 2.0, 3.0],
            "z" => &[1.0_f64, 2.0, 3.0],
            "subj" => &["s1", "s2", "s3"],
        )
        .unwrap();
        let _ = lmer("y ~ x + z + (1 | subj)", &df, true);
    });
    assert!(
        outcome.is_err(),
        "expected panic from singular downdated X'WX until a Result path exists"
    );
}

#[test]
fn lmer_tight_fixed_rhs_with_few_observations_is_fragile() {
    // Fewer effective rows than fixed parameters tends to hit the same Cholesky `.expect` path.
    let outcome = std::panic::catch_unwind(|| {
        let df = df!(
            "y" => &[1.0_f64, 2.0],
            "a" => &[1.0_f64, 0.0],
            "b" => &[0.0_f64, 1.0],
            "c" => &[0.5_f64, 0.5],
            "g" => &["g1", "g2"],
        )
        .unwrap();
        lmer("y ~ a + b + c + (1 | g)", &df, true)
    });
    match outcome {
        Err(_) => {}
        Ok(Err(_)) => {}
        Ok(Ok(_)) => {
            panic!("expected panic or Err for underdetermined / ill-conditioned fixed system")
        }
    }
}

#[test]
fn glmm_laplace_deviance_non_finite_when_response_invalid_after_matrix_build() {
    // `glmer()` validates responses before PIRLS; this documents lower-level behavior when `y`
    // is corrupted after construction (e.g. bad FFI): deviance must not look like a healthy fit.
    let df = df!(
        "y" => &[1.0_f64, 2.0],
        "x" => &[1.0_f64, 2.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let ast = parse("y ~ x + (1 | g)").unwrap();
    let matrices = build_design_matrices(&ast, &df).unwrap();
    let mut glmm = GlmmData::new(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        Family::Poisson.build(),
        1,
    );
    glmm.y = array![-1.0, 1.0];
    let dev = glmm.laplace_deviance(&[1.0], None, 1);
    assert!(
        !dev.is_finite() || dev >= 1e200 || dev == f64::MAX,
        "expected unusable deviance for invalid Poisson counts, got {}",
        dev
    );
}

#[test]
fn glmer_can_surface_pirls_failure_at_optimal_theta() {
    // Tiny separable-ish binomial setup: optimizer may pick a θ where the inner PIRLS Cholesky
    // path fails (returns `None`), which `glmer` maps to a NotImplemented error.
    let df = df!(
        "y" => &[1.0_f64, 1.0, 0.0, 0.0],
        "x" => &[0.0_f64, 1.0, 0.0, 1.0],
        "g" => &["a", "a", "b", "b"],
    )
    .unwrap();
    let res = glmer("y ~ x + (1 | g)", &df, Family::Binomial, 1);
    match res {
        Ok(fit) => {
            // If the fit succeeds, optimizer and PIRLS stayed well-conditioned for this seed.
            assert!(
                fit.theta.is_some(),
                "successful fit should still carry theta: {:?}",
                fit
            );
        }
        Err(LmeError::NotImplemented { feature }) => {
            assert!(
                feature.contains("PIRLS") || feature.contains("GLMM optimizer"),
                "unexpected failure mode: {}",
                feature
            );
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}
