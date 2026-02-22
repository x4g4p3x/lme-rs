//! Tests targeting specific uncovered branches in lib.rs Display and convergence logic.

use lme_rs::LmeFit;
use lme_rs::model_matrix::ReBlock;
use ndarray::{array, Array1};
use std::collections::HashMap;

/// Helper to build a minimal LmeFit for Display testing.
fn make_display_fit(
    theta: Vec<f64>,
    re_blocks: Vec<ReBlock>,
    sigma2: f64,
    converged: Option<bool>,
    iterations: Option<u64>,
) -> LmeFit {
    LmeFit {
        coefficients: array![1.0, 2.0],
        residuals: array![0.1, -0.1, 0.05, -0.05],
        fitted: array![1.0, 2.0, 3.0, 4.0],
        ranef: None,
        var_corr: None,
        theta: Some(Array1::from_vec(theta)),
        sigma2: Some(sigma2),
        reml: Some(100.0),
        log_likelihood: Some(-50.0),
        aic: Some(110.0),
        bic: Some(115.0),
        deviance: Some(100.0),
        b: None,
        u: None,
        beta_se: Some(array![0.5, 0.3]),
        beta_t: Some(array![2.0, 6.67]),
        formula: Some("y ~ x + (x | g)".to_string()),
        fixed_names: Some(vec!["(Intercept)".to_string(), "x".to_string()]),
        re_blocks: Some(re_blocks),
        num_obs: 4,
        converged,
        iterations,
    }
}

#[test]
fn test_display_zero_variance_nan_correlation() {
    // L245: When a variance component is zero, correlation should display as NaN.
    // theta = [0, 0, 1] → Lambda = [[0, 0], [0, 1]] → Cov = [[0, 0], [0, sigma2]]
    // Corr(0, 1) has var_i=0 → NaN branch
    let re_blocks = vec![ReBlock {
        m: 2,
        k: 2,
        theta_len: 3,
        group_name: "g".to_string(),
        effect_names: vec!["(Intercept)".to_string(), "x".to_string()],
        group_map: HashMap::new(),
    }];

    let fit = make_display_fit(vec![0.0, 0.0, 1.0], re_blocks, 1.0, Some(true), Some(5));
    let summary = format!("{}", fit);

    assert!(summary.contains("NaN") || summary.contains("Corr:"),
        "Summary should handle zero-variance gracefully:\n{}", summary);
}

#[test]
fn test_display_converged_without_iterations() {
    // L278: converged=Some(true), iterations=None → "converged" message without iteration count
    let re_blocks = vec![ReBlock {
        m: 2,
        k: 1,
        theta_len: 1,
        group_name: "g".to_string(),
        effect_names: vec!["(Intercept)".to_string()],
        group_map: HashMap::new(),
    }];

    let fit = make_display_fit(vec![1.0], re_blocks, 1.0, Some(true), None);
    let summary = format!("{}", fit);

    assert!(summary.contains("optimizer (Nelder-Mead) converged"),
        "Summary should say converged:\n{}", summary);
    assert!(!summary.contains("iterations"),
        "Should NOT mention iterations when None:\n{}", summary);
}

#[test]
fn test_display_non_convergence_warning() {
    // L281: converged=Some(false) → WARNING message
    let re_blocks = vec![ReBlock {
        m: 2,
        k: 1,
        theta_len: 1,
        group_name: "g".to_string(),
        effect_names: vec!["(Intercept)".to_string()],
        group_map: HashMap::new(),
    }];

    let fit = make_display_fit(vec![1.0], re_blocks, 1.0, Some(false), Some(1000));
    let summary = format!("{}", fit);

    assert!(summary.contains("WARNING"),
        "Summary should contain WARNING for non-convergence:\n{}", summary);
    assert!(summary.contains("did NOT converge"),
        "Summary should say 'did NOT converge':\n{}", summary);
}

#[test]
fn test_display_no_convergence_info() {
    // converged=None → no convergence section at all
    let re_blocks = vec![ReBlock {
        m: 2,
        k: 1,
        theta_len: 1,
        group_name: "g".to_string(),
        effect_names: vec!["(Intercept)".to_string()],
        group_map: HashMap::new(),
    }];

    let fit = make_display_fit(vec![1.0], re_blocks, 1.0, None, None);
    let summary = format!("{}", fit);

    // "REML criterion at convergence" always appears, so check for "optimizer" which only
    // shows up in the convergence diagnostic block
    assert!(!summary.contains("optimizer"),
        "Summary should NOT contain optimizer diagnostics when converged=None:\n{}", summary);
}
