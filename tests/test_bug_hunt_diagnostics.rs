//! Independent robust-inference checks from the reusable bug-hunt workflow.

use lme_rs::lm_df;
use polars::prelude::*;

fn robust_covariance(scale: f64, cluster: Option<&str>) -> f64 {
    let x = [1., 2., 3., 4., 5., 6., 7., 8.];
    let y = [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1];
    let data = df!(
        "x" => x.map(|v| v * scale),
        "y" => y.map(|v| v * scale),
        "group" => ["A", "A", "B", "B", "C", "C", "D", "D"],
        "row" => [0, 1, 2, 3, 4, 5, 6, 7]
    )
    .unwrap();
    let mut fit = lm_df("y ~ 0 + x", &data).unwrap();
    fit.with_robust_se(&data, cluster).unwrap();
    let robust = fit.robust.unwrap();
    assert!(
        robust.robust_se[0].is_finite() && robust.robust_se[0] > 0.0,
        "scale {scale}, cluster {cluster:?}: SE {}",
        robust.robust_se[0]
    );
    robust.v_beta_robust[[0, 0]]
}

fn independent_covariance(clustered: bool) -> f64 {
    let x = [1., 2., 3., 4., 5., 6., 7., 8.];
    let y = [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1];
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let beta = x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>() / sum_xx;
    let scores: Vec<f64> = x.iter().zip(y).map(|(x, y)| x * (y - beta * x)).collect();
    let sum_scores_sq: f64 = if clustered {
        scores
            .chunks_exact(2)
            .map(|g| g.iter().sum::<f64>().powi(2))
            .sum()
    } else {
        scores.iter().map(|s| s * s).sum()
    };
    sum_scores_sq / sum_xx.powi(2)
}

#[test]
fn robust_covariance_matches_scalar_identity_and_singleton_clusters() {
    let expected = independent_covariance(false);
    let actual = robust_covariance(1., None);
    assert!((actual / expected - 1.0).abs() < 1e-12);
    let singletons = robust_covariance(1., Some("row"));
    assert!((singletons / expected - 1.0).abs() < 1e-12);
    let clustered = robust_covariance(1., Some("group"));
    assert!((clustered / independent_covariance(true) - 1.0).abs() < 1e-12);
}

fn check_units(scale: f64, cluster: Option<&str>) {
    let expected = independent_covariance(cluster.is_some());
    let actual = robust_covariance(scale, cluster);
    assert!(
        (actual / expected - 1.0).abs() < 1e-12,
        "scale {scale}: {actual} vs {expected}"
    );
}

#[test]
fn robust_hc0_preserves_small_units() {
    check_units(1e-100, None);
}

#[test]
fn robust_hc0_preserves_large_units() {
    check_units(1e100, None);
}

#[test]
fn robust_cr0_preserves_small_units() {
    check_units(1e-100, Some("group"));
}

#[test]
fn robust_cr0_preserves_large_units() {
    check_units(1e100, Some("group"));
}
