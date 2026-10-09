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

#[test]
fn robust_wald_intervals_use_the_sandwich_standard_error() {
    let data = df!(
        "x" => [1., 2., 3., 4., 5., 6., 7., 8.],
        "y" => [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1],
        "group" => ["A", "A", "B", "B", "C", "C", "D", "D"]
    )
    .unwrap();
    for cluster in [None, Some("group")] {
        let mut fit = lm_df("y ~ 0 + x", &data).unwrap();
        fit.with_robust_se(&data, cluster).unwrap();
        let ci = fit.confint(0.95).unwrap();
        // Analytic scalar HC0/CR0 covariance above, and the standard normal
        // 97.5th percentile, independently specify the two-sided Wald interval.
        let margin = 1.959_963_984_540_054 * independent_covariance(cluster.is_some()).sqrt();
        let beta = fit.coefficients[0];
        assert!(
            (ci.lower[0] - (beta - margin)).abs() < 1e-12,
            "cluster {cluster:?}: lower {} versus {}",
            ci.lower[0],
            beta - margin
        );
        assert!((ci.upper[0] - (beta + margin)).abs() < 1e-12);
        for invalid in [0.0, 1.0, f64::NAN] {
            assert!(fit.confint(invalid).is_err());
        }
    }

    // Preserve the existing rejection of models without estimable uncertainty.
    let saturated = df!("x" => [1., 2.], "y" => [1., 4.]).unwrap();
    let mut fit = lm_df("y ~ x", &saturated).unwrap();
    fit.with_robust_se(&saturated, None).unwrap();
    assert!(fit.confint(0.95).is_err());
}

#[test]
fn robust_wald_intervals_ignore_stored_model_based_ddf_adjustments() {
    let file = std::fs::File::open("tests/data/sleepstudy.csv").unwrap();
    let data = CsvReader::new(file).finish().unwrap();
    let base = lme_rs::lmer("Reaction ~ Days + (Days | Subject)", &data, true).unwrap();
    for kenward_roger in [false, true] {
        for robust_first in [false, true] {
            let mut fit = base.clone();
            if robust_first {
                fit.with_robust_se(&data, Some("Subject")).unwrap();
            }
            if kenward_roger {
                fit.with_kenward_roger(&data).unwrap();
            } else {
                fit.with_satterthwaite(&data).unwrap();
            }
            if !robust_first {
                fit.with_robust_se(&data, Some("Subject")).unwrap();
            }
            let robust = fit.robust.as_ref().unwrap();
            let ci = fit.confint(0.95).unwrap();
            for i in 0..fit.coefficients.len() {
                let margin = 1.959_963_984_540_054 * robust.robust_se[i];
                assert!(
                    (ci.lower[i] - (fit.coefficients[i] - margin)).abs() < 1e-10,
                    "KR {kenward_roger}, robust first {robust_first}, coefficient {i}"
                );
                assert!((ci.upper[i] - (fit.coefficients[i] + margin)).abs() < 1e-10);
            }
        }
    }
}
