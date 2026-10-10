//! Robust Standard Errors (Sandwich Estimators) for Mixed Models
//!
//! Computes empirical variance estimators robust to misspecification of the
//! variance-covariance structure, typically using the Huber-White estimator
//! or cluster-robust standard errors (CRSE).

use crate::LmeFit;
use ndarray::{Array1, Array2};

// Keep row-scan buffers out of the small-design validation path.
#[inline(never)]
fn matches_contiguous_design(actual: &[f64], expected: &[f64], p: usize) -> bool {
    let mut scales = vec![0.0_f64; p];
    let mut differences = vec![0.0_f64; p];
    for (a, b) in actual.chunks_exact(p).zip(expected.chunks_exact(p)) {
        for (((scale, difference), &a), &b) in scales.iter_mut().zip(&mut differences).zip(a).zip(b)
        {
            *scale = scale.max(a.abs()).max(b.abs());
            *difference = difference.max((a - b).abs());
        }
    }
    scales
        .iter()
        .zip(differences)
        .all(|(&scale, difference)| difference <= 64.0 * f64::EPSILON * scale)
}

/// Result of computing Robust Standard Errors (Sandwich Estimators)
#[derive(Debug, Clone)]
pub struct RobustResult {
    /// The robust variance-covariance matrix of the fixed effects
    pub v_beta_robust: Array2<f64>,
    /// Robust standard errors for the fixed effects
    pub robust_se: Array1<f64>,
    /// Robust t-values (or z-values) for the fixed effects
    pub robust_t: Array1<f64>,
    /// Asymptotic p-values based on a normal distribution
    pub robust_p_values: Option<Array1<f64>>,
}

/// Compute observation-level (HC0) robust standard errors.
///
/// Formula: V_robust = (X^T V^{-1} X)^{-1} (X^T V^{-1} diag(r^2) V^{-1} X) (X^T V^{-1} X)^{-1}
/// We approximate V^{-1} X using the weighted design matrices already computed during the fit.
pub fn compute_robust_se(
    fit: &LmeFit,
    data: &polars::prelude::DataFrame,
    cluster_col: Option<&str>,
) -> Result<RobustResult, String> {
    fit.ensure_converged().map_err(|e| e.to_string())?;
    let p = fit.coefficients.len();
    let n = fit.residuals.len();
    if data.height() != n {
        return Err(format!(
            "Robust inference data has {} rows but the fitted model has {n}",
            data.height()
        ));
    }

    let (x_mat, _) = fit
        .model_spec()
        .prediction_matrix(data)
        .map_err(|e| format!("Failed building X matrix: {e}"))?;
    let training_x = fit
        .fixed_design_x
        .as_ref()
        .ok_or("Training design missing for robust inference")?;
    let mismatch = x_mat.dim() != training_x.dim()
        || if x_mat.ncols() >= 8 && x_mat.is_standard_layout() && training_x.is_standard_layout() {
            // Scan wider contiguous designs once by row, retaining a distinct
            // scale and maximum difference for each column.
            !matches_contiguous_design(
                x_mat.as_slice().expect("standard-layout prediction design"),
                training_x
                    .as_slice()
                    .expect("standard-layout training design"),
                x_mat.ncols(),
            )
        } else {
            x_mat
                .columns()
                .into_iter()
                .zip(training_x.columns())
                .any(|(actual, expected)| {
                    // Reapplying a stored QR/spline basis can round differently at zero.
                    // Compare in each column's units, allowing only floating-point noise.
                    let scale = actual
                        .iter()
                        .chain(expected.iter())
                        .map(|v| v.abs())
                        .fold(0.0_f64, f64::max);
                    actual
                        .iter()
                        .zip(expected)
                        .any(|(&a, &b)| (a - b).abs() > 64.0 * f64::EPSILON * scale)
                })
        };
    if mismatch {
        return Err(
            "Robust inference requires the original fixed design in training row order".into(),
        );
    }

    let v_beta_unscaled = fit
        .v_beta_unscaled
        .as_ref()
        .ok_or("Unscaled V_beta missing")?;
    crate::validate_observation_weights(fit.weights.as_ref(), n).map_err(|e| e.to_string())?;
    // Precision-weighted scores equal the scores of the explicitly whitened
    // model: (sqrt(w) X) * (sqrt(w) residual) = w X residual.
    let eps = match &fit.weights {
        Some(weights) => &fit.residuals * weights,
        None => fit.residuals.clone(),
    };

    // Form coefficient influences before squaring scores. Computing the raw
    // meat first can underflow or overflow after harmless changes of units,
    // even when the final sandwich covariance is readily representable.
    let mut influences = x_mat.dot(&v_beta_unscaled.t());
    for (mut row, &residual) in influences.outer_iter_mut().zip(eps.iter()) {
        row *= residual;
    }
    let mut v_robust = Array2::<f64>::zeros((p, p));

    match cluster_col {
        Some(col_name) => {
            let series = data
                .column(col_name)
                .map_err(|e| e.to_string())?
                .cast(&polars::datatypes::DataType::String)
                .map_err(|e| e.to_string())?;
            let str_ca = series
                .str()
                .map_err(|_| "Failed to cast to string chunked array")?;
            if str_ca.null_count() > 0 {
                return Err(format!("Cluster column '{col_name}' contains nulls"));
            }

            // Map clusters
            use std::collections::HashMap;
            let mut cluster_scores: HashMap<String, Array1<f64>> = HashMap::new();

            for i in 0..n {
                let g = str_ca.get(i).unwrap_or("").to_string();
                let score = cluster_scores.entry(g).or_insert_with(|| Array1::zeros(p));
                for j in 0..p {
                    score[j] += influences[[i, j]];
                }
            }

            for score in cluster_scores.values() {
                for i in 0..p {
                    for j in 0..p {
                        v_robust[[i, j]] += score[i] * score[j];
                    }
                }
            }
        }
        None => {
            for i in 0..n {
                for i_dim in 0..p {
                    for j_dim in 0..p {
                        v_robust[[i_dim, j_dim]] += influences[[i, i_dim]] * influences[[i, j_dim]];
                    }
                }
            }
        }
    }

    let mut robust_se = Array1::zeros(p);
    let mut robust_t = Array1::zeros(p);
    for i in 0..p {
        robust_se[i] = v_robust[[i, i]].sqrt();
        robust_t[i] = fit.coefficients[i] / robust_se[i];
    }

    // Normal approximation for p-values
    let mut robust_p_values = Array1::zeros(p);
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();
    for i in 0..p {
        robust_p_values[i] = 2.0 * normal.sf(robust_t[i].abs());
    }

    Ok(RobustResult {
        v_beta_robust: v_robust,
        robust_se,
        robust_t,
        robust_p_values: Some(robust_p_values),
    })
}
