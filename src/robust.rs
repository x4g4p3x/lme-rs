//! Robust Standard Errors (Sandwich Estimators) for Mixed Models
//!
//! Computes empirical variance estimators robust to misspecification of the
//! variance-covariance structure, typically using the Huber-White estimator
//! or cluster-robust standard errors (CRSE).

use crate::LmeFit;
use ndarray::{Array1, Array2};

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
    let p = fit.coefficients.len();
    let n = fit.residuals.len();

    let ast = crate::formula::parse(&fit.formula.clone().unwrap_or_default())
        .map_err(|e| format!("Failed to parse formula: {}", e))?;

    let mut response_col = String::new();
    for (name, info) in &ast.columns {
        if info.roles.contains(&"Response".to_string()) {
            response_col = name.clone();
            break;
        }
    }

    let (x_mat, _, _) = crate::model_matrix::build_x_matrix(&ast, data, &response_col, n, fit.categorical_levels.as_ref())
        .map_err(|e| format!("Failed building X matrix: {}", e))?;

    let v_beta_unscaled = fit
        .v_beta_unscaled
        .as_ref()
        .ok_or("Unscaled V_beta missing")?;
    let eps = &fit.residuals;

    let mut meat = Array2::<f64>::zeros((p, p));

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

            // Map clusters
            use std::collections::HashMap;
            let mut cluster_scores: HashMap<String, Array1<f64>> = HashMap::new();

            for i in 0..n {
                let g = str_ca.get(i).unwrap_or("").to_string();
                let score = cluster_scores.entry(g).or_insert_with(|| Array1::zeros(p));
                for j in 0..p {
                    score[j] += x_mat[[i, j]] * eps[i];
                }
            }

            for score in cluster_scores.values() {
                for i in 0..p {
                    for j in 0..p {
                        meat[[i, j]] += score[i] * score[j];
                    }
                }
            }
        }
        None => {
            for i in 0..n {
                for i_dim in 0..p {
                    for j_dim in 0..p {
                        meat[[i_dim, j_dim]] +=
                            x_mat[[i, i_dim]] * x_mat[[i, j_dim]] * eps[i] * eps[i];
                    }
                }
            }
        }
    }

    let b_matrix = v_beta_unscaled;
    let v_robust = b_matrix.dot(&meat).dot(b_matrix); // B M B

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
        robust_p_values[i] = 2.0 * (1.0 - normal.cdf(robust_t[i].abs()));
    }

    Ok(RobustResult {
        v_beta_robust: v_robust,
        robust_se,
        robust_t,
        robust_p_values: Some(robust_p_values),
    })
}
