//! Random-effect covariance from lower-triangular Cholesky factors.
//!
//! **Scale vs `lme4`:** Packed θ is the lower triangle of relative `Λ` in `Σ = σ² Λ Λᵀ`
//! (same packing as `getME(., "theta")`). Inner PLS penalizes `‖Λ⁻¹b‖²`; reported RE
//! standard deviations are `θᵢ · σ` on the diagonal of `σ Λ`.

use ndarray::Array2;

/// Convert `lme4` relative θ and residual σ to absolute Cholesky scale (RE SDs).
#[allow(dead_code)]
pub fn theta_abs_from_lme4(theta_rel: &[f64], sigma: f64) -> Vec<f64> {
    theta_rel.iter().map(|t| t * sigma).collect()
}

/// Number of θ parameters for a `k`-dimensional random effect.
pub fn theta_len(k: usize) -> usize {
    k * (k + 1) / 2
}

/// Build lower-triangular `L` from packed θ (column-major lower triangle).
pub fn lower_chol(k: usize, theta: &[f64]) -> Array2<f64> {
    let mut l = Array2::<f64>::zeros((k, k));
    let mut t = 0usize;
    for col in 0..k {
        for row in col..k {
            let val = if row == col {
                theta[t].max(1e-8)
            } else {
                theta[t]
            };
            l[[row, col]] = val;
            t += 1;
        }
    }
    l
}

/// Covariance `Σ = L Lᵀ`.
pub fn sigma_from_theta(k: usize, theta: &[f64]) -> Array2<f64> {
    let l = lower_chol(k, theta);
    l.dot(&l.t())
}

/// Inverse covariance `Σ⁻¹` for penalties in penalized least squares.
pub fn sigma_inv_from_theta(k: usize, theta: &[f64]) -> Array2<f64> {
    let sigma = sigma_from_theta(k, theta);
    if k == 1 {
        let v = sigma[[0, 0]].max(1e-16);
        return Array2::from_elem((1, 1), 1.0 / v);
    }
    let det = sigma[[0, 0]] * sigma[[1, 1]] - sigma[[0, 1]] * sigma[[1, 0]];
    let det = det.max(1e-16);
    let mut inv = Array2::<f64>::zeros((2, 2));
    inv[[0, 0]] = sigma[[1, 1]] / det;
    inv[[1, 1]] = sigma[[0, 0]] / det;
    inv[[0, 1]] = -sigma[[0, 1]] / det;
    inv[[1, 0]] = -sigma[[1, 0]] / det;
    inv
}

/// `log |Σ|` for profiling (`Σ = L Lᵀ`).
pub fn log_det_sigma(k: usize, theta: &[f64]) -> f64 {
    let l = lower_chol(k, theta);
    let mut log_det = 0.0;
    for i in 0..k {
        log_det += l[[i, i]].ln();
    }
    2.0 * log_det
}

/// Solve `L x = b` for lower-triangular `L`.
pub fn lower_solve(k: usize, theta: &[f64], rhs: &[f64]) -> Vec<f64> {
    let l = lower_chol(k, theta);
    let mut x = vec![0.0; k];
    for i in 0..k {
        let mut s = rhs[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        x[i] = s / l[[i, i]].max(1e-12);
    }
    x
}

/// Inverse of the lower Cholesky factor `L` (`Λ⁻¹` in lme4 notation).
#[allow(dead_code)]
pub fn inv_lower_chol(k: usize, theta: &[f64]) -> Array2<f64> {
    let l = lower_chol(k, theta);
    if k == 1 {
        return Array2::from_elem((1, 1), 1.0 / l[[0, 0]]);
    }
    let mut inv = Array2::<f64>::zeros((k, k));
    for col in 0..k {
        let mut e = vec![0.0; k];
        e[col] = 1.0;
        let x = lower_solve(k, theta, &e);
        for row in 0..k {
            inv[[row, col]] = x[row];
        }
    }
    inv
}

/// Penalty residuals `Λ⁻¹ b` (one `k`-vector per group) for augmented least squares.
pub fn re_penalty_residuals(k: usize, theta: &[f64], b: &[f64]) -> Vec<f64> {
    lower_solve(k, theta, b)
}

/// Quadratic penalty `bᵀ Σ⁻¹ b` for a `k`-vector of random effects.
pub fn re_penalty(k: usize, theta: &[f64], b: &[f64]) -> f64 {
    let r = re_penalty_residuals(k, theta, b);
    r.iter().map(|v| v * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_matches_variance() {
        let inv = sigma_inv_from_theta(1, &[5.0]);
        assert!((inv[[0, 0]] - 0.04).abs() < 1e-12);
        assert!((log_det_sigma(1, &[5.0]) - (25.0_f64).ln()).abs() < 1e-12);
    }
}
