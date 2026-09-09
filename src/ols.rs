//! QR-based least squares without normal-equation inversion.
use crate::{LmeError, Result};
use ndarray::{Array1, Array2};

pub(crate) fn solve_qr(r: &Array2<f64>, qty: &Array1<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = r.ncols();
    let scale = r.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let tolerance = f64::EPSILON * (r.nrows().max(p) as f64) * scale;
    if (0..p).any(|i| !r[[i, i]].is_finite() || r[[i, i]].abs() <= tolerance) {
        return Err(LmeError::LinearAlgebra {
            message: "rank-deficient fixed-effects design".into(),
        });
    }
    let backsolve = |rhs: &Array1<f64>| {
        let mut out = rhs.clone();
        for i in (0..p).rev() {
            for j in i + 1..p {
                out[i] -= r[[i, j]] * out[j];
            }
            out[i] /= r[[i, i]];
        }
        out
    };
    let coefficients = backsolve(qty);
    // Cov(beta)/sigma^2 = R^-1 R^-T. Solve triangular systems for the columns.
    let mut inverse_factor = Array2::zeros((p, p));
    for j in 0..p {
        let mut unit = Array1::zeros(p);
        unit[j] = 1.0;
        inverse_factor.column_mut(j).assign(&backsolve(&unit));
    }
    Ok((coefficients, inverse_factor.dot(&inverse_factor.t())))
}
