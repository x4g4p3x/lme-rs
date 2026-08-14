//! Orthogonal polynomials (`poly`) and natural cubic splines (`ns`) for formula terms.
//!
//! Orthogonal `poly` uses a QR factorization of the centered Vandermonde matrix, matching
//! the column space of R `stats::poly`. Natural splines use cubic B-splines with a linear
//! constraint that the second derivative vanishes at the boundary knots, matching the
//! column space of R `splines::ns`. Training encodings are stored so `predict` reapplies
//! the same basis on new data.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, QRInto};

use crate::LmeError;

/// Stored training parameters so a basis can be evaluated on new data.
#[derive(Clone, Debug)]
pub enum BasisEncoding {
    /// Raw or orthogonal polynomial of a numeric column.
    Poly {
        /// Polynomial degree (number of columns).
        degree: usize,
        /// When true, columns are \(x, x^2, \ldots\) rather than an orthogonal basis.
        raw: bool,
        /// Mean of the training predictor (orthogonal only).
        xbar: f64,
        /// Upper-triangular factor of the training Vandermonde QR (orthogonal only).
        r: Array2<f64>,
    },
    /// Natural cubic spline basis.
    Ns {
        /// Full cubic B-spline knot vector (boundary knots repeated four times).
        knots: Vec<f64>,
        /// When false, the intercept B-spline column is dropped before the constraint.
        intercept: bool,
        /// Columns of the B-spline basis that span the natural-spline constraint null space.
        projection: Array2<f64>,
    },
}

const MAX_DEGREE: usize = 32;
const BSPLINE_DEGREE: usize = 3;

fn basis_error(feature: impl Into<String>) -> LmeError {
    LmeError::NotImplemented {
        feature: feature.into(),
    }
}

/// Evaluate `poly(x, degree)` or `poly(x, degree, raw = TRUE)`.
pub fn eval_poly(
    x: &Array1<f64>,
    degree: usize,
    raw: bool,
    training: Option<&BasisEncoding>,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    if !(1..=MAX_DEGREE).contains(&degree) {
        return Err(basis_error(format!(
            "poly() degree must be between 1 and {MAX_DEGREE}"
        )));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(basis_error("poly() requires finite values"));
    }
    if raw {
        return eval_raw_poly(x, degree);
    }
    match training {
        Some(BasisEncoding::Poly {
            degree: d,
            raw: false,
            xbar,
            r,
        }) if *d == degree => eval_orthogonal_poly_with_r(x, degree, *xbar, r),
        Some(BasisEncoding::Poly { raw: true, .. }) => eval_raw_poly(x, degree),
        Some(_) => Err(basis_error(
            "poly() training encoding does not match the formula term",
        )),
        None => eval_orthogonal_poly_train(x, degree),
    }
}

fn eval_raw_poly(
    x: &Array1<f64>,
    degree: usize,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    let n = x.len();
    let mut cols = Vec::with_capacity(degree);
    for d in 1..=degree {
        let mut col = Array1::zeros(n);
        for i in 0..n {
            col[i] = x[i].powi(d as i32);
            if !col[i].is_finite() {
                return Err(basis_error("non-finite value in poly(..., raw = TRUE)"));
            }
        }
        cols.push(col);
    }
    Ok((
        cols,
        BasisEncoding::Poly {
            degree,
            raw: true,
            xbar: 0.0,
            r: Array2::zeros((0, 0)),
        },
    ))
}

fn unique_sorted(x: &Array1<f64>) -> Vec<f64> {
    let mut values = x.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| a == b);
    values
}

fn eval_orthogonal_poly_train(
    x: &Array1<f64>,
    degree: usize,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    if degree >= unique_sorted(x).len() {
        return Err(basis_error(
            "poly() degree must be less than the number of unique points",
        ));
    }
    let xbar = x.mean().unwrap_or(0.0);
    let vandermonde = centered_vandermonde(x, xbar, degree);
    let (q, r) = vandermonde.qr_into().map_err(|e| LmeError::LinearAlgebra {
        message: format!("poly() QR failed: {e}"),
    })?;
    if q.ncols() < degree + 1 {
        return Err(basis_error("poly() QR produced too few columns"));
    }
    let mut cols = Vec::with_capacity(degree);
    for d in 1..=degree {
        cols.push(q.column(d).to_owned());
    }
    Ok((
        cols,
        BasisEncoding::Poly {
            degree,
            raw: false,
            xbar,
            r,
        },
    ))
}

fn eval_orthogonal_poly_with_r(
    x: &Array1<f64>,
    degree: usize,
    xbar: f64,
    r: &Array2<f64>,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    let vandermonde = centered_vandermonde(x, xbar, degree);
    let r_inv = r.inv().map_err(|e| LmeError::LinearAlgebra {
        message: format!("poly() R inverse failed: {e}"),
    })?;
    let q = vandermonde.dot(&r_inv);
    let mut cols = Vec::with_capacity(degree);
    for d in 1..=degree {
        cols.push(q.column(d).to_owned());
    }
    Ok((
        cols,
        BasisEncoding::Poly {
            degree,
            raw: false,
            xbar,
            r: r.clone(),
        },
    ))
}

fn centered_vandermonde(x: &Array1<f64>, xbar: f64, degree: usize) -> Array2<f64> {
    let n = x.len();
    let mut v = Array2::zeros((n, degree + 1));
    for i in 0..n {
        let xc = x[i] - xbar;
        let mut p = 1.0;
        v[[i, 0]] = 1.0;
        for d in 1..=degree {
            p *= xc;
            v[[i, d]] = p;
        }
    }
    v
}

/// Evaluate `ns(x, df)` or `ns(x, df, intercept = TRUE)`.
pub fn eval_ns(
    x: &Array1<f64>,
    df: usize,
    intercept: bool,
    training: Option<&BasisEncoding>,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    if !(1..=MAX_DEGREE).contains(&df) {
        return Err(basis_error(format!(
            "ns() df must be between 1 and {MAX_DEGREE}"
        )));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(basis_error("ns() requires finite values"));
    }
    match training {
        Some(BasisEncoding::Ns {
            knots,
            intercept: stored_intercept,
            projection,
        }) if *stored_intercept == intercept => {
            eval_ns_with_encoding(x, knots, intercept, projection)
        }
        Some(_) => Err(basis_error(
            "ns() training encoding does not match the formula term",
        )),
        None => eval_ns_train(x, df, intercept),
    }
}

fn eval_ns_train(
    x: &Array1<f64>,
    df: usize,
    intercept: bool,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    let unique = unique_sorted(x);
    if unique.len() < 2 {
        return Err(basis_error("ns() requires at least two unique points"));
    }
    let boundary = [unique[0], unique[unique.len() - 1]];
    if boundary[0] >= boundary[1] {
        return Err(basis_error("ns() boundary knots must have positive width"));
    }
    let intercept_flag = usize::from(intercept);
    let n_interior = df.saturating_sub(1 + intercept_flag);
    let interior = interior_knots(&unique, n_interior);
    let knots = cubic_knots(boundary, &interior);
    let bspline = bspline_design(x, &knots)?;
    let constrained = if intercept {
        bspline
    } else {
        drop_first_column(&bspline)
    };
    let const_mat = second_deriv_at_boundaries(&knots, boundary, intercept)?;
    if const_mat.ncols() != constrained.ncols() {
        return Err(basis_error(
            "ns() constraint width does not match the basis",
        ));
    }
    let projection = constraint_null_space(&const_mat)?;
    if projection.ncols() != df {
        return Err(basis_error(format!(
            "ns() produced {} columns, expected df={df}",
            projection.ncols()
        )));
    }
    let natural = constrained.dot(&projection);
    Ok((
        columns_of(&natural),
        BasisEncoding::Ns {
            knots,
            intercept,
            projection,
        },
    ))
}

fn eval_ns_with_encoding(
    x: &Array1<f64>,
    knots: &[f64],
    intercept: bool,
    projection: &Array2<f64>,
) -> crate::Result<(Vec<Array1<f64>>, BasisEncoding)> {
    let bspline = bspline_design(x, knots)?;
    let constrained = if intercept {
        bspline
    } else {
        drop_first_column(&bspline)
    };
    let natural = constrained.dot(projection);
    Ok((
        columns_of(&natural),
        BasisEncoding::Ns {
            knots: knots.to_vec(),
            intercept,
            projection: projection.clone(),
        },
    ))
}

fn interior_knots(sorted_unique: &[f64], n_interior: usize) -> Vec<f64> {
    if n_interior == 0 {
        return Vec::new();
    }
    (1..=n_interior)
        .map(|i| {
            let p = i as f64 / (n_interior + 1) as f64;
            quantile_type7(sorted_unique, p)
        })
        .collect()
}

/// R `quantile(..., type = 7)` on an already-sorted sample.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi || hi >= n {
        sorted[lo.min(n - 1)]
    } else {
        let w = h - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn cubic_knots(boundary: [f64; 2], interior: &[f64]) -> Vec<f64> {
    let mut knots = Vec::with_capacity(8 + interior.len());
    knots.extend(std::iter::repeat(boundary[0]).take(4));
    knots.extend(interior.iter().copied());
    knots.extend(std::iter::repeat(boundary[1]).take(4));
    knots
}

fn bspline_design(x: &Array1<f64>, knots: &[f64]) -> crate::Result<Array2<f64>> {
    let n_basis = knots.len().saturating_sub(BSPLINE_DEGREE + 1);
    if n_basis == 0 {
        return Err(basis_error("ns() knot vector is too short"));
    }
    let mut basis = Array2::zeros((x.len(), n_basis));
    for (row, &xi) in x.iter().enumerate() {
        for j in 0..n_basis {
            basis[[row, j]] = bspline_value(j, BSPLINE_DEGREE, xi, knots);
        }
    }
    Ok(basis)
}

fn bspline_value(i: usize, degree: usize, x: f64, knots: &[f64]) -> f64 {
    if i + degree + 1 >= knots.len() {
        return 0.0;
    }
    if degree == 0 {
        let right_end =
            i + 1 == knots.len() - 1 || knots[i + 1] == *knots.last().unwrap_or(&knots[i]);
        let in_left = x >= knots[i] && x < knots[i + 1];
        let in_closed_end = right_end && x == knots[i + 1] && knots[i] < knots[i + 1];
        // The last distinct knot interval is closed on the right so x = max(x) is in the span.
        let last_span = knots[i + 1] == knots[knots.len() - 1] && knots[i] < knots[i + 1];
        return if in_left || (last_span && x == knots[i + 1]) || in_closed_end {
            1.0
        } else {
            0.0
        };
    }
    let mut left = 0.0;
    let d1 = knots[i + degree] - knots[i];
    if d1 != 0.0 {
        left = (x - knots[i]) / d1 * bspline_value(i, degree - 1, x, knots);
    }
    let mut right = 0.0;
    let d2 = knots[i + degree + 1] - knots[i + 1];
    if d2 != 0.0 {
        right = (knots[i + degree + 1] - x) / d2 * bspline_value(i + 1, degree - 1, x, knots);
    }
    left + right
}

fn bspline_deriv(i: usize, degree: usize, x: f64, knots: &[f64], deriv: usize) -> f64 {
    if deriv == 0 {
        return bspline_value(i, degree, x, knots);
    }
    if degree == 0 || i + degree + 1 >= knots.len() {
        return 0.0;
    }
    let mut left = 0.0;
    let d1 = knots[i + degree] - knots[i];
    if d1 != 0.0 {
        left = degree as f64 / d1 * bspline_deriv(i, degree - 1, x, knots, deriv - 1);
    }
    let mut right = 0.0;
    let d2 = knots[i + degree + 1] - knots[i + 1];
    if d2 != 0.0 {
        right = degree as f64 / d2 * bspline_deriv(i + 1, degree - 1, x, knots, deriv - 1);
    }
    left - right
}

fn second_deriv_at_boundaries(
    knots: &[f64],
    boundary: [f64; 2],
    intercept: bool,
) -> crate::Result<Array2<f64>> {
    let n_basis = knots.len().saturating_sub(BSPLINE_DEGREE + 1);
    let start = usize::from(!intercept);
    if start >= n_basis {
        return Err(basis_error(
            "ns() basis is empty after dropping the intercept",
        ));
    }
    let width = n_basis - start;
    let mut const_mat = Array2::zeros((2, width));
    for (row, &knot) in boundary.iter().enumerate() {
        for j in 0..width {
            const_mat[[row, j]] = bspline_deriv(j + start, BSPLINE_DEGREE, knot, knots, 2);
        }
    }
    Ok(const_mat)
}

fn drop_first_column(m: &Array2<f64>) -> Array2<f64> {
    m.slice(ndarray::s![.., 1..]).to_owned()
}

fn columns_of(m: &Array2<f64>) -> Vec<Array1<f64>> {
    (0..m.ncols()).map(|j| m.column(j).to_owned()).collect()
}

fn constraint_null_space(const_mat: &Array2<f64>) -> crate::Result<Array2<f64>> {
    let p = const_mat.ncols();
    if p < 2 {
        return Err(basis_error("ns() constraint matrix is too narrow"));
    }
    let a = const_mat.t().to_owned();
    let (q, _) = a.qr_into().map_err(|e| LmeError::LinearAlgebra {
        message: format!("ns() constraint QR failed: {e}"),
    })?;
    let q1 = if q.ncols() >= 2 {
        q.slice(ndarray::s![.., ..2]).to_owned()
    } else {
        return Err(basis_error("ns() constraint QR produced too few columns"));
    };
    complete_orthonormal_complement(&q1)
}

fn complete_orthonormal_complement(q_thin: &Array2<f64>) -> crate::Result<Array2<f64>> {
    let p = q_thin.nrows();
    let drop = q_thin.ncols();
    let keep = p.saturating_sub(drop);
    if keep == 0 {
        return Err(basis_error("ns() natural-spline null space is empty"));
    }
    let mut cols: Vec<Array1<f64>> = Vec::with_capacity(keep);
    for j in 0..p {
        let mut v = Array1::zeros(p);
        v[j] = 1.0;
        for k in 0..drop {
            let qk = q_thin.column(k);
            let dot = v.dot(&qk);
            for i in 0..p {
                v[i] -= dot * qk[i];
            }
        }
        for prev in &cols {
            let dot = v.dot(prev);
            v = &v - &(prev * dot);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 {
            v /= norm;
            cols.push(v);
            if cols.len() == keep {
                break;
            }
        }
    }
    if cols.len() != keep {
        return Err(basis_error("ns() failed to complete the spline null space"));
    }
    let mut out = Array2::zeros((p, keep));
    for (j, col) in cols.iter().enumerate() {
        out.column_mut(j).assign(col);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn raw_poly_matches_powers() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let (cols, enc) = eval_poly(&x, 2, true, None).unwrap();
        assert!(matches!(
            enc,
            BasisEncoding::Poly {
                raw: true,
                degree: 2,
                ..
            }
        ));
        assert_eq!(cols[0], x);
        assert!((cols[1][2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn orthogonal_poly_columns_are_orthonormal() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (cols, enc) = eval_poly(&x, 2, false, None).unwrap();
        assert_eq!(cols.len(), 2);
        let d00 = cols[0].dot(&cols[0]);
        let d11 = cols[1].dot(&cols[1]);
        let d01 = cols[0].dot(&cols[1]);
        assert!((d00 - 1.0).abs() < 1e-10, "d00={d00}");
        assert!((d11 - 1.0).abs() < 1e-10, "d11={d11}");
        assert!(d01.abs() < 1e-10, "d01={d01}");
        let mean0: f64 = cols[0].mean().unwrap();
        assert!(mean0.abs() < 1e-10, "mean0={mean0}");
        let (again, _) = eval_poly(&x, 2, false, Some(&enc)).unwrap();
        for (a, b) in cols.iter().zip(again.iter()) {
            let err = (a - b).mapv(f64::abs).sum();
            assert!(err < 1e-10, "training encode/decode mismatch {err}");
        }
    }

    #[test]
    fn ns_has_requested_df_and_predict_matches() {
        let x = Array1::linspace(0.0, 1.0, 11);
        let (cols, enc) = eval_ns(&x, 3, false, None).unwrap();
        assert_eq!(cols.len(), 3);
        let (again, _) = eval_ns(&x, 3, false, Some(&enc)).unwrap();
        for (a, b) in cols.iter().zip(again.iter()) {
            let err = (a - b).mapv(f64::abs).sum();
            assert!(err < 1e-8, "ns encode/decode mismatch {err}");
        }
    }
}
