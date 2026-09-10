//! Kenward–Roger model comparison F-tests (`pbkrtest::KRmodcomp` / `.KR_adjust`).
//!
//! Multi-DoF fixed-effect tests use the same adjusted covariance and variance-parameter
//! derivatives as univariate KR inference, not marginal-df pooling.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Solve, SVD};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::LmeError;

/// Cached matrices from [`crate::kenward_roger::compute_kenward_roger`] for contrast F-tests.
#[derive(Debug, Clone)]
pub(crate) struct KenwardRogerModcompData {
    /// Unadjusted fixed-effects covariance `Phi` (= `vcov`).
    pub phi: Array2<f64>,
    /// KR-adjusted covariance `PhiA` (= `vcovAdj`).
    pub phi_a: Array2<f64>,
    /// `P[[i]]` = ∂Φ/∂ρᵢ (one matrix per variance parameter, including σ²).
    pub p_list: Vec<Array2<f64>>,
    /// `W` = 2 × inverse expected information for ρ (pbkrtest `attr(PhiA, "W")`).
    pub w: Array2<f64>,
}

/// Result of one `KRmodcomp` contrast test.
#[derive(Debug, Clone, Copy)]
pub struct KrModcompTest {
    /// Scaled F statistic (`Ftest` in pbkrtest).
    pub f_stat: f64,
    /// Denominator df (`ddf` in pbkrtest).
    pub den_df: f64,
    /// Upper-tail p-value from the scaled F.
    pub p_value: f64,
}

#[inline]
fn spur(m: &Array2<f64>) -> f64 {
    m.diag().sum()
}

/// R `sum(ui * t(uj))` — Frobenius inner product.
#[inline]
fn sum_ui_t_uj(ui: &Array2<f64>, uj: &Array2<f64>) -> f64 {
    let ujt = uj.t();
    ui.iter().zip(ujt.iter()).map(|(&a, &b)| a * b).sum()
}

#[inline]
fn div_zero(num: f64, denom: f64, tol: f64) -> f64 {
    if denom.abs() < tol {
        0.0
    } else {
        num / denom
    }
}

/// Represent the hypothesis by independent, well-scaled rows. Redundant
/// restrictions add no information and must not make the covariance solve singular.
fn contrast_basis(l: &Array2<f64>) -> crate::Result<Array2<f64>> {
    if l.is_empty() || l.iter().any(|v| !v.is_finite()) {
        return Err(LmeError::InvalidInput {
            message: "Contrast matrix must be nonempty and finite".to_string(),
        });
    }
    let mut normalized = l.clone();
    for mut row in normalized.rows_mut() {
        let scale = row.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if scale > 0.0 {
            row.mapv_inplace(|v| v / scale);
        }
    }
    if normalized.iter().all(|v| *v == 0.0) {
        return Err(LmeError::InvalidInput {
            message: "Contrast matrix must have positive rank".to_string(),
        });
    }
    if l.nrows() == 1 {
        return Ok(normalized);
    }
    let (_, singular, vt) = normalized
        .svd(false, true)
        .map_err(|e| LmeError::LinearAlgebra {
            message: format!("Contrast rank decomposition failed: {e}"),
        })?;
    let tol = f64::EPSILON * l.nrows().max(l.ncols()) as f64 * singular[0];
    let rank = singular.iter().filter(|&&v| v > tol).count();
    let vt = vt.ok_or_else(|| LmeError::LinearAlgebra {
        message: "Contrast decomposition did not return a row basis".to_string(),
    })?;
    Ok(vt.slice(ndarray::s![..rank, ..]).to_owned())
}

/// Solve `A X = B` for `X` (square `A`, `B` is q × p).
fn solve_left(a: &Array2<f64>, b: &Array2<f64>) -> crate::Result<Array2<f64>> {
    let q = a.nrows();
    let p = b.ncols();
    let mut x = Array2::zeros((q, p));
    for j in 0..p {
        let col = b.column(j).to_owned();
        let sol = a.solve(&col).map_err(|e| LmeError::NotImplemented {
            feature: format!("KRmodcomp linear solve failed: {}", e),
        })?;
        x.column_mut(j).assign(&sol);
    }
    Ok(x)
}

/// `pbkrtest::.KR_adjust` for hypothesis `L β = L β_H` (default `β_H = 0`).
pub fn kr_modcomp_test(
    data: &KenwardRogerModcompData,
    l_mat: &Array2<f64>,
    beta: &Array1<f64>,
    beta_h: Option<&Array1<f64>>,
) -> crate::Result<KrModcompTest> {
    let basis = contrast_basis(l_mat)?;
    kr_modcomp_full_rank(data, &basis, beta, beta_h)
}

fn kr_modcomp_full_rank(
    data: &KenwardRogerModcompData,
    l_mat: &Array2<f64>,
    beta: &Array1<f64>,
    beta_h: Option<&Array1<f64>>,
) -> crate::Result<KrModcompTest> {
    let p = beta.len();
    if l_mat.ncols() != p {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Contrast columns {} do not match coefficient length {}",
                l_mat.ncols(),
                p
            ),
        });
    }

    let q = l_mat.nrows() as f64;

    let beta_h = beta_h.cloned().unwrap_or_else(|| Array1::zeros(p));
    let beta_diff = beta - &beta_h;

    let phi = &data.phi;
    let phi_a = &data.phi_a;
    let n_g = data.p_list.len();

    // Theta = t(L) %*% solve(L %*% Phi %*% t(L), L)
    let lpl = l_mat.dot(phi).dot(&l_mat.t());
    let theta_rhs = solve_left(&lpl, l_mat)?;
    let theta = l_mat.t().dot(&theta_rhs);
    let theta_phi = theta.dot(phi);

    let mut a1 = 0.0;
    let mut a2 = 0.0;
    for i in 0..n_g {
        for j in i..n_g {
            let e = if i == j { 1.0 } else { 2.0 };
            let ui = theta_phi.dot(&data.p_list[i]).dot(phi);
            let uj = theta_phi.dot(&data.p_list[j]).dot(phi);
            a1 += e * data.w[[i, j]] * spur(&ui) * spur(&uj);
            a2 += e * data.w[[i, j]] * sum_ui_t_uj(&ui, &uj);
        }
    }

    let qi = q as usize;
    let b = (a1 + 6.0 * a2) / (2.0 * q);
    let g = if a2.abs() > 1e-15 {
        ((qi + 1) as f64 * a1 - (qi + 4) as f64 * a2) / ((qi + 2) as f64 * a2)
    } else {
        0.0
    };
    let denom_g = 3.0 * q + 2.0 * (1.0 - g);
    let c1 = if denom_g.abs() > 1e-15 {
        g / denom_g
    } else {
        0.0
    };
    let c2 = (q - g) / denom_g;
    let c3 = (q + 2.0 - g) / denom_g;

    let v0 = {
        let x = 1.0 + c1 * b;
        if x.abs() < 1e-10 {
            0.0
        } else {
            x
        }
    };
    let v1 = 1.0 - c2 * b;
    let v2 = {
        let x = 1.0 - c3 * b;
        if x.abs() < 1e-10 {
            0.0
        } else {
            x
        }
    };

    let rho = (1.0 / q) * div_zero(1.0 - a2 / q, v1, 1e-10).powi(2) * div_zero(v0, v2, 1e-10);
    let mut df2 = if (q * rho - 1.0).abs() > 1e-12 {
        4.0 + (q + 2.0) / (q * rho - 1.0)
    } else {
        f64::NAN
    };
    if df2.is_nan() || df2 <= 0.0 {
        df2 = f64::NAN;
    }

    let f_scaling = if (df2 - 2.0).abs() < 1e-2 {
        1.0
    } else if df2.is_nan() {
        f64::NAN
    } else {
        df2 * (1.0 - a2 / q) / (df2 - 2.0)
    };

    // The KR F statistic uses the adjusted covariance (pbkrtest's Wald).
    let lpl_a = l_mat.dot(phi_a).dot(&l_mat.t());
    let rhs_wald = l_mat.dot(&beta_diff);
    let wald = solve_wald_quadratic(&lpl_a, &rhs_wald)?;

    let f_stat_u = wald / q;
    let f_stat = if f_scaling.is_nan() {
        f64::NAN
    } else {
        f_scaling * f_stat_u
    };

    let p_value = fisher_upper_tail(f_stat, q, df2);

    Ok(KrModcompTest {
        f_stat,
        den_df: df2,
        p_value,
    })
}

/// `t(betaDiff) %*% t(L) %*% solve(L %*% V %*% t(L), L %*% betaDiff)` for vector `rhs = L %*% betaDiff`.
fn solve_wald_quadratic(v: &Array2<f64>, rhs: &Array1<f64>) -> crate::Result<f64> {
    let q = v.nrows();
    if q == 0 {
        return Ok(f64::NAN);
    }
    if q == 1 {
        let denom = v[[0, 0]];
        return Ok(if denom.abs() > 1e-15 {
            rhs[0] * rhs[0] / denom
        } else {
            f64::NAN
        });
    }
    let rhs2 = rhs.clone().insert_axis(ndarray::Axis(1));
    let sol = solve_left(v, &rhs2)?;
    Ok(rhs.dot(&sol.column(0)))
}

fn fisher_upper_tail(f_stat: f64, num_df: f64, den_df: f64) -> f64 {
    if f_stat.is_nan() || den_df.is_nan() || den_df <= 0.0 || num_df <= 0.0 {
        f64::NAN
    } else if let Ok(dist) = FisherSnedecor::new(num_df, den_df) {
        dist.sf(f_stat)
    } else {
        f64::NAN
    }
}

/// True when KR covariance adjustment is negligible (`vcovAdj` ≈ `vcov`).
#[cfg(test)]
pub(crate) fn phi_a_near_phi(phi: &Array2<f64>, phi_a: &Array2<f64>, rtol: f64) -> bool {
    phi.iter().zip(phi_a.iter()).all(|(&a, &b)| {
        let scale = a.abs().max(b.abs()).max(1e-10);
        (a - b).abs() <= rtol * scale
    })
}

/// Multi-DoF Kenward–Roger F-test for contrast matrix `l_mat` (q × p).
///
/// Uses the same adjusted covariance and contrast-specific degrees of freedom
/// for every representation of the hypothesis.
pub fn kenward_roger_contrast_f_test(
    data: &KenwardRogerModcompData,
    beta: &Array1<f64>,
    l_mat: &Array2<f64>,
    beta_h: Option<&Array1<f64>>,
) -> crate::Result<(f64, f64, f64, f64)> {
    let basis = contrast_basis(l_mat)?;
    let num_df = basis.nrows() as f64;
    let res = kr_modcomp_full_rank(data, &basis, beta, beta_h)?;
    Ok((res.f_stat, res.den_df, res.p_value, num_df))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn div_zero_matches_r_tol() {
        assert_eq!(div_zero(1.0, 0.0, 1e-10), 0.0);
        assert!((div_zero(2.0, 4.0, 1e-10) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn contrast_rank_single_row() {
        let l = array![[0.0, 1.0, 0.0]];
        assert_eq!(contrast_basis(&l).unwrap().nrows(), 1);
    }

    #[test]
    fn kr_wald_uses_adjusted_covariance() {
        // One-dimensional case: A1 = A2 = 0.1, hence df = 20 and
        // F scaling = 1. The adjusted Wald statistic is 3^2 / 3 = 3,
        // whereas the unadjusted covariance would incorrectly give 4.5.
        let data = KenwardRogerModcompData {
            phi: array![[2.0]],
            phi_a: array![[3.0]],
            p_list: vec![array![[0.5]]],
            w: array![[0.1]],
        };
        let result = kr_modcomp_test(&data, &array![[1.0]], &array![3.0], None).unwrap();
        assert!((result.den_df - 20.0).abs() < 1e-10);
        assert!((result.f_stat - 3.0).abs() < 1e-10);
    }
}
