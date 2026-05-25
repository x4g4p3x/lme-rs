//! Kenward–Roger model comparison F-tests (`pbkrtest::KRmodcomp` / `.KR_adjust`).
//!
//! Multi-DoF fixed-effect tests use the same adjusted covariance and variance-parameter
//! derivatives as univariate KR inference, not marginal-df pooling.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};
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

/// Effective row rank of contrast matrix `L` (pbkrtest `rankMatrix(L)`).
fn contrast_rank(l: &Array2<f64>, eps: f64) -> usize {
    let q = l.nrows();
    if q == 0 {
        return 0;
    }
    if q == 1 {
        return if l.iter().any(|v| v.abs() > eps) { 1 } else { 0 };
    }
    let gram = l.dot(&l.t());
    let (eval, _) = gram
        .eigh(UPLO::Upper)
        .unwrap_or_else(|_| (Array1::zeros(q), Array2::zeros((q, q))));
    let d_max = eval.iter().copied().fold(0.0_f64, f64::max);
    let tol_eig = (eps * d_max).max(eps);
    eval.iter().filter(|&&d| d > tol_eig).count().max(1)
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

    let eps = f64::EPSILON.sqrt();
    let q = contrast_rank(l_mat, eps) as f64;
    if q <= 0.0 {
        return Ok(KrModcompTest {
            f_stat: f64::NAN,
            den_df: f64::NAN,
            p_value: f64::NAN,
        });
    }

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
        if x.abs() < 1e-10 { 0.0 } else { x }
    };
    let v1 = 1.0 - c2 * b;
    let v2 = {
        let x = 1.0 - c3 * b;
        if x.abs() < 1e-10 { 0.0 } else { x }
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

    // Wald statistics (adjusted and unadjusted covariance).
    let lpl_a = l_mat.dot(phi_a).dot(&l_mat.t());
    let rhs_wald = l_mat.dot(&beta_diff);
    let wald_u = solve_wald_quadratic(&lpl, &rhs_wald)?;
    let _ = solve_wald_quadratic(&lpl_a, &rhs_wald)?;

    let f_stat_u = wald_u / q;
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
        1.0 - dist.cdf(f_stat)
    } else {
        f64::NAN
    }
}

/// True when KR adjustment is negligible (`vcovAdj` ≈ `vcov`), so marginal-df pooling matches `lmerTest`/`KRmodcomp` on separable designs.
pub(crate) fn phi_a_near_phi(phi: &Array2<f64>, phi_a: &Array2<f64>, rtol: f64) -> bool {
    phi.iter().zip(phi_a.iter()).all(|(&a, &b)| {
        let scale = a.abs().max(b.abs()).max(1e-10);
        (a - b).abs() <= rtol * scale
    })
}

/// Multi-DoF Kenward–Roger F-test for contrast matrix `l_mat` (q × p).
///
/// Uses full `pbkrtest::KRmodcomp` when `PhiA` differs materially from `Phi`. When the adjusted
/// covariance equals the model covariance (common for simple random-intercept structures),
/// denominator df is pooled from marginal Kenward–Roger dfs via `get_fstat_ddf`, matching R.
pub fn kenward_roger_contrast_f_test(
    data: &KenwardRogerModcompData,
    beta: &Array1<f64>,
    l_mat: &Array2<f64>,
    marginal_dfs: &Array1<f64>,
    beta_h: Option<&Array1<f64>>,
) -> crate::Result<(f64, f64, f64, f64)> {
    let eps = f64::EPSILON.sqrt();
    let num_df = contrast_rank(l_mat, eps) as f64;
    if phi_a_near_phi(&data.phi, &data.phi_a, 1e-8) && beta_h.is_none() {
        let (f, ddf, p) = kenward_roger_contrast_marginal_pool(beta, l_mat, marginal_dfs, data)?;
        return Ok((f, ddf, p, num_df));
    }
    let res = kr_modcomp_test(data, l_mat, beta, beta_h)?;
    Ok((res.f_stat, res.den_df, res.p_value, num_df))
}

/// Marginal-df pooling on unadjusted `Phi` (equivalent to `KRmodcomp` when `PhiA` = `Phi`).
fn kenward_roger_contrast_marginal_pool(
    beta: &Array1<f64>,
    l_mat: &Array2<f64>,
    marginal_dfs: &Array1<f64>,
    data: &KenwardRogerModcompData,
) -> crate::Result<(f64, f64, f64)> {
    let q = contrast_rank(l_mat, f64::EPSILON.sqrt()) as f64;
    let beta_s = l_mat.dot(beta);
    let v_s = l_mat.dot(&data.phi).dot(&l_mat.t());
    let f_stat = {
        use ndarray_linalg::Inverse;
        if let Ok(v_inv) = v_s.inv() {
            beta_s.dot(&v_inv.dot(&beta_s)) / q
        } else {
            f64::NAN
        }
    };
    let nu_m: Vec<f64> = (0..l_mat.ncols())
        .filter(|&j| l_mat.column(j).iter().any(|v| v.abs() > 1e-12))
        .map(|j| marginal_dfs[j])
        .collect();
    let den_df = if nu_m.iter().any(|d| d.is_nan()) {
        f64::NAN
    } else {
        crate::ddf::get_fstat_ddf(&nu_m, 1e-8)
    };
    let p = fisher_upper_tail(f_stat, q, den_df);
    Ok((f_stat, den_df, p))
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
        assert_eq!(contrast_rank(&l, 1e-10), 1);
    }
}
