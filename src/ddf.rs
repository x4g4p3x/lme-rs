//! Denominator degrees of freedom for multi-dimensional Wald / F tests.
//!
//! Follows `lmerTest::get_Fstat_ddf()` and `contestMD.lmerModLmerTest()` (Satterthwaite path).

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

use crate::LmeError;

/// Quadratic form `x' M x` for symmetric `M`.
#[inline]
pub(crate) fn qform(x: &Array1<f64>, m: &Array2<f64>) -> f64 {
    x.dot(&(m.dot(x)))
}

/// Quadratic form `g' A g` with `g` a row vector stored as length-`n` array.
#[inline]
pub(crate) fn qform_grad(g: ndarray::ArrayView1<f64>, a: &Array2<f64>) -> f64 {
    let mut denom = 0.0;
    for j in 0..g.len() {
        for k in 0..g.len() {
            denom += g[j] * a[[j, k]] * g[k];
        }
    }
    denom
}

/// Denominator df for an F statistic formed from independent squared t-statistics (`lmerTest`).
pub(crate) fn get_fstat_ddf(nu: &[f64], tol: f64) -> f64 {
    debug_assert!(!nu.is_empty());
    if nu.len() == 1 {
        return nu[0];
    }
    if nu.iter().all(|&df| (df - nu[0]).abs() < tol) {
        return nu[0];
    }
    if nu.iter().any(|&df| df <= 2.0) {
        return 2.0;
    }
    let q = nu.len() as f64;
    let e: f64 = nu.iter().map(|&df| df / (df - 2.0)).sum();
    2.0 * e / (e - q)
}

/// Apply contrast rows `l_mat` (q × p) to variance blocks.
fn transform_contrast(
    l_mat: &Array2<f64>,
    v_beta: &Array2<f64>,
    jac_vcov: &[Array2<f64>],
) -> (Array2<f64>, Vec<Array2<f64>>) {
    let v_s = l_mat.dot(v_beta).dot(&l_mat.t());
    let jac_s: Vec<Array2<f64>> = jac_vcov
        .iter()
        .map(|jac| l_mat.dot(jac).dot(&l_mat.t()))
        .collect();
    (v_s, jac_s)
}

/// Multi-DoF Satterthwaite F-test for contrast matrix `l_mat` (q × p), `lmerTest::contestMD`.
pub(crate) fn satterthwaite_contrast_f_test(
    beta: &Array1<f64>,
    v_beta: &Array2<f64>,
    l_mat: &Array2<f64>,
    jac_vcov: &[Array2<f64>],
    a_mat: &Array2<f64>,
) -> crate::Result<(f64, f64, f64)> {
    let q = l_mat.nrows();
    if q == 0 {
        return Err(LmeError::NotImplemented {
            feature: "No coefficients to test".to_string(),
        });
    }

    let beta_s = l_mat.dot(beta);
    let (v_s, jac_s) = transform_contrast(l_mat, v_beta, jac_vcov);

    if q == 1 {
        let f = {
            let t = beta_s[0] / v_s[[0, 0]].sqrt();
            t * t
        };
        let grad = Array1::from_iter(jac_s.iter().map(|js| js[[0, 0]]));
        let denom = qform_grad(grad.view(), a_mat);
        let var_m = v_s[[0, 0]];
        let mut den_df = if denom > 1e-12 {
            2.0 * var_m * var_m / denom
        } else {
            f64::NAN
        };
        if den_df.is_nan() || den_df <= 0.0 {
            den_df = f64::NAN;
        }
        let p = fisher_p(f, 1.0, den_df);
        return Ok((f, den_df, p));
    }

    let (eval, evec) = v_s
        .eigh(UPLO::Upper)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!(
                "Eigen decomposition for multi-DoF Satterthwaite failed: {}",
                e
            ),
        })?;

    let eps = f64::EPSILON.sqrt();
    let d_max = eval.iter().copied().fold(0.0_f64, f64::max);
    let tol = (eps * d_max).max(0.0);

    let mut directions: Vec<(usize, f64)> = eval
        .iter()
        .enumerate()
        .filter(|(_, &d)| d > tol)
        .map(|(i, &d)| (i, d))
        .collect();
    directions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let q_eff = directions.len();
    if q_eff == 0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let mut t2_sum = 0.0;
    let mut nu_m = Vec::with_capacity(q_eff);

    for (col, var_m) in directions {
        let c_m = evec.column(col);
        let contrast = c_m.dot(&beta_s);
        t2_sum += contrast * contrast / var_m;

        let c_owned = c_m.to_owned();
        let grad_var: Array1<f64> = Array1::from_iter(jac_s.iter().map(|js| qform(&c_owned, js)));
        let denom = qform_grad(grad_var.view(), a_mat);
        let mut df_m = if denom > 1e-12 {
            2.0 * var_m * var_m / denom
        } else {
            f64::NAN
        };
        if df_m.is_nan() || df_m <= 0.0 {
            df_m = f64::NAN;
        } else if df_m > 3000.0 {
            df_m = 3000.0;
        }
        nu_m.push(df_m);
    }

    let f_stat = t2_sum / (q_eff as f64);
    let den_df = if nu_m.iter().any(|d| d.is_nan()) {
        f64::NAN
    } else {
        get_fstat_ddf(&nu_m, 1e-8)
    };
    let p = fisher_p(f_stat, q_eff as f64, den_df);
    Ok((f_stat, den_df, p))
}

fn fisher_p(f_stat: f64, num_df: f64, den_df: f64) -> f64 {
    if f_stat.is_nan() || den_df.is_nan() || den_df <= 0.0 || num_df <= 0.0 {
        f64::NAN
    } else if let Ok(dist) = FisherSnedecor::new(num_df, den_df) {
        1.0 - dist.cdf(f_stat)
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_fstat_ddf_single_returns_nu() {
        assert_eq!(get_fstat_ddf(&[17.5], 1e-8), 17.5);
    }

    #[test]
    fn get_fstat_ddf_equal_nu_returns_mean() {
        assert!((get_fstat_ddf(&[48.0, 48.0], 1e-8) - 48.0).abs() < 1e-10);
    }

    #[test]
    fn get_fstat_ddf_unequal_nu_above_minimum() {
        let ddf = get_fstat_ddf(&[20.0, 48.0], 1e-8);
        assert!(ddf > 20.0);
        assert!(ddf < 48.0);
        let e = 20.0 / 18.0 + 48.0 / 46.0;
        let expected = 2.0 * e / (e - 2.0);
        assert!((ddf - expected).abs() < 1e-10);
    }

    #[test]
    fn get_fstat_ddf_any_nu_le_two_returns_two() {
        assert_eq!(get_fstat_ddf(&[1.5, 40.0], 1e-8), 2.0);
    }
}
