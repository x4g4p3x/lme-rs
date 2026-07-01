//! User-defined fixed-effects contrast tests (`lmerTest::contestMD` / `pbkrtest::KRmodcomp`).
//!
//! A contrast matrix `L` with shape **q × p** tests **H₀: L β = L β_H** (default **β_H = 0**).

use ndarray::{Array1, Array2};
use std::fmt;

use crate::anova::DdfMethod;
use crate::LmeError;
use crate::LmeFit;

/// Result of a Wald F-test for a user-supplied contrast matrix.
#[derive(Debug, Clone)]
pub struct ContrastTestResult {
    /// Denominator degrees-of-freedom method.
    pub method: DdfMethod,
    /// Numerator degrees of freedom (effective contrast rank).
    pub num_df: f64,
    /// Denominator degrees of freedom.
    pub den_df: f64,
    /// F statistic.
    pub f_value: f64,
    /// Upper-tail p-value.
    pub p_value: f64,
}

/// One contrast row as `(column_index, weight)` pairs.
pub type ContrastRow = Vec<(usize, f64)>;

/// Build a **q × p** contrast matrix from sparse row specifications.
pub fn contrast_matrix(p: usize, rows: &[ContrastRow]) -> Array2<f64> {
    let q = rows.len();
    let mut l = Array2::<f64>::zeros((q, p));
    for (i, row) in rows.iter().enumerate() {
        for &(j, w) in row {
            if j < p {
                l[[i, j]] = w;
            }
        }
    }
    l
}

/// Build a contrast matrix using fixed-effect coefficient names.
pub fn contrast_matrix_from_names(
    fixed_names: &[String],
    rows: &[ContrastRowSpec<'_>],
) -> crate::Result<Array2<f64>> {
    let p = fixed_names.len();
    let mut index_rows: Vec<ContrastRow> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut idx_row = Vec::with_capacity(row.weights.len());
        for &(name, w) in row.weights {
            let j = fixed_names.iter().position(|n| n == name).ok_or_else(|| {
                LmeError::NotImplemented {
                    feature: format!("Unknown coefficient name '{name}' in contrast"),
                }
            })?;
            idx_row.push((j, w));
        }
        index_rows.push(idx_row);
    }
    Ok(contrast_matrix(p, &index_rows))
}

/// Named weights for one contrast row.
pub struct ContrastRowSpec<'a> {
    /// Optional label (for display only).
    pub label: &'a str,
    /// `(coefficient_name, weight)` pairs.
    pub weights: &'a [(&'a str, f64)],
}

impl LmeFit {
    /// Wald F-test for **H₀: L β = 0** with Satterthwaite or Kenward–Roger denominator df.
    ///
    /// `l_mat` must have `ncols()` equal to the number of fixed-effect coefficients.
    /// Call [`Self::with_satterthwaite`] / [`Self::with_kenward_roger`] before testing when using
    /// the corresponding `ddf` method.
    pub fn test_contrast(
        &self,
        l_mat: &Array2<f64>,
        ddf: DdfMethod,
    ) -> crate::Result<ContrastTestResult> {
        fixed_effect_contrast_test(self, l_mat, ddf, None)
    }

    /// Wald F-test for **H₀: L β = L β_H** (full-length **β_H**, length **p**).
    pub fn test_contrast_vs(
        &self,
        l_mat: &Array2<f64>,
        beta_h: &Array1<f64>,
        ddf: DdfMethod,
    ) -> crate::Result<ContrastTestResult> {
        fixed_effect_contrast_test(self, l_mat, ddf, Some(beta_h))
    }
}

/// Shared contrast F-test used by [`LmeFit::test_contrast`] and [`crate::anova::LmeFit::anova_typed`].
pub(crate) fn fixed_effect_contrast_test(
    fit: &LmeFit,
    l_mat: &Array2<f64>,
    ddf: DdfMethod,
    beta_h: Option<&Array1<f64>>,
) -> crate::Result<ContrastTestResult> {
    let beta = &fit.coefficients;
    let p = beta.len();
    if l_mat.ncols() != p {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Contrast matrix has {} columns but the model has {} fixed-effect coefficients",
                l_mat.ncols(),
                p
            ),
        });
    }
    if let Some(h) = beta_h {
        if h.len() != p {
            return Err(LmeError::NotImplemented {
                feature: format!(
                    "Null vector beta_h has length {} but {} coefficients were expected",
                    h.len(),
                    p
                ),
            });
        }
    }
    if l_mat.nrows() == 0 {
        return Err(LmeError::NotImplemented {
            feature: "Contrast matrix must have at least one row".to_string(),
        });
    }

    let v_beta = if let Some(robust) = &fit.robust {
        robust.v_beta_robust.clone()
    } else {
        let xtx_inv = fit
            .v_beta_unscaled
            .as_ref()
            .ok_or(LmeError::NotImplemented {
                feature: "Covariance matrix missing".to_string(),
            })?;
        let sigma2 = fit.sigma2.unwrap_or(1.0);
        xtx_inv * sigma2
    };

    let q = l_mat.nrows();
    let (f_value, den_df, p_value, num_df) = match ddf {
        DdfMethod::Satterthwaite => {
            let (dfs, pvals) = fit
                .satterthwaite
                .as_ref()
                .map(|r| (&r.dfs, &r.p_values))
                .ok_or(LmeError::NotImplemented {
                    feature: "Satterthwaite values missing. Call with_satterthwaite() first."
                        .to_string(),
                })?;
            if q == 1 && beta_h.is_none() {
                if let Some(idx) = single_unit_contrast_index(l_mat) {
                    let t_stats = fit.beta_t.as_ref().ok_or(LmeError::NotImplemented {
                        feature: "t-statistics missing".to_string(),
                    })?;
                    return Ok(ContrastTestResult {
                        method: ddf,
                        num_df: 1.0,
                        den_df: dfs[idx],
                        f_value: t_stats[idx] * t_stats[idx],
                        p_value: pvals[idx],
                    });
                }
            }
            let multi = fit
                .satterthwaite
                .as_ref()
                .and_then(|r| r.multi_dof.as_ref())
                .ok_or(LmeError::NotImplemented {
                    feature: "Multi-DoF Satterthwaite requires with_satterthwaite() on the fit."
                        .to_string(),
                })?;
            let (f_stat, ddf_val, p_val, ndf) = crate::ddf::satterthwaite_contrast_f_test(
                beta,
                &v_beta,
                l_mat,
                &multi.jac_vcov,
                &multi.a_mat,
                beta_h,
            )?;
            (f_stat, ddf_val, p_val, ndf)
        }
        DdfMethod::KenwardRoger => {
            if fit.family.is_some() {
                return Err(LmeError::NotImplemented {
                    feature: "Kenward-Roger contrasts are only available for LMMs.".to_string(),
                });
            }
            let kr = fit.kenward_roger.as_ref().ok_or(LmeError::NotImplemented {
                feature: "Kenward-Roger values missing. Call with_kenward_roger() first."
                    .to_string(),
            })?;
            if q == 1 && beta_h.is_none() {
                if let Some(idx) = single_unit_contrast_index(l_mat) {
                    let t_stats = fit.beta_t.as_ref().ok_or(LmeError::NotImplemented {
                        feature: "t-statistics missing".to_string(),
                    })?;
                    return Ok(ContrastTestResult {
                        method: ddf,
                        num_df: 1.0,
                        den_df: kr.dfs[idx],
                        f_value: t_stats[idx] * t_stats[idx],
                        p_value: kr.p_values[idx],
                    });
                }
            }
            let (f_stat, ddf_val, p_val, ndf) = crate::kr_modcomp::kenward_roger_contrast_f_test(
                &kr.modcomp,
                beta,
                l_mat,
                &kr.dfs,
                beta_h,
            )?;
            (f_stat, ddf_val, p_val, ndf)
        }
    };

    Ok(ContrastTestResult {
        method: ddf,
        num_df,
        den_df,
        f_value,
        p_value,
    })
}

/// If `L` is a single row with one `1`, return that column index.
pub(crate) fn single_unit_contrast_index(l_mat: &Array2<f64>) -> Option<usize> {
    if l_mat.nrows() != 1 {
        return None;
    }
    let mut found = None;
    for j in 0..l_mat.ncols() {
        let v = l_mat[[0, j]];
        if v.abs() <= 1e-12 {
            continue;
        }
        if found.is_some() || (v - 1.0).abs() > 1e-12 {
            return None;
        }
        found = Some(j);
    }
    found
}

impl fmt::Display for ContrastTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let method = match self.method {
            DdfMethod::Satterthwaite => "Satterthwaite",
            DdfMethod::KenwardRoger => "Kenward-Roger",
        };
        writeln!(f, "Contrast test ({method})")?;
        writeln!(
            f,
            "  NumDF: {:.0}  DenDF: {:.4}  F: {:.4}  Pr(>F): {:.4e}",
            self.num_df, self.den_df, self.f_value, self.p_value
        )
    }
}
