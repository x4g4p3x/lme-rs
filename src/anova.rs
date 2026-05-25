pub use crate::anova_contrasts::AnovaType;
use crate::anova_contrasts::{self, contrast_for_term, AnovaTerm};
use crate::LmeFit;
use ndarray::Array1;
use std::fmt;

/// Approximation methods for creating denominator degrees of freedom in an ANOVA table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdfMethod {
    /// Satterthwaite's widely used moment-matching approximation.
    Satterthwaite,
    /// Kenward-Roger's small-sample adjusted approximation (LMMs only).
    KenwardRoger,
}

/// A structured container for analysis of variance (ANOVA) tests of fixed-effect terms.
#[derive(Debug, Clone)]
pub struct FixedEffectsAnovaResult {
    /// Sum-of-squares type used for contrasts.
    pub anova_type: AnovaType,
    /// The denominator degrees of freedom method used for the estimates.
    pub method: DdfMethod,
    /// The string name of the fixed effect variable.
    pub terms: Vec<String>,
    /// Numerator degrees of freedom per row.
    pub num_df: Array1<f64>,
    /// Denominator degrees of freedom approximated by the chosen method.
    pub den_df: Array1<f64>,
    /// The computed F-statistic for the effect term.
    pub f_value: Array1<f64>,
    /// Significance probability value derived from an F distribution.
    pub p_value: Array1<f64>,
}

impl LmeFit {
    /// Type III fixed-effects ANOVA (backward-compatible default).
    pub fn anova(&self, ddf: DdfMethod) -> crate::Result<FixedEffectsAnovaResult> {
        self.anova_typed(AnovaType::Type3, ddf)
    }

    /// Fixed-effects ANOVA with explicit Type II or Type III contrasts.
    pub fn anova_typed(
        &self,
        anova_type: AnovaType,
        ddf: DdfMethod,
    ) -> crate::Result<FixedEffectsAnovaResult> {
        let fixed_names = self.fixed_names.clone().unwrap_or_default();
        if fixed_names.is_empty() {
            return Err(crate::LmeError::NotImplemented {
                feature: "No fixed effects to test".to_string(),
            });
        }

        let has_intercept = fixed_names[0] == "(Intercept)";
        let start_idx = if has_intercept { 1 } else { 0 };
        if fixed_names.len() <= start_idx {
            return Err(crate::LmeError::NotImplemented {
                feature: "Model contains only an intercept".to_string(),
            });
        }

        let (dfs, _pvals) = match ddf {
            DdfMethod::Satterthwaite => {
                if let Some(res) = &self.satterthwaite {
                    (&res.dfs, &res.p_values)
                } else {
                    return Err(crate::LmeError::NotImplemented {
                        feature: "Satterthwaite values missing. Please ensure the model was evaluated with them tracking on.".to_string()
                    });
                }
            }
            DdfMethod::KenwardRoger => {
                if let Some(res) = &self.kenward_roger {
                    (&res.dfs, &res.p_values)
                } else {
                    return Err(crate::LmeError::NotImplemented {
                        feature: "Kenward-Roger values missing. Please ensure the model evaluated them tracking on.".to_string()
                    });
                }
            }
        };

        let anova_terms = self.grouped_anova_terms(&fixed_names, start_idx)?;
        if anova_terms.is_empty() {
            return Err(crate::LmeError::NotImplemented {
                feature: "No testable fixed-effect terms".to_string(),
            });
        }

        let x = self
            .fixed_design_x
            .as_ref()
            .ok_or_else(|| crate::LmeError::NotImplemented {
                feature: "Fixed-effects design matrix missing; refit with a formula-based model."
                    .to_string(),
            })?;

        let col_terms = self.fixed_term_assign.clone().unwrap_or_else(|| {
            fixed_names
                .iter()
                .map(|n| legacy_term_label(n, &self.categorical_levels))
                .collect()
        });
        let term_names: Vec<String> = anova_terms.iter().map(|t| t.name.clone()).collect();
        let containment = anova_contrasts::term_containment(&term_names);

        let v_beta = if let Some(robust) = &self.robust {
            robust.v_beta_robust.clone()
        } else {
            let xtx_inv = self
                .v_beta_unscaled
                .as_ref()
                .ok_or(crate::LmeError::NotImplemented {
                    feature: "Covariance matrix missing".to_string(),
                })?;
            let sigma2 = self.sigma2.unwrap_or(1.0);
            xtx_inv * sigma2
        };

        let n_terms = anova_terms.len();
        let mut terms = Vec::with_capacity(n_terms);
        let mut num_df = Array1::<f64>::zeros(n_terms);
        let mut den_df = Array1::<f64>::zeros(n_terms);
        let mut f_value = Array1::<f64>::zeros(n_terms);
        let mut p_value = Array1::<f64>::zeros(n_terms);

        for (term_idx, term) in anova_terms.iter().enumerate() {
            terms.push(term.name.clone());
            let l_mat = contrast_for_term(
                anova_type,
                x,
                &col_terms,
                &term.name,
                &term.col_indices,
                &containment,
            );
            let q = l_mat.nrows();
            num_df[term_idx] = q as f64;

            match ddf {
                DdfMethod::Satterthwaite => {
                    if q == 1 {
                        if let Some(idx) = single_unit_contrast_index(&l_mat) {
                            let t_stats =
                                self.beta_t
                                    .as_ref()
                                    .ok_or(crate::LmeError::NotImplemented {
                                        feature: "t-statistics missing".to_string(),
                                    })?;
                            f_value[term_idx] = t_stats[idx] * t_stats[idx];
                            den_df[term_idx] = dfs[idx];
                            p_value[term_idx] = _pvals[idx];
                        } else {
                            let (f_stat, ddf_val, p_val) =
                                run_satterthwaite_contrast(self, &l_mat, &v_beta)?;
                            f_value[term_idx] = f_stat;
                            den_df[term_idx] = ddf_val;
                            p_value[term_idx] = p_val;
                        }
                    } else {
                        let (f_stat, ddf_val, p_val) =
                            run_satterthwaite_contrast(self, &l_mat, &v_beta)?;
                        f_value[term_idx] = f_stat;
                        den_df[term_idx] = ddf_val;
                        p_value[term_idx] = p_val;
                    }
                }
                DdfMethod::KenwardRoger => {
                    let kr = self.kenward_roger.as_ref().ok_or(
                        crate::LmeError::NotImplemented {
                            feature: "Kenward-Roger values missing.".to_string(),
                        },
                    )?;
                    if q == 1 {
                        if let Some(idx) = single_unit_contrast_index(&l_mat) {
                            let t_stats =
                                self.beta_t
                                    .as_ref()
                                    .ok_or(crate::LmeError::NotImplemented {
                                        feature: "t-statistics missing".to_string(),
                                    })?;
                            f_value[term_idx] = t_stats[idx] * t_stats[idx];
                            den_df[term_idx] = dfs[idx];
                            p_value[term_idx] = _pvals[idx];
                        } else {
                            let (f_stat, ddf_val, p_val) = crate::kr_modcomp::kenward_roger_contrast_f_test(
                                &kr.modcomp,
                                &self.coefficients,
                                &l_mat,
                                dfs,
                            )?;
                            f_value[term_idx] = f_stat;
                            den_df[term_idx] = ddf_val;
                            p_value[term_idx] = p_val;
                        }
                    } else {
                        let (f_stat, ddf_val, p_val) = crate::kr_modcomp::kenward_roger_contrast_f_test(
                            &kr.modcomp,
                            &self.coefficients,
                            &l_mat,
                            dfs,
                        )?;
                        f_value[term_idx] = f_stat;
                        den_df[term_idx] = ddf_val;
                        p_value[term_idx] = p_val;
                    }
                }
            }
        }

        Ok(FixedEffectsAnovaResult {
            anova_type,
            method: ddf,
            terms,
            num_df,
            den_df,
            f_value,
            p_value,
        })
    }

    fn grouped_anova_terms(
        &self,
        fixed_names: &[String],
        start_idx: usize,
    ) -> crate::Result<Vec<AnovaTerm>> {
        if let Some(assign) = &self.fixed_term_assign {
            return Ok(anova_contrasts::anova_terms_from_assign(
                fixed_names,
                assign,
            ));
        }

        // Legacy grouping by categorical variable prefix
        let mut grouped_indices: Vec<(String, Vec<usize>)> = Vec::new();
        let mut i = start_idx;
        while i < fixed_names.len() {
            let name = &fixed_names[i];
            let mut matched_categorical = false;
            if let Some(cat_levels) = &self.categorical_levels {
                for cat_var in cat_levels.keys() {
                    if name.starts_with(cat_var) {
                        let mut term_indices = Vec::new();
                        while i < fixed_names.len() && fixed_names[i].starts_with(cat_var) {
                            term_indices.push(i);
                            i += 1;
                        }
                        grouped_indices.push((cat_var.clone(), term_indices));
                        matched_categorical = true;
                        break;
                    }
                }
            }
            if !matched_categorical {
                grouped_indices.push((name.clone(), vec![i]));
                i += 1;
            }
        }
        Ok(grouped_indices
            .into_iter()
            .map(|(name, col_indices)| AnovaTerm { name, col_indices })
            .collect())
    }
}

fn single_unit_contrast_index(l_mat: &ndarray::Array2<f64>) -> Option<usize> {
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

fn run_satterthwaite_contrast(
    fit: &LmeFit,
    l_mat: &ndarray::Array2<f64>,
    v_beta: &ndarray::Array2<f64>,
) -> crate::Result<(f64, f64, f64)> {
    let multi = fit
        .satterthwaite
        .as_ref()
        .and_then(|r| r.multi_dof.as_ref())
        .ok_or(crate::LmeError::NotImplemented {
            feature: "Multi-DoF Satterthwaite requires with_satterthwaite() on the fit."
                .to_string(),
        })?;
    crate::ddf::satterthwaite_contrast_f_test(
        &fit.coefficients,
        v_beta,
        l_mat,
        &multi.jac_vcov,
        &multi.a_mat,
    )
}

fn legacy_term_label(
    name: &str,
    categorical_levels: &Option<std::collections::HashMap<String, Vec<String>>>,
) -> String {
    if let Some(levels) = categorical_levels {
        for cat_var in levels.keys() {
            if name.starts_with(cat_var) {
                return cat_var.clone();
            }
        }
    }
    name.to_string()
}

impl fmt::Display for FixedEffectsAnovaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match self.anova_type {
            AnovaType::Type2 => "II",
            AnovaType::Type3 => "III",
        };
        let title = match self.method {
            DdfMethod::Satterthwaite => format!(
                "Type {} Analysis of Variance Table with Satterthwaite's method",
                type_str
            ),
            DdfMethod::KenwardRoger => format!(
                "Type {} Analysis of Variance Table with Kenward-Roger's method",
                type_str
            ),
        };
        writeln!(f, "{}", title)?;
        writeln!(
            f,
            "{:<15} {:>8}  {:>8}  {:>8}  {:>8}",
            "Term", "NumDF", "DenDF", "F value", "Pr(>F)"
        )?;

        for i in 0..self.terms.len() {
            let p_star = if self.p_value[i] < 0.001 {
                "***"
            } else if self.p_value[i] < 0.01 {
                "**"
            } else if self.p_value[i] < 0.05 {
                "*"
            } else if self.p_value[i] < 0.1 {
                "."
            } else {
                ""
            };

            writeln!(
                f,
                "{:<15} {:>8.0}  {:>8.4}  {:>8.4}  {:>8.4e} {}",
                self.terms[i],
                self.num_df[i],
                self.den_df[i],
                self.f_value[i],
                self.p_value[i],
                p_star
            )?;
        }

        writeln!(
            f,
            "---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )?;
        Ok(())
    }
}
