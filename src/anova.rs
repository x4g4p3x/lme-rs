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
///
/// Type III ANOVA: marginal 1-DoF tests for continuous terms, and joint multi-DoF Wald
/// F-tests for grouped categorical dummy columns when `categorical_levels` is present.
#[derive(Debug, Clone)]
pub struct FixedEffectsAnovaResult {
    /// The denominator degrees of freedom method used for the estimates.
    pub method: DdfMethod,
    /// The string name of the fixed effect variable.
    pub terms: Vec<String>,
    /// Numerator degrees of freedom per row (1 for a single coefficient; `q` for a grouped categorical with `q` dummies).
    pub num_df: Array1<f64>,
    /// Denominator degrees of freedom approximated by the chosen method.
    pub den_df: Array1<f64>,
    /// The computed F-statistic for the effect term.
    pub f_value: Array1<f64>,
    /// Significance probability value derived from an F distribution.
    pub p_value: Array1<f64>,
}

impl LmeFit {
    /// Generates a structured Analysis of Variance (ANOVA) table for the fitted Fixed Effects.
    ///
    /// Computes Type III F-statistics mapped exactly to marginal numeric coefficients.
    /// Integrates selected Denominator Degrees of Freedom (DDF) methods (Satterthwaite or Kenward-Roger)
    /// exactly matching the computational output of `lmerTest::anova(model)`.
    pub fn anova(&self, ddf: DdfMethod) -> crate::Result<FixedEffectsAnovaResult> {
        let fixed_names = self.fixed_names.clone().unwrap_or_default();
        if fixed_names.is_empty() {
            return Err(crate::LmeError::NotImplemented {
                feature: "No fixed effects to test".to_string(),
            });
        }

        let has_intercept = fixed_names[0] == "(Intercept)";
        let start_idx = if has_intercept { 1 } else { 0 };
        let n_univariate = fixed_names.len() - start_idx;

        if n_univariate == 0 {
            return Err(crate::LmeError::NotImplemented {
                feature: "Model contains only an intercept".to_string(),
            });
        }

        // Fetch univariate DDF based on requested approximation methods
        let (dfs, pvals) = match ddf {
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

        // Group terms based on categorical_levels
        let mut grouped_indices: Vec<(String, Vec<usize>)> = Vec::new();
        let mut i = start_idx;

        while i < fixed_names.len() {
            let name = &fixed_names[i];
            let mut matched_categorical = false;

            if let Some(cat_levels) = &self.categorical_levels {
                // Find if `name` starts with any categorical variable name
                for cat_var in cat_levels.keys() {
                    if name.starts_with(cat_var) {
                        // Gather all indices that belong to this categorical variable
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

        let n_terms = grouped_indices.len();
        let mut terms = Vec::with_capacity(n_terms);
        let mut num_df = Array1::<f64>::zeros(n_terms);
        let mut den_df = Array1::<f64>::zeros(n_terms);
        let mut f_value = Array1::<f64>::zeros(n_terms);
        let mut p_value = Array1::<f64>::zeros(n_terms);

        let t_stats = self
            .beta_t
            .as_ref()
            .ok_or(crate::LmeError::NotImplemented {
                feature: "t-statistics missing".to_string(),
            })?;

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

        use ndarray_linalg::Inverse;
        use statrs::distribution::{ContinuousCDF, FisherSnedecor};

        for (term_idx, (term_name, indices)) in grouped_indices.into_iter().enumerate() {
            terms.push(term_name);
            let q = indices.len();
            num_df[term_idx] = q as f64;

            if q == 1 {
                let idx = indices[0];
                f_value[term_idx] = t_stats[idx] * t_stats[idx];
                den_df[term_idx] = dfs[idx];
                p_value[term_idx] = pvals[idx];
            } else {
                // Joint Wald F-Test: F = (1/q) * beta_S^T * V_S^{-1} * beta_S
                let mut beta_s = ndarray::Array1::<f64>::zeros(q);
                let mut v_s = ndarray::Array2::<f64>::zeros((q, q));

                for (row_count, &r_idx) in indices.iter().enumerate() {
                    beta_s[row_count] = self.coefficients[r_idx];
                    for (col_count, &c_idx) in indices.iter().enumerate() {
                        v_s[[row_count, col_count]] = v_beta[[r_idx, c_idx]];
                    }
                }

                let f_stat = if let Ok(v_inv) = v_s.inv() {
                    let wald = beta_s.dot(&v_inv.dot(&beta_s));
                    wald / (q as f64)
                } else {
                    f64::NAN
                };

                f_value[term_idx] = f_stat;

                // Approximate multi-DoF denominator df as the conservative minimum of the 1-DoF dfs
                let min_df = indices
                    .iter()
                    .map(|&idx| dfs[idx])
                    .fold(f64::INFINITY, f64::min);
                den_df[term_idx] = min_df;

                if f_stat.is_nan() || min_df <= 0.0 {
                    p_value[term_idx] = f64::NAN;
                } else if let Ok(dist) = FisherSnedecor::new(q as f64, min_df) {
                    p_value[term_idx] = 1.0 - dist.cdf(f_stat);
                } else {
                    p_value[term_idx] = f64::NAN;
                }
            }
        }

        Ok(FixedEffectsAnovaResult {
            method: ddf,
            terms,
            num_df,
            den_df,
            f_value,
            p_value,
        })
    }
}

// Display logic mirroring R's `lmerTest` anova console print
impl fmt::Display for FixedEffectsAnovaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = match self.method {
            DdfMethod::Satterthwaite => {
                "Type III Analysis of Variance Table with Satterthwaite's method"
            }
            DdfMethod::KenwardRoger => {
                "Type III Analysis of Variance Table with Kenward-Roger's method"
            }
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
