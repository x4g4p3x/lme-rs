pub use crate::anova_contrasts::AnovaType;
use crate::anova_contrasts::{self, contrast_for_term, AnovaTerm};
use crate::contrast::fixed_effect_contrast_test;
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
                &term_names,
            );

            let res = fixed_effect_contrast_test(self, &l_mat, ddf, None)?;
            num_df[term_idx] = res.num_df;
            den_df[term_idx] = res.den_df;
            f_value[term_idx] = res.f_value;
            p_value[term_idx] = res.p_value;
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

    /// Wald F-test for one fixed-effect term (`car::linearHypothesis` on a single term name).
    pub fn linear_hypothesis(
        &self,
        term: &str,
        ddf: DdfMethod,
    ) -> crate::Result<crate::contrast::ContrastTestResult> {
        self.linear_hypothesis_terms(&[term], ddf)
    }

    /// Joint Wald F-test for one or more fixed-effect terms by ANOVA term label.
    pub fn linear_hypothesis_terms(
        &self,
        terms: &[&str],
        ddf: DdfMethod,
    ) -> crate::Result<crate::contrast::ContrastTestResult> {
        if terms.is_empty() {
            return Err(crate::LmeError::NotImplemented {
                feature: "linear_hypothesis requires at least one term".to_string(),
            });
        }

        let fixed_names = self.fixed_names.clone().unwrap_or_default();
        let has_intercept = fixed_names.first().is_some_and(|n| n == "(Intercept)");
        let start_idx = if has_intercept { 1 } else { 0 };
        let anova_terms = self.grouped_anova_terms(&fixed_names, start_idx)?;
        let p = self.coefficients.len();

        let mut l_mat = ndarray::Array2::<f64>::zeros((0, p));
        for name in terms {
            let term = anova_terms
                .iter()
                .find(|t| t.name == *name)
                .ok_or_else(|| crate::LmeError::NotImplemented {
                    feature: format!("Unknown fixed-effect term '{name}' in linear_hypothesis"),
                })?;
            let block = anova_contrasts::marginal_contrast(p, &term.col_indices);
            let q = l_mat.nrows();
            let new_q = q + block.nrows();
            let mut stacked = ndarray::Array2::<f64>::zeros((new_q, p));
            if q > 0 {
                stacked.slice_mut(ndarray::s![..q, ..]).assign(&l_mat);
            }
            stacked.slice_mut(ndarray::s![q..new_q, ..]).assign(&block);
            l_mat = stacked;
        }

        fixed_effect_contrast_test(self, &l_mat, ddf, None)
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
            AnovaType::Type1 => "I",
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
