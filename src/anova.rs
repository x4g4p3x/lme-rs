use crate::LmeFit;
use ndarray::Array1;
use std::fmt;

/// Approximation methods for creating denominator degrees of freedom in an ANOVA table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdfMethod {
    Satterthwaite,
    KenwardRoger,
}

/// A structured container for analysis of variance (ANOVA) tests of fixed-effect terms.
///
/// Under `lme-rs`'s strictly continuous/1-DoF design matrix parsing, Type III ANOVA 
/// evaluates independent marginal F-tests for each fixed effect analogous to `lmerTest`.
#[derive(Debug, Clone)]
pub struct FixedEffectsAnovaResult {
    pub method: DdfMethod,
    pub terms: Vec<String>,
    pub num_df: Array1<f64>,
    pub den_df: Array1<f64>,
    pub f_value: Array1<f64>,
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
            return Err(crate::LmeError::NotImplemented { feature: "No fixed effects to test".to_string() });
        }

        // We exclude the Intercept from the ANOVA table by standard convention.
        let has_intercept = fixed_names[0] == "(Intercept)";
        let start_idx = if has_intercept { 1 } else { 0 };
        let n_terms = fixed_names.len() - start_idx;

        if n_terms == 0 {
            return Err(crate::LmeError::NotImplemented { feature: "Model contains only an intercept".to_string() });
        }

        let mut terms = Vec::with_capacity(n_terms);
        let num_df = Array1::<f64>::ones(n_terms); // All strictly 1-DoF in current fiasto parser
        let mut den_df = Array1::<f64>::zeros(n_terms);
        let mut f_value = Array1::<f64>::zeros(n_terms);
        let mut p_value = Array1::<f64>::zeros(n_terms);

        // Fetch DDF based on requested approximation methods
        let (dfs, pvals) = match ddf {
            DdfMethod::Satterthwaite => {
                if let Some(res) = &self.satterthwaite {
                    (&res.dfs, &res.p_values)
                } else {
                    return Err(crate::LmeError::NotImplemented { 
                        feature: "Satterthwaite values missing. Please ensure the model was evaluated with them tracking on.".to_string() 
                    });
                }
            },
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

        let t_stats = self.beta_t.as_ref().ok_or(crate::LmeError::NotImplemented { 
            feature: "t-statistics missing".to_string() 
        })?;

        for i in 0..n_terms {
            let coef_idx = start_idx + i;
            terms.push(fixed_names[coef_idx].clone());
            
            // F = t^2 for 1-DoF contrasts
            f_value[i] = t_stats[coef_idx] * t_stats[coef_idx];
            den_df[i] = dfs[coef_idx];
            p_value[i] = pvals[coef_idx];
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
            DdfMethod::Satterthwaite => "Type III Analysis of Variance Table with Satterthwaite's method",
            DdfMethod::KenwardRoger => "Type III Analysis of Variance Table with Kenward-Roger's method",
        };
        writeln!(f, "{}", title)?;
        writeln!(f, "{:<15} {:>8}  {:>8}  {:>8}  {:>8}", "Term", "NumDF", "DenDF", "F value", "Pr(>F)")?;
        
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
        
        writeln!(f, "---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")?;
        Ok(())
    }
}
