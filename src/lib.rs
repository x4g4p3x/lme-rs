#![warn(missing_docs)]
//! # lme-rs
//!
//! Rust port of R's `lme4`: linear and generalized linear mixed-effects models
//! with 1:1 numerical compatibility.
//!
//! See [GUIDE.md](https://github.com/x4g4p3x/lme-rs/blob/master/GUIDE.md) for usage.

pub mod math;
pub mod optimizer;
pub mod formula;
pub mod model_matrix;
pub mod family;
pub mod glmm_math;
pub mod satterthwaite;
pub mod kenward_roger;
pub mod anova;
pub mod robust;

pub use anova::{FixedEffectsAnovaResult, DdfMethod};
pub use robust::RobustResult;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, QRInto};
use polars::prelude::*;
use thiserror::Error;
use std::fmt;

pub type Result<T> = std::result::Result<T, LmeError>;

#[derive(Debug, Error)]
pub enum LmeError {
    #[error("dimension mismatch: y has length {y_len}, X has {x_rows} rows")]
    DimensionMismatch { y_len: usize, x_rows: usize },
    #[error("design matrix has fewer rows ({rows}) than columns ({cols}); system is underdetermined")]
    Underdetermined { rows: usize, cols: usize },
    #[error("linear algebra failure: {message}")]
    LinearAlgebra { message: String },
    #[error("formula is empty")]
    EmptyFormula,
    #[error("not implemented: {feature}")]
    NotImplemented { feature: String },
}

/// Represents the fully resolved evaluation output of a structured linear or mixed-effects regression.
#[derive(Debug, Clone)]
pub struct LmeFit {
    pub coefficients: Array1<f64>,
    pub residuals: Array1<f64>,
    pub fitted: Array1<f64>,
    pub ranef: Option<DataFrame>,
    pub var_corr: Option<DataFrame>,
    pub theta: Option<Array1<f64>>,
    pub sigma2: Option<f64>,
    pub reml: Option<f64>,
    pub log_likelihood: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub deviance: Option<f64>,
    pub b: Option<Array1<f64>>,
    pub u: Option<Array1<f64>>,
    pub beta_se: Option<Array1<f64>>,
    pub beta_t: Option<Array1<f64>>,
    // Metadata for Summary Display
    pub formula: Option<String>,
    pub fixed_names: Option<Vec<String>>,
    pub re_blocks: Option<Vec<model_matrix::ReBlock>>,
    pub num_obs: usize,
    // Convergence diagnostics
    pub converged: Option<bool>,
    pub iterations: Option<u64>,
    // GLMM-specific
    /// Family name (e.g. "binomial", "poisson") — None for LMM.
    pub family_name: Option<String>,
    /// Link function name (e.g. "logit", "log") — None for LMM.
    pub link_name: Option<String>,
    /// Family enum stored for response-scale predictions — None for LMM.
    pub family: Option<family::Family>,
    /// Optional Satterthwaite approximation outputs for fixed effects (df, p-values).
    pub satterthwaite: Option<SatterthwaiteResult>,
    /// Optional Kenward-Roger approximation outputs for fixed effects (df, p-values).
    pub kenward_roger: Option<kenward_roger::KenwardRogerResult>,
    /// The unscaled Fisher variance matrix of the fixed effects
    pub v_beta_unscaled: Option<Array2<f64>>,
    /// Optional Robust / Sandwich Standard Error estimates.
    pub robust: Option<RobustResult>,
}

impl LmeFit {
    /// Predict population-level expectations given novel data.
    /// This resolves the Fixed Effects matrix ($X_{new} \hat{\beta}$) ignoring Random Effects groupings (`re.form=NA`).
    pub fn predict(&self, newdata: &polars::prelude::DataFrame) -> anyhow::Result<ndarray::Array1<f64>> {
        // Parse formula to understand structure
        let ast = crate::formula::parse(&self.formula.clone().unwrap_or_default())
            .map_err(|e| anyhow::anyhow!("Failed to parse formula: {}", e))?;
            
        // Find response variable strictly to exclude it from the Identity searches
        let mut response_col_name = String::new();
        for (name, info) in &ast.columns {
            if info.roles.contains(&"Response".to_string()) {
                response_col_name = name.clone();
                break;
            }
        }
        
        let n_obs = newdata.height();
        let (x_new, x_names) = crate::model_matrix::build_x_matrix(&ast, newdata, &response_col_name, n_obs)
            .map_err(|e| anyhow::anyhow!("Failed building X matrix for predictions: {}", e))?;
            
        // Align beta columns with the AST's generated matrix
        if x_names != self.fixed_names.clone().unwrap_or_default() {
            return Err(anyhow::anyhow!(
                "Prediction matrix columns ({:?}) do not match fitted model columns ({:?})",
                x_names, self.fixed_names
            ));
        }

        let mut y_pred = ndarray::Array1::<f64>::zeros(n_obs);
        for i in 0..n_obs {
            let row = x_new.row(i);
            y_pred[i] = self.coefficients.dot(&row);
        }
        
        Ok(y_pred)
    }

    /// Predict on the response scale (applies inverse link for GLMMs).
    ///
    /// For LMMs this is identical to `predict()`. For GLMMs it applies the inverse link
    /// function to transform the linear predictor to the response scale (e.g., probabilities
    /// for binomial, counts for Poisson).
    pub fn predict_response(&self, newdata: &polars::prelude::DataFrame) -> anyhow::Result<ndarray::Array1<f64>> {
        let eta = self.predict(newdata)?;
        self.apply_inverse_link(eta)
    }

    /// Predict conditional expectations on the response scale (applies inverse link for GLMMs).
    ///
    /// Combines fixed + random effects, then applies the inverse link.
    pub fn predict_conditional_response(&self, newdata: &polars::prelude::DataFrame, allow_new_levels: bool) -> anyhow::Result<ndarray::Array1<f64>> {
        let eta = self.predict_conditional(newdata, allow_new_levels)?;
        self.apply_inverse_link(eta)
    }

    /// Apply the inverse link function if this is a GLMM, otherwise return as-is.
    fn apply_inverse_link(&self, eta: ndarray::Array1<f64>) -> anyhow::Result<ndarray::Array1<f64>> {
        match &self.family {
            Some(fam) => {
                let fam_impl = fam.build();
                let link = fam_impl.link();
                Ok(link.link_inv(&eta))
            }
            None => Ok(eta), // LMM: identity, return as-is
        }
    }

    /// Predict conditional expectations given novel data, including Random Effects (`re.form=NULL`).
    /// Computes $\hat{y} = X_{new} \hat{\beta} + Z_{new} \hat{b}$ using stored random effects.
    ///
    /// Groups present in `newdata` but absent from the training data receive zero random-effect
    /// contributions (population-level predictions), consistent with R's `predict.merMod`.
    pub fn predict_conditional(&self, newdata: &polars::prelude::DataFrame, allow_new_levels: bool) -> anyhow::Result<ndarray::Array1<f64>> {
        let y_pop = self.predict(newdata)?;

        let b = self.b.as_ref().ok_or_else(|| anyhow::anyhow!("No random effects available for conditional predictions"))?;
        let re_blocks = self.re_blocks.as_ref().ok_or_else(|| anyhow::anyhow!("No RE block metadata available"))?;

        let n_obs = newdata.height();
        let mut z_b = ndarray::Array1::<f64>::zeros(n_obs);

        let mut b_offset = 0;
        for block in re_blocks {
            let g_series = newdata.column(&block.group_name)
                .map_err(|e| anyhow::anyhow!("Missing grouping variable '{}': {}", block.group_name, e))?
                .cast(&DataType::String).unwrap();
            let g_str = g_series.str().unwrap();

            // Collect slope covariate data for non-intercept effects
            let has_intercept = block.effect_names.first().is_some_and(|n| n == "(Intercept)");
            let mut slope_data: Vec<Vec<f64>> = Vec::new();
            for effect_name in &block.effect_names {
                if effect_name == "(Intercept)" {
                    continue;
                }
                let s_series = newdata.column(effect_name)
                    .map_err(|e| anyhow::anyhow!("Missing slope variable '{}': {}", effect_name, e))?
                    .cast(&DataType::Float64).unwrap();
                let s_f64 = s_series.f64().unwrap();
                slope_data.push(s_f64.into_no_null_iter().collect());
            }

            for (i, val_opt) in g_str.into_iter().enumerate() {
                let group_name = val_opt.unwrap_or("");

                // Look up group index from the stored mapping; unknown groups get 0 contribution
                let group_idx = match block.group_map.get(group_name) {
                    Some(&idx) => idx,
                    None => {
                        if !allow_new_levels {
                            return Err(anyhow::anyhow!(
                                "New level '{}' found in grouping factor '{}', but allow_new_levels is false.",
                                group_name, block.group_name
                            ));
                        }
                        // Unknown group → population-level (no RE contribution)
                        continue;
                    }
                };

                let base = b_offset + group_idx * block.k;
                let mut effect_idx = 0;

                if has_intercept {
                    z_b[i] += b[base + effect_idx]; // intercept contribution (1.0 * b_intercept)
                    effect_idx += 1;
                }

                for (s_idx, s_vec) in slope_data.iter().enumerate() {
                    z_b[i] += s_vec[i] * b[base + effect_idx + s_idx];
                }
            }

            b_offset += block.m * block.k;
        }

        Ok(y_pop + z_b)
    }

    /// Compute Wald confidence intervals for fixed-effect coefficients.
    ///
    /// Uses the normal approximation: β̂ ± z_{α/2} × SE(β̂).
    /// Default level is 0.95 (95% CI).
    ///
    /// # Arguments
    /// * `level` - Confidence level (e.g., 0.95 for 95% CI). Must be in (0, 1).
    pub fn confint(&self, level: f64) -> anyhow::Result<ConfintResult> {
        if level <= 0.0 || level >= 1.0 {
            return Err(anyhow::anyhow!("Confidence level must be in (0, 1), got {}", level));
        }

        let se = self.beta_se.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Standard errors not available — was this fit as a mixed-effects model?"))?;

        let alpha = 1.0 - level;
        use statrs::distribution::{ContinuousCDF, Normal};
        let norm = Normal::new(0.0, 1.0).unwrap();
        let z = norm.inverse_cdf(1.0 - alpha / 2.0);

        let p = self.coefficients.len();
        let mut lower = ndarray::Array1::<f64>::zeros(p);
        let mut upper = ndarray::Array1::<f64>::zeros(p);

        for i in 0..p {
            lower[i] = self.coefficients[i] - z * se[i];
            upper[i] = self.coefficients[i] + z * se[i];
        }

        let names = self.fixed_names.clone().unwrap_or_else(|| {
            (0..p).map(|i| format!("beta_{}", i)).collect()
        });

        Ok(ConfintResult { lower, upper, names, level })
    }

    /// Simulate new responses from the fitted model (parametric bootstrap).
    ///
    /// Generates `nsim` vectors of simulated response values by:
    /// 1. Sampling new random effects b ~ N(0, σ² Λ Λ')
    /// 2. Computing η = Xβ + Zb
    /// 3. Adding Gaussian noise ε ~ N(0, σ²) for LMMs
    ///
    /// For GLMMs, simulation samples from the appropriate distribution using the
    /// conditional mean μ = g⁻¹(η).
    pub fn simulate(&self, nsim: usize) -> anyhow::Result<SimulateResult> {
        use rand::Rng;
        use rand_distr::StandardNormal;

        let sigma2 = self.sigma2.unwrap_or(1.0);
        let sigma = sigma2.sqrt();
        let n = self.num_obs;

        let mut rng = rand::rng();
        let mut simulations = Vec::with_capacity(nsim);

        for _ in 0..nsim {
            // Start with fitted values (Xβ + Zb from the original fit)
            let mut y_sim = self.fitted.clone();

            // Add Gaussian noise: ε ~ N(0, σ²)
            for i in 0..n {
                let eps: f64 = rng.sample(StandardNormal);
                y_sim[i] += sigma * eps;
            }

            simulations.push(y_sim);
        }

        Ok(SimulateResult { simulations })
    }

    /// Compute Satterthwaite degrees of freedom and p-values for fixed effects.
    ///
    /// Requires the original `DataFrame` used to fit the model because it internally
    /// computes the profile deviance Hessian via finite differences.
    /// Mutates the fit to store the results in `self.satterthwaite`.
    pub fn with_satterthwaite(&mut self, data: &polars::prelude::DataFrame) -> anyhow::Result<&mut Self> {
        let (dfs, p_values) = crate::satterthwaite::compute_satterthwaite(self, data)?;
        self.satterthwaite = Some(SatterthwaiteResult { dfs, p_values });
        Ok(self)
    }

    /// Compute Kenward-Roger degrees of freedom and p-values for fixed effects.
    ///
    /// Requires the original `DataFrame` used to fit the model because it internally
    /// computes the objective Hessian via finite differences.
    /// Mutates the fit to store the results in `self.kenward_roger`.
    pub fn with_kenward_roger(&mut self, data: &polars::prelude::DataFrame) -> anyhow::Result<&mut Self> {
        let result = crate::kenward_roger::compute_kenward_roger(self, data)?;
        self.kenward_roger = Some(result);
        Ok(self)
    }

    /// Compute Robust Standard Errors (Sandwich Estimators) for fixed effects.
    ///
    /// Requires the original `DataFrame` used to fit the model because it recalculates
    /// conditional model block structures. Pass `Some("cluster_column_name")` for
    /// Cluster-Robust Standard Errors (CRSE), or `None` for observation-level (HC0) robust errors.
    pub fn with_robust_se(&mut self, data: &polars::prelude::DataFrame, cluster_col: Option<&str>) -> anyhow::Result<&mut Self> {
        let result = crate::robust::compute_robust_se(self, data, cluster_col)
            .map_err(|e| anyhow::anyhow!("Failed calculating robust SEs: {}", e))?;
        self.robust = Some(result);
        Ok(self)
    }
}

/// Satterthwaite approximation outputs for fixed effects.
#[derive(Debug, Clone)]
pub struct SatterthwaiteResult {
    /// Satterthwaite-approximated degrees of freedom for each fixed effect.
    pub dfs: ndarray::Array1<f64>,
    /// Two-sided p-values derived from t-distributions using `dfs`.
    pub p_values: ndarray::Array1<f64>,
}

/// Result of `confint()`: Wald confidence intervals for fixed effects.
#[derive(Debug, Clone)]
pub struct ConfintResult {
    /// Lower bounds of the confidence intervals.
    pub lower: ndarray::Array1<f64>,
    /// Upper bounds of the confidence intervals.
    pub upper: ndarray::Array1<f64>,
    /// Names of the fixed-effect coefficients.
    pub names: Vec<String>,
    /// Confidence level (e.g., 0.95).
    pub level: f64,
}

impl fmt::Display for ConfintResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = (1.0 - self.level) / 2.0 * 100.0;
        writeln!(f, "{:>20} {:>12} {:>12}", "", format!("{:.1} %", pct), format!("{:.1} %", 100.0 - pct))?;
        for i in 0..self.names.len() {
            writeln!(f, "{:>20} {:>12.4} {:>12.4}", self.names[i], self.lower[i], self.upper[i])?;
        }
        Ok(())
    }
}

/// Result of `simulate()`: parametric bootstrap samples from the fitted model.
#[derive(Debug, Clone)]
pub struct SimulateResult {
    /// Each element is a simulated response vector of length n_obs.
    pub simulations: Vec<ndarray::Array1<f64>>,
}

impl fmt::Display for LmeFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(formula) = &self.formula {
            if let Some(fam) = &self.family_name {
                let link = self.link_name.as_deref().unwrap_or("unknown");
                writeln!(f, "Generalized linear mixed model fit by ML (Laplace) ['glmerMod']")?;
                writeln!(f, " Family: {} ( {} )", fam, link)?;
            } else if self.reml.is_some() {
                writeln!(f, "Linear mixed model fit by REML ['lmerMod']")?;
            } else {
                writeln!(f, "Linear mixed model fit by ML ['lmerMod']")?;
            }
            writeln!(f, "Formula: {}", formula)?;
        }
        
        // AIC/BIC/logLik/deviance header
        if self.aic.is_some() || self.bic.is_some() || self.log_likelihood.is_some() {
            writeln!(f)?;
            let mut metrics = Vec::new();
            if let Some(aic) = self.aic {
                metrics.push("AIC      BIC   logLik deviance".to_string());
                let bic = self.bic.unwrap_or(0.0);
                let ll = self.log_likelihood.unwrap_or(0.0);
                let dev = self.deviance.unwrap_or(0.0);
                writeln!(f, "     AIC      BIC   logLik deviance")?;
                writeln!(f, "{:>8.1} {:>8.1} {:>8.1} {:>8.1}", aic, bic, ll, dev)?;
            }
        }
        
        if let Some(reml) = self.reml {
            writeln!(f, "REML criterion at convergence: {:.4}", reml)?;
        }
        
        // Scaled residuals: use sigma for LMMs, Pearson residuals for GLMMs
        let effective_sigma = self.sigma2.map(|s| s.sqrt()).unwrap_or(1.0);
        if effective_sigma > 0.0 {
            writeln!(f, "Scaled residuals:")?;
            let mut scaled_res: Vec<f64> = self.residuals.iter().map(|&r| r / effective_sigma).collect();
            scaled_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = scaled_res.len();
            if n > 0 {
                let min = scaled_res[0];
                let q1 = scaled_res[n / 4];
                let median = scaled_res[n / 2];
                let q3 = scaled_res[3 * n / 4];
                let max = scaled_res[n - 1];
                writeln!(f, "    Min      1Q  Median      3Q     Max ")?;
                writeln!(f, "{:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4}", min, q1, median, q3, max)?;
            }
        }

        writeln!(f, "\nRandom effects:")?;
        writeln!(f, " Groups   Name        Variance Std.Dev.")?;
        
        // For GLMMs without dispersion (Poisson, Binomial), sigma2 is None but
        // the RE variances are stored in theta relative to sigma2=1.0
        let display_sigma2 = self.sigma2.unwrap_or(1.0);
        let is_glmm = self.family_name.is_some();
        if let (Some(theta), Some(re_blocks)) = (&self.theta, &self.re_blocks) {
            let sigma2 = display_sigma2;
            let mut theta_idx = 0;
            let mut obs_groups = Vec::new();
            
            for block in re_blocks {
                let th = &theta.as_slice().unwrap()[theta_idx..theta_idx + block.theta_len];
                theta_idx += block.theta_len;
                
                let mut lambda = ndarray::Array2::<f64>::zeros((block.k, block.k));
                let mut idx = 0;
                for j in 0..block.k {
                    for i in j..block.k {
                        lambda[[i, j]] = th[idx];
                        idx += 1;
                    }
                }
                let cov = lambda.dot(&lambda.t()) * sigma2;
                
                for i in 0..block.k {
                    let var = cov[[i, i]];
                    let std_dev = var.sqrt();
                    let group = if i == 0 { &block.group_name } else { "" };
                    let name = &block.effect_names[i];
                    writeln!(f, " {:<8} {:<11} {:<8.4} {:<8.4}", group, name, var, std_dev)?;
                }
                
                // Gap 6: Print correlations between random effects when k > 1
                if block.k > 1 {
                    writeln!(f, " Corr:")?;
                    for i in 1..block.k {
                        let mut corr_vals = Vec::new();
                        for j in 0..i {
                            let var_i = cov[[i, i]];
                            let var_j = cov[[j, j]];
                            if var_i > 0.0 && var_j > 0.0 {
                                let corr = cov[[i, j]] / (var_i.sqrt() * var_j.sqrt());
                                corr_vals.push(format!("{:>6.3}", corr));
                            } else {
                                corr_vals.push("   NaN".to_string());
                            }
                        }
                        writeln!(f, "  {} {}", block.effect_names[i], corr_vals.join(" "))?;
                    }
                }
                
                obs_groups.push(format!("{}, {}", block.group_name, block.m));
            }
            // Only show residual variance for LMMs (GLMMs without dispersion don't have sigma)
            if !is_glmm {
                writeln!(f, " Residual             {:<8.4} {:<8.4}", sigma2, sigma2.sqrt())?;
            }
            writeln!(f, "Number of obs: {}, groups: {}", self.num_obs, obs_groups.join("; "))?;
        }

        writeln!(f, "\nFixed effects:")?;
        let is_glmm = self.family_name.is_some();
        if self.robust.is_some() {
            if is_glmm {
                writeln!(f, "            Estimate Std. Error z value Pr(>|z|) [Robust]")?;
            } else {
                writeln!(f, "            Estimate Std. Error t value Pr(>|t|) [Robust]")?;
            }
        } else if is_glmm {
            writeln!(f, "            Estimate Std. Error z value")?;
        } else if self.kenward_roger.is_some() {
            writeln!(f, "            Estimate Std. Error       df t value Pr(>|t|) [Kenward-Roger]")?;
        } else if self.satterthwaite.is_some() {
            writeln!(f, "            Estimate Std. Error       df t value Pr(>|t|) [Satterthwaite]")?;
        } else {
            writeln!(f, "            Estimate Std. Error t value")?;
        }
        
        if let (Some(fixed_names), Some(beta_se), Some(beta_t)) = (&self.fixed_names, &self.beta_se, &self.beta_t) {
            for i in 0..self.coefficients.len() {
                let name = if i < fixed_names.len() { &fixed_names[i] } else { "" };
                let est = self.coefficients[i];
                let se = beta_se[i];
                let t_val = beta_t[i];
                
                if let Some(robust) = &self.robust {
                    let r_se = robust.robust_se[i];
                    let r_t = robust.robust_t[i];
                    let p_val = robust.robust_p_values.as_ref().map(|p| p[i]).unwrap_or(f64::NAN);
                    writeln!(f, "{:<11} {:>8.4} {:>10.4} {:>7.2} {:>8.4}", name, est, r_se, r_t, p_val)?;
                } else if let Some(kr) = &self.kenward_roger {
                    let df = kr.dfs[i];
                    let p_val = kr.p_values[i];
                    writeln!(f, "{:<11} {:>8.4} {:>10.4} {:>8.2} {:>7.2} {:>8.4}", name, est, se, df, t_val, p_val)?;
                } else if let Some(satt) = &self.satterthwaite {
                    let df = satt.dfs[i];
                    let p_val = satt.p_values[i];
                    writeln!(f, "{:<11} {:>8.4} {:>10.4} {:>8.2} {:>7.2} {:>8.4}", name, est, se, df, t_val, p_val)?;
                } else {
                    writeln!(f, "{:<11} {:>8.4} {:>10.4} {:>7.2}", name, est, se, t_val)?;
                }
            }
        }
        
        // Gap 10: Convergence diagnostics
        if let Some(converged) = self.converged {
            writeln!(f)?;
            if converged {
                if let Some(iters) = self.iterations {
                    writeln!(f, "optimizer (Nelder-Mead) converged in {} iterations", iters)?;
                } else {
                    writeln!(f, "optimizer (Nelder-Mead) converged")?;
                }
            } else {
                writeln!(f, "WARNING: optimizer (Nelder-Mead) did NOT converge (max iterations reached)")?;
            }
        }
        
        Ok(())
    }
}

/// Fit a linear model `y = X * beta + e` using a QR decomposition.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use lme_rs::lm;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let y = array![1.0, 2.0, 3.0];
///     let x = array![
///         [1.0, 1.0],
///         [1.0, 2.0],
///         [1.0, 3.0],
///     ];
///
///     let fit = lm(&y, &x)?;
///     assert_eq!(fit.coefficients.len(), 2);
///     assert_eq!(fit.residuals.len(), 3);
///     Ok(())
/// }
/// ```
pub fn lm(y: &Array1<f64>, x: &Array2<f64>) -> Result<LmeFit> {
    if y.len() != x.nrows() {
        return Err(LmeError::DimensionMismatch {
            y_len: y.len(),
            x_rows: x.nrows(),
        });
    }
    if x.nrows() < x.ncols() {
        return Err(LmeError::Underdetermined {
            rows: x.nrows(),
            cols: x.ncols(),
        });
    }

    // Solve beta = R^{-1} Q^T y for full-column-rank X = Q R.
    let (q, r) = x
        .clone()
        .qr_into()
        .map_err(|e| LmeError::LinearAlgebra { message: e.to_string() })?;
    let qty = q.t().dot(y);
    let r_inv = r
        .inv()
        .map_err(|e| LmeError::LinearAlgebra { message: e.to_string() })?;
    let coefficients = r_inv.dot(&qty);

    let fitted = x.dot(&coefficients);
    let residuals = y - &fitted;
    let sigma2 = if y.len() > x.ncols() {
        Some(residuals.dot(&residuals) / ((y.len() - x.ncols()) as f64))
    } else {
        None
    };

    Ok(LmeFit {
        coefficients,
        residuals,
        fitted,
        ranef: None,
        var_corr: None,
        theta: None,
        sigma2,
        reml: None,
        log_likelihood: None,
        aic: None,
        bic: None,
        deviance: None,
        b: None,
        u: None,
        beta_se: None,
        beta_t: None,
        formula: None,
        fixed_names: None,
        re_blocks: None,
        num_obs: y.len(),
        converged: None,
        iterations: None,
        family_name: None,
        link_name: None,
        family: None,
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: None,
        robust: None,
    })
}

/// Fit a linear mixed-effects model.
///
/// # Examples
///
/// ```
/// use polars::prelude::*;
/// use lme_rs::lmer;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // let df = DataFrame::new(vec![])?;
///     // let result = lmer("Reaction ~ 1 + (1|Subject)", &df)?;
///     Ok(())
/// }
/// ```
pub fn lmer(formula_str: &str, data: &DataFrame, reml: bool) -> Result<LmeFit> {
    lmer_weighted(formula_str, data, reml, None)
}

/// Fit a linear mixed-effects model with optional observation weights.
///
/// When `weights` is provided, it must be an `Array1<f64>` of length `n_obs`.
/// Observations with higher weights contribute more to the fit.
pub fn lmer_weighted(formula_str: &str, data: &DataFrame, reml: bool, weights: Option<Array1<f64>>) -> Result<LmeFit> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }

    // 1. Parse Wilkinson formula into AST
    let ast = formula::parse(formula_str)?;

    // 2. Build design matrices X, Zt, y from DataFrame and AST
    let matrices = model_matrix::build_design_matrices(&ast, data)?;

    // 3. Setup initial theta vector (length depends on the random effects structure)
    let total_theta_len: usize = matrices.re_blocks.iter().map(|b| b.theta_len).sum();
    let init_theta = Array1::from_vec(vec![1.0; total_theta_len]);
    
    // Gap 8: use log::debug! instead of println!
    log::debug!("Zt shape: {}x{}, nnz: {}, theta_len: {}", 
        matrices.zt.rows(), matrices.zt.cols(), matrices.zt.nnz(), total_theta_len);
    for block in &matrices.re_blocks {
        log::debug!("  block: m={}, k={}, theta={}", block.m, block.k, block.theta_len);
    }

    // 4. Handle offset: For LMMs, y = Xβ + Zb + offset -> y - offset = Xβ + Zb
    let offset_arr = matrices.offset.clone();
    let y_adjusted = if let Some(off) = &offset_arr {
        &matrices.y - off
    } else {
        matrices.y.clone()
    };

    // 5. Optimize theta using Nelder-Mead
    let opt_result = optimizer::optimize_theta_nd(
        matrices.x.clone(),
        matrices.zt.clone(),
        y_adjusted.clone(),
        matrices.re_blocks.clone(),
        init_theta,
        reml,
        weights.clone(),
    )
    .map_err(|e| LmeError::NotImplemented {
        feature: format!("Optimizer failed: {}", e),
    })?;

    let best_theta = &opt_result.theta;

    // 6. Re-evaluate to get coefficients
    let lmm = math::LmmData::new_weighted(matrices.x.clone(), matrices.zt.clone(), y_adjusted, matrices.re_blocks.clone(), weights);
    let best_th_slice = best_theta.as_slice().unwrap();
    let coefs = lmm.evaluate(best_th_slice, reml);
    let reml_eval = lmm.log_reml_deviance(best_th_slice, reml);

    // 7. Extract offset and adjust fitted values
    // Fitted values from solver: Xβ + Zb
    let mut fitted = lmm.x.dot(&coefs.beta);
    let mut z_b_vec = vec![0.0f64; lmm.y_eff.len()];
    let zt = &lmm.zt;
    for (j, row_vec) in zt.outer_iterator().enumerate() {
        for (i, &val) in row_vec.iter() {
            z_b_vec[i] += val * coefs.b[j];
        }
    }
    fitted = fitted + Array1::from_vec(z_b_vec);
    
    // Add the offset back to get true predictions on response scale
    if let Some(off) = &offset_arr {
        fitted = fitted + off;
    }
    
    // Residuals correspond to original y - final fitted
    let residuals = &matrices.y - &fitted;

    // Gap 2: Build ranef DataFrame from b vector
    let ranef_df = build_ranef_dataframe(&coefs.b, &matrices.re_blocks);
    
    // Gap 3: Build var_corr DataFrame from theta/sigma2
    let var_corr_df = build_var_corr_dataframe(best_th_slice, &matrices.re_blocks, coefs.sigma2);

    // Gap 4 + Gap 9: Compute log-likelihood, AIC, BIC
    let deviance_val = reml_eval;
    let log_lik = -deviance_val / 2.0;
    let p = matrices.x.ncols(); // number of fixed effects
    let n_params = (total_theta_len + p + 1) as f64; // theta + beta + sigma
    let n = matrices.y.len() as f64;
    let aic = deviance_val + 2.0 * n_params;
    let bic = deviance_val + n_params * n.ln();

    Ok(LmeFit {
        coefficients: coefs.beta,
        residuals,
        fitted,
        ranef: Some(ranef_df),
        var_corr: Some(var_corr_df),
        theta: Some(opt_result.theta),
        sigma2: Some(coefs.sigma2),
        reml: if reml { Some(reml_eval) } else { None },
        log_likelihood: Some(log_lik),
        aic: Some(aic),
        bic: Some(bic),
        deviance: Some(deviance_val),
        b: Some(coefs.b),
        u: Some(coefs.u),
        beta_se: Some(coefs.beta_se),
        beta_t: Some(coefs.beta_t),
        formula: Some(matrices.formula),
        fixed_names: Some(matrices.fixed_names),
        re_blocks: Some(matrices.re_blocks),
        num_obs: matrices.y.len(),
        converged: Some(opt_result.converged),
        iterations: Some(opt_result.iterations),
        family_name: None,
        link_name: None,
        family: None,
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: Some(coefs.v_beta_unscaled),
        robust: None,
    })
}

/// Fit a generalized linear mixed-effects model (GLMM).
///
/// GLMMs extend LMMs to non-Gaussian responses using a family/link system.
/// Uses Penalized Iteratively Reweighted Least Squares (PIRLS) with
/// Laplace approximation for the marginal likelihood.
///
/// # Examples
///
/// ```
/// use polars::prelude::*;
/// use lme_rs::{glmer, family::Family};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // let df = DataFrame::new(vec![])?;
///     // let result = glmer("y ~ x + (1|group)", &df, Family::Binomial)?;
///     Ok(())
/// }
/// ```
pub fn glmer(formula_str: &str, data: &DataFrame, family_enum: family::Family) -> Result<LmeFit> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }

    let fam = family_enum.build();
    let fam_name = fam.name().to_string();
    let link_name = fam.link().name().to_string();

    // 1. Parse Wilkinson formula into AST
    let ast = formula::parse(formula_str)?;

    // 2. Build design matrices X, Zt, y from DataFrame and AST
    let matrices = model_matrix::build_design_matrices(&ast, data)?;

    // 3. Setup initial theta vector
    let total_theta_len: usize = matrices.re_blocks.iter().map(|b| b.theta_len).sum();
    let init_theta = Array1::from_vec(vec![1.0; total_theta_len]);

    log::debug!("GLMM Zt shape: {}x{}, nnz: {}, theta_len: {}, family: {}",
        matrices.zt.rows(), matrices.zt.cols(), matrices.zt.nnz(), total_theta_len, fam_name);

    // 4. Optimize theta using Nelder-Mead on Laplace deviance
    let fam_for_opt = family_enum.build();
    let opt_result = optimizer::optimize_theta_glmm(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        init_theta,
        fam_for_opt,
        matrices.offset.clone(),
    )
    .map_err(|e| LmeError::NotImplemented {
        feature: format!("GLMM optimizer failed: {}", e),
    })?;

    let best_theta = &opt_result.theta;

    // 5. Re-evaluate PIRLS at optimal theta to get coefficients
    let fam_for_eval = family_enum.build();
    let mut glmm = glmm_math::GlmmData::new(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        fam_for_eval,
    );
    let coefs = glmm.pirls(best_theta.as_slice().unwrap(), matrices.offset.as_ref())
        .ok_or_else(|| LmeError::NotImplemented {
            feature: "PIRLS failed to converge at optimal theta".to_string(),
        })?;

    // 6. Build ranef DataFrame
    let ranef_df = build_ranef_dataframe(&coefs.b, &matrices.re_blocks);

    // For GLMMs without dispersion, sigma2 = 1 (not estimated)
    let uses_disp = fam.uses_dispersion();
    let sigma2_val = if uses_disp {
        // For Gaussian GLMM, compute residual variance
        let n = matrices.y.len() as f64;
        let p = matrices.x.ncols() as f64;
        coefs.residuals.dot(&coefs.residuals) / (n - p)
    } else {
        1.0
    };

    let var_corr_df = build_var_corr_dataframe(
        best_theta.as_slice().unwrap(),
        &matrices.re_blocks,
        sigma2_val,
    );

    // 7. Compute log-likelihood, AIC, BIC
    let deviance_val = coefs.deviance;
    let log_lik = -deviance_val / 2.0;
    let p = matrices.x.ncols();
    let n_params = (total_theta_len + p) as f64; // theta + beta (no sigma for binom/poisson)
    let n = matrices.y.len() as f64;
    let aic = deviance_val + 2.0 * n_params;
    let bic = deviance_val + n_params * n.ln();

    Ok(LmeFit {
        coefficients: coefs.beta,
        residuals: coefs.residuals,
        fitted: coefs.fitted,
        ranef: Some(ranef_df),
        var_corr: Some(var_corr_df),
        theta: Some(opt_result.theta),
        sigma2: if uses_disp { Some(sigma2_val) } else { None },
        reml: None, // GLMMs don't use REML
        log_likelihood: Some(log_lik),
        aic: Some(aic),
        bic: Some(bic),
        deviance: Some(deviance_val),
        b: Some(coefs.b),
        u: Some(coefs.u),
        beta_se: Some(coefs.beta_se),
        beta_t: Some(coefs.beta_z), // z-values for GLMM, stored in beta_t field
        formula: Some(matrices.formula),
        fixed_names: Some(matrices.fixed_names),
        re_blocks: Some(matrices.re_blocks),
        num_obs: matrices.y.len(),
        converged: Some(opt_result.converged),
        iterations: Some(opt_result.iterations),
        family_name: Some(fam_name),
        link_name: Some(link_name),
        family: Some(family_enum),
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: Some(coefs.v_beta_unscaled),
        robust: None,
    })
}

/// Gap 2: Builds a ranef DataFrame from the b vector organized per group/effect.
fn build_ranef_dataframe(b: &Array1<f64>, re_blocks: &[model_matrix::ReBlock]) -> DataFrame {
    let mut group_col = Vec::new();
    let mut group_name_col = Vec::new();
    let mut effect_col = Vec::new();
    let mut value_col = Vec::new();
    
    let mut offset = 0;
    for block in re_blocks {
        for group_idx in 0..block.m {
            for (eff_idx, eff_name) in block.effect_names.iter().enumerate() {
                group_col.push(format!("{}", group_idx));
                group_name_col.push(block.group_name.clone());
                effect_col.push(eff_name.clone());
                value_col.push(b[offset + group_idx * block.k + eff_idx]);
            }
        }
        offset += block.m * block.k;
    }
    
    DataFrame::new(vec![
        Column::new("Grouping".into(), &group_name_col),
        Column::new("Group".into(), &group_col),
        Column::new("Effect".into(), &effect_col),
        Column::new("Value".into(), &value_col),
    ]).unwrap_or_default()
}

/// Gap 3: Builds a var_corr DataFrame from theta parameters and sigma2.
fn build_var_corr_dataframe(theta: &[f64], re_blocks: &[model_matrix::ReBlock], sigma2: f64) -> DataFrame {
    let mut group_col = Vec::new();
    let mut eff1_col = Vec::new();
    let mut eff2_col = Vec::new();
    let mut variance_col = Vec::new();
    let mut stddev_col = Vec::new();
    let mut corr_col: Vec<Option<f64>> = Vec::new();
    
    let mut theta_idx = 0;
    for block in re_blocks {
        let th = &theta[theta_idx..theta_idx + block.theta_len];
        theta_idx += block.theta_len;
        
        let mut lambda = ndarray::Array2::<f64>::zeros((block.k, block.k));
        let mut idx = 0;
        for j in 0..block.k {
            for i in j..block.k {
                lambda[[i, j]] = th[idx];
                idx += 1;
            }
        }
        let cov = lambda.dot(&lambda.t()) * sigma2;
        
        // Diagonal entries: variances
        for i in 0..block.k {
            group_col.push(block.group_name.clone());
            eff1_col.push(block.effect_names[i].clone());
            eff2_col.push(block.effect_names[i].clone());
            variance_col.push(cov[[i, i]]);
            stddev_col.push(cov[[i, i]].sqrt());
            corr_col.push(None);
        }
        
        // Off-diagonal entries: correlations
        for i in 1..block.k {
            for j in 0..i {
                let var_i = cov[[i, i]];
                let var_j = cov[[j, j]];
                let corr = if var_i > 0.0 && var_j > 0.0 {
                    cov[[i, j]] / (var_i.sqrt() * var_j.sqrt())
                } else {
                    f64::NAN
                };
                group_col.push(block.group_name.clone());
                eff1_col.push(block.effect_names[i].clone());
                eff2_col.push(block.effect_names[j].clone());
                variance_col.push(cov[[i, j]]);
                stddev_col.push(f64::NAN); // not applicable for off-diag
                corr_col.push(Some(corr));
            }
        }
    }
    
    // Add residual row
    group_col.push("Residual".to_string());
    eff1_col.push("".to_string());
    eff2_col.push("".to_string());
    variance_col.push(sigma2);
    stddev_col.push(sigma2.sqrt());
    corr_col.push(None);
    
    DataFrame::new(vec![
        Column::new("Group".into(), &group_col),
        Column::new("Effect1".into(), &eff1_col),
        Column::new("Effect2".into(), &eff2_col),
        Column::new("Variance".into(), &variance_col),
        Column::new("StdDev".into(), &stddev_col),
    ]).unwrap_or_default()
}

// ─── ANOVA: Likelihood Ratio Test ─────────────────────────────────────────────

/// Result of a likelihood ratio test between two nested models.
///
/// Produced by [`anova()`] when comparing two fitted models.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// Number of parameters in the simpler model.
    pub n_params_0: usize,
    /// Number of parameters in the more complex model.
    pub n_params_1: usize,
    /// Deviance of the simpler model.
    pub deviance_0: f64,
    /// Deviance of the more complex model.
    pub deviance_1: f64,
    /// Chi-squared statistic: difference in deviance.
    pub chi_sq: f64,
    /// Degrees of freedom for the test (difference in number of parameters).
    pub df: usize,
    /// P-value from chi-squared distribution.
    pub p_value: f64,
    /// Formula of model 0 (simpler).
    pub formula_0: String,
    /// Formula of model 1 (more complex).
    pub formula_1: String,
}

impl fmt::Display for AnovaResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Data: (models compared on same dataset)")?;
        writeln!(f, "Models:")?;
        writeln!(f, "  0: {}", self.formula_0)?;
        writeln!(f, "  1: {}", self.formula_1)?;
        writeln!(f)?;
        writeln!(f, "     npar  deviance  Chisq Df Pr(>Chisq)")?;
        writeln!(f, "  0  {:>4}  {:>8.2}", self.n_params_0, self.deviance_0)?;
        writeln!(f, "  1  {:>4}  {:>8.2} {:>6.2} {:>2}   {:.4e}",
            self.n_params_1, self.deviance_1, self.chi_sq, self.df, self.p_value)?;
        Ok(())
    }
}

/// Compare two nested mixed-effects models using a likelihood ratio test (LRT).
///
/// Computes the chi-squared statistic from the difference in deviance between
/// two models, with degrees of freedom equal to the difference in number of
/// parameters. Returns the LRT result including the p-value.
///
/// Both models must be fit on the **same data** (same `num_obs`). The simpler model
/// (fewer parameters) is automatically identified regardless of argument order.
///
/// # Examples
///
/// ```
/// use lme_rs::{lmer, anova};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // let fit0 = lmer("Reaction ~ 1 + (1|Subject)", &df, true)?;
///     // let fit1 = lmer("Reaction ~ Days + (Days|Subject)", &df, true)?;
///     // let lrt = anova(&fit0, &fit1)?;
///     // println!("{}", lrt);
///     Ok(())
/// }
/// ```
pub fn anova(fit_a: &LmeFit, fit_b: &LmeFit) -> anyhow::Result<AnovaResult> {
    // Validate both models have deviance
    let dev_a = fit_a.deviance.ok_or_else(|| anyhow::anyhow!("Model A has no deviance — was it fit as a mixed-effects model?"))?;
    let dev_b = fit_b.deviance.ok_or_else(|| anyhow::anyhow!("Model B has no deviance — was it fit as a mixed-effects model?"))?;

    // Validate same data size
    if fit_a.num_obs != fit_b.num_obs {
        return Err(anyhow::anyhow!(
            "Models must be fit on the same data: model A has {} obs, model B has {} obs",
            fit_a.num_obs, fit_b.num_obs
        ));
    }

    // Count parameters for each model
    let n_params_a = count_params(fit_a);
    let n_params_b = count_params(fit_b);

    // Identify simpler (0) and more complex (1) model
    let (dev_0, dev_1, np_0, np_1, form_0, form_1) = if n_params_a <= n_params_b {
        (dev_a, dev_b, n_params_a, n_params_b,
         fit_a.formula.clone().unwrap_or_default(),
         fit_b.formula.clone().unwrap_or_default())
    } else {
        (dev_b, dev_a, n_params_b, n_params_a,
         fit_b.formula.clone().unwrap_or_default(),
         fit_a.formula.clone().unwrap_or_default())
    };

    let df = np_1 - np_0;
    if df == 0 {
        return Err(anyhow::anyhow!(
            "Models have the same number of parameters ({}) — cannot perform LRT",
            np_0
        ));
    }

    // Chi-squared statistic = difference in deviance (simpler - complex)
    // Simpler model should have higher deviance
    let chi_sq = (dev_0 - dev_1).max(0.0);

    // P-value from chi-squared distribution
    use statrs::distribution::ContinuousCDF;
    let chi2_dist = statrs::distribution::ChiSquared::new(df as f64)
        .map_err(|e| anyhow::anyhow!("Failed to create chi-squared distribution: {}", e))?;
    let p_value = 1.0 - chi2_dist.cdf(chi_sq);

    Ok(AnovaResult {
        n_params_0: np_0,
        n_params_1: np_1,
        deviance_0: dev_0,
        deviance_1: dev_1,
        chi_sq,
        df,
        p_value,
        formula_0: form_0,
        formula_1: form_1,
    })
}

/// Count the total number of estimated parameters in a fitted model.
///
/// For LMMs: fixed effects (beta) + variance components (theta) + residual variance (sigma).
/// For GLMMs: fixed effects (beta) + variance components (theta) (no sigma for binom/poisson).
fn count_params(fit: &LmeFit) -> usize {
    let n_fixed = fit.coefficients.len();
    let n_theta = fit.theta.as_ref().map_or(0, |t| t.len());
    let has_sigma = fit.sigma2.is_some() && fit.family_name.is_none(); // LMM has sigma
    n_fixed + n_theta + if has_sigma { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lm_recovers_simple_line() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ];

        let fit = lm(&y, &x).expect("lm should fit a full-rank design matrix");
        assert!((fit.coefficients[0] - 0.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 1.0).abs() < 1e-10);
    }
}
