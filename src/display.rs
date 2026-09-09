//! Human-readable fit and interval summaries.
use super::*;
impl fmt::Display for ConfintResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = (1.0 - self.level) / 2.0 * 100.0;
        writeln!(
            f,
            "{:>20} {:>12} {:>12}",
            "",
            format!("{:.1} %", pct),
            format!("{:.1} %", 100.0 - pct)
        )?;
        for i in 0..self.names.len() {
            writeln!(
                f,
                "{:>20} {:>12.4} {:>12.4}",
                self.names[i], self.lower[i], self.upper[i]
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for LmeFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(formula) = &self.formula {
            if let Some(fam) = &self.family_name {
                let link = self.link_name.as_deref().unwrap_or("unknown");
                if self.family == Some(family::Family::Gaussian) {
                    writeln!(f, "Generalized linear mixed model fit by ML ['glmerMod']")?;
                } else {
                    writeln!(
                        f,
                        "Generalized linear mixed model fit by ML (Laplace) ['glmerMod']"
                    )?;
                }
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
            let mut scaled_res: Vec<f64> = self
                .residuals
                .iter()
                .map(|&r| r / effective_sigma)
                .collect();
            scaled_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = scaled_res.len();
            if n > 0 {
                let min = scaled_res[0];
                let q1 = scaled_res[n / 4];
                let median = scaled_res[n / 2];
                let q3 = scaled_res[3 * n / 4];
                let max = scaled_res[n - 1];
                writeln!(f, "    Min      1Q  Median      3Q     Max ")?;
                writeln!(
                    f,
                    "{:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4}",
                    min, q1, median, q3, max
                )?;
            }
        }

        if effective_sigma == 0.0 {
            writeln!(f, "Scaled residuals: unavailable (zero residual variance)")?;
        }
        writeln!(f, "\nRandom effects:")?;
        writeln!(f, " Groups   Name        Variance Std.Dev.")?;

        // For GLMMs, RE variances are ΛΛ′ on the linear-predictor scale (not
        // multiplied by the residual dispersion). LMMs use Var(b) = σ² ΛΛ′.
        let display_sigma2 = self.sigma2.unwrap_or(1.0);
        let re_scale = if self.family.is_some() {
            1.0
        } else {
            display_sigma2
        };
        if let (Some(theta), Some(re_blocks)) = (&self.theta, &self.re_blocks) {
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
                let cov = lambda.dot(&lambda.t()) * re_scale;

                for i in 0..block.k {
                    let var = cov[[i, i]];
                    let std_dev = var.sqrt();
                    let group = if i == 0 { &block.group_name } else { "" };
                    let name = &block.effect_names[i];
                    writeln!(
                        f,
                        " {:<8} {:<11} {:<8.4} {:<8.4}",
                        group, name, var, std_dev
                    )?;
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
            // Residual variance: always for LMMs; for GLMMs only when dispersion is estimated
            // (Gaussian, Gamma, …). Binomial/Poisson fits keep sigma² implicit (None here).
            if self.sigma2.is_some() {
                writeln!(
                    f,
                    " Residual             {:<8.4} {:<8.4}",
                    display_sigma2,
                    display_sigma2.sqrt()
                )?;
            }
            writeln!(
                f,
                "Number of obs: {}, groups: {}",
                self.num_obs,
                obs_groups.join("; ")
            )?;
        }

        writeln!(f, "\nFixed effects:")?;
        let is_glmm = self.family_name.is_some();
        if self.robust.is_some() {
            if is_glmm {
                writeln!(
                    f,
                    "            Estimate Std. Error z value Pr(>|z|) [Robust]"
                )?;
            } else {
                writeln!(
                    f,
                    "            Estimate Std. Error t value Pr(>|t|) [Robust]"
                )?;
            }
        } else if is_glmm {
            writeln!(f, "            Estimate Std. Error z value")?;
        } else if self.kenward_roger.is_some() {
            writeln!(
                f,
                "            Estimate Std. Error       df t value Pr(>|t|) [Kenward-Roger]"
            )?;
        } else if self.satterthwaite.is_some() {
            writeln!(
                f,
                "            Estimate Std. Error       df t value Pr(>|t|) [Satterthwaite]"
            )?;
        } else {
            writeln!(f, "            Estimate Std. Error t value")?;
        }

        if let (Some(fixed_names), Some(beta_se), Some(beta_t)) =
            (&self.fixed_names, &self.beta_se, &self.beta_t)
        {
            for i in 0..self.coefficients.len() {
                let name = if i < fixed_names.len() {
                    &fixed_names[i]
                } else {
                    ""
                };
                let est = self.coefficients[i];
                let se = beta_se[i];
                let t_val = beta_t[i];

                if let Some(robust) = &self.robust {
                    let r_se = robust.robust_se[i];
                    let r_t = robust.robust_t[i];
                    let p_val = robust
                        .robust_p_values
                        .as_ref()
                        .map(|p| p[i])
                        .unwrap_or(f64::NAN);
                    writeln!(
                        f,
                        "{:<11} {:>8.4} {:>10.4} {:>7.2} {:>8.4}",
                        name, est, r_se, r_t, p_val
                    )?;
                } else if let Some(kr) = &self.kenward_roger {
                    let df = kr.dfs[i];
                    let p_val = kr.p_values[i];
                    writeln!(
                        f,
                        "{:<11} {:>8.4} {:>10.4} {:>8.2} {:>7.2} {:>8.4}",
                        name, est, se, df, t_val, p_val
                    )?;
                } else if let Some(satt) = &self.satterthwaite {
                    let df = satt.dfs[i];
                    let p_val = satt.p_values[i];
                    writeln!(
                        f,
                        "{:<11} {:>8.4} {:>10.4} {:>8.2} {:>7.2} {:>8.4}",
                        name, est, se, df, t_val, p_val
                    )?;
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
                    writeln!(f, "optimizer converged in {} iterations", iters)?;
                } else {
                    writeln!(f, "optimizer converged")?;
                }
            } else {
                writeln!(f, "WARNING: optimizer did NOT converge")?;
            }
        }

        if let Some(diagnostics) = &self.diagnostics {
            writeln!(f, "Termination: {:?}", diagnostics.termination)?;
            if diagnostics.requested_n_agq != diagnostics.effective_n_agq {
                writeln!(
                    f,
                    "Quadrature: requested {}, used {}",
                    diagnostics.requested_n_agq, diagnostics.effective_n_agq
                )?;
            }
            for warning in &diagnostics.warnings {
                writeln!(f, "WARNING: {warning}")?;
            }
        }

        Ok(())
    }
}
