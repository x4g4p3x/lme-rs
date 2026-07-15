//! Parametric and residual bootstrap refits for linear mixed models.
//!
//! Mirrors R's `lme4::bootMer` workflow: simulate or resample responses, refit,
//! and summarize bootstrap draws. Uses [`crate::prepare_lmer`] + [`crate::fit_prepared_with_response`]
//! so design-matrix setup is amortized across replicates.

use std::sync::Arc;

use ndarray::Array1;
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::fmt;

use crate::family::{Family, Link};
use crate::simulate;
use crate::{
    fit_prepared_glmer_with_response, fit_prepared_with_response, prepare_glmer_weighted_with_link,
    prepare_lmer, GlmerPrepared, LmeError, LmeFit, LmerPrepared, Result,
};

/// Bootstrap resampling strategy for [`boot_lmer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootLmerMethod {
    /// Draw new Gaussian responses from fitted conditional means (`bootMer` parametric).
    Parametric,
    /// Add resampled residuals to fitted values (`bootMer` residual).
    Residual,
}

/// One bootstrap refit summary.
#[derive(Debug, Clone)]
pub struct BootReplicate {
    /// Zero-based replicate index.
    pub index: usize,
    /// Fixed-effect estimates from this refit.
    pub coefficients: Array1<f64>,
    /// Variance-component Cholesky factors θ (when present).
    pub theta: Option<Array1<f64>>,
    /// Residual variance σ².
    pub sigma2: Option<f64>,
    /// Optimizer convergence flag.
    pub converged: bool,
}

/// Result of [`boot_lmer`].
#[derive(Debug, Clone)]
pub struct BootLmerResult {
    /// Resampling method used.
    pub method: BootLmerMethod,
    /// Number of bootstrap replicates requested.
    pub nsim: usize,
    /// Fixed-effect names (matches coefficient columns).
    pub fixed_names: Vec<String>,
    /// Original (t₀) fixed-effect estimates from the reference fit.
    pub t0: Array1<f64>,
    /// Original σ² from the reference fit.
    pub t0_sigma2: Option<f64>,
    /// Per-replicate refit summaries.
    pub replicates: Vec<BootReplicate>,
    /// Fraction of replicates that converged.
    pub prop_converged: f64,
}

/// Percentile bootstrap confidence intervals for fixed effects.
#[derive(Debug, Clone)]
pub struct BootConfintResult {
    /// Fixed-effect names.
    pub names: Vec<String>,
    /// Point estimates (t₀) from the reference fit.
    pub estimate: Array1<f64>,
    /// Lower percentile limits.
    pub lower: Array1<f64>,
    /// Upper percentile limits.
    pub upper: Array1<f64>,
    /// Confidence level (e.g. `0.95`).
    pub level: f64,
}

impl fmt::Display for BootConfintResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = (1.0 - self.level) / 2.0 * 100.0;
        writeln!(
            f,
            "Bootstrap percentile CIs ({:.0}%):\n{:>20} {:>12} {:>12} {:>12}",
            self.level * 100.0,
            "",
            "Estimate",
            format!("{:.1}%", pct),
            format!("{:.1}%", 100.0 - pct)
        )?;
        for i in 0..self.names.len() {
            writeln!(
                f,
                "{:>20} {:>12.4} {:>12.4} {:>12.4}",
                self.names[i], self.estimate[i], self.lower[i], self.upper[i]
            )?;
        }
        Ok(())
    }
}

impl BootLmerResult {
    /// Percentile confidence intervals from bootstrap coefficient draws.
    ///
    /// Uses only converged replicates. `level` must be in `(0, 1)` (e.g. `0.95`).
    pub fn confint_percentile(&self, level: f64) -> Result<BootConfintResult> {
        if level <= 0.0 || level >= 1.0 {
            return Err(LmeError::NotImplemented {
                feature: format!("Bootstrap confint level must be in (0, 1), got {level}"),
            });
        }
        let p = self.t0.len();
        let converged: Vec<&BootReplicate> = self
            .replicates
            .iter()
            .filter(|r| r.converged && r.coefficients.len() == p)
            .collect();
        if converged.is_empty() {
            return Err(LmeError::NotImplemented {
                feature: "No converged bootstrap replicates for confint".to_string(),
            });
        }

        let alpha = 1.0 - level;
        let lower_q = alpha / 2.0;
        let upper_q = 1.0 - alpha / 2.0;

        let mut lower = Array1::<f64>::zeros(p);
        let mut upper = Array1::<f64>::zeros(p);

        for j in 0..p {
            let mut vals: Vec<f64> = converged.iter().map(|r| r.coefficients[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            lower[j] = percentile_sorted(&vals, lower_q);
            upper[j] = percentile_sorted(&vals, upper_q);
        }

        Ok(BootConfintResult {
            names: self.fixed_names.clone(),
            estimate: self.t0.clone(),
            lower,
            upper,
            level,
        })
    }
}

/// Parametric or residual bootstrap for a fitted LMM.
///
/// # Arguments
/// * `formula_str` - Wilkinson formula (must match the reference fit).
/// * `data` - Original dataset used to fit `fit`.
/// * `fit` - Reference LMM fit (Gaussian; not GLMM/NLMM).
/// * `nsim` - Number of bootstrap replicates.
/// * `method` - [`BootLmerMethod::Parametric`] or [`BootLmerMethod::Residual`].
/// * `reml` - REML (`true`) or ML (`false`) for each bootstrap refit.
/// * `seed` - Optional RNG seed for reproducible resampling.
/// * `n_jobs` - Parallel workers (`None` = all CPUs, capped at `nsim`; `Some(1)` sequential).
#[allow(clippy::too_many_arguments)]
pub fn boot_lmer(
    formula_str: &str,
    data: &DataFrame,
    fit: &LmeFit,
    nsim: usize,
    method: BootLmerMethod,
    reml: bool,
    seed: Option<u64>,
    n_jobs: Option<usize>,
) -> Result<BootLmerResult> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }
    if nsim == 0 {
        return Err(LmeError::NotImplemented {
            feature: "boot_lmer requires nsim >= 1".to_string(),
        });
    }
    if matches!(n_jobs, Some(0)) {
        return Err(LmeError::NotImplemented {
            feature: "boot_lmer requires n_jobs >= 1".to_string(),
        });
    }
    if fit.family.is_some() || fit.nlmm_mean.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "boot_lmer currently supports Gaussian LMMs only".to_string(),
        });
    }

    let fixed_names = fit.fixed_names.clone().unwrap_or_default();
    if fixed_names.is_empty() {
        return Err(LmeError::NotImplemented {
            feature: "boot_lmer requires fixed-effect names on the reference fit".to_string(),
        });
    }
    let p = fit.coefficients.len();
    if p != fixed_names.len() {
        return Err(LmeError::NotImplemented {
            feature: "boot_lmer: coefficient length does not match fixed_names".to_string(),
        });
    }

    let prepared = prepare_lmer(formula_str, data)?;
    if prepared.lmm.y.len() != fit.num_obs {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "boot_lmer: data has {} observations but fit has {}",
                prepared.lmm.y.len(),
                fit.num_obs
            ),
        });
    }

    let bootstrap_y = generate_bootstrap_responses(fit, method, nsim, seed)?;
    let prepared = Arc::new(prepared);
    let workers = resolve_n_jobs(n_jobs, nsim);

    let replicates = if workers == 1 {
        run_bootstrap_sequential(&prepared, &bootstrap_y, reml)
    } else {
        pin_competing_threadpools_single_thread();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("boot_lmer failed to build thread pool: {e}"),
            })?;
        pool.install(|| run_bootstrap_parallel(&prepared, &bootstrap_y, reml, workers))
    };

    let n_conv = replicates.iter().filter(|r| r.converged).count();
    Ok(BootLmerResult {
        method,
        nsim,
        fixed_names,
        t0: fit.coefficients.clone(),
        t0_sigma2: fit.sigma2,
        prop_converged: n_conv as f64 / nsim as f64,
        replicates,
    })
}

/// Parametric bootstrap for a fitted GLMM (`bootMer`-style).
///
/// Residual bootstrap is rejected for discrete families (support-breaking). Gaussian GLMMs
/// should use [`boot_lmer`] via the LMM path. Weighted binomial (proportion + trials) is
/// rejected until simulation grows an explicit trials path.
#[allow(clippy::too_many_arguments)]
pub fn boot_glmer(
    formula_str: &str,
    data: &DataFrame,
    fit: &LmeFit,
    nsim: usize,
    method: BootLmerMethod,
    seed: Option<u64>,
    n_jobs: Option<usize>,
) -> Result<BootLmerResult> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }
    if nsim == 0 {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer requires nsim >= 1".to_string(),
        });
    }
    if matches!(n_jobs, Some(0)) {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer requires n_jobs >= 1".to_string(),
        });
    }
    let family = fit.family.ok_or_else(|| LmeError::NotImplemented {
        feature: "boot_glmer requires a GLMM reference fit (family is set)".to_string(),
    })?;
    if fit.nlmm_mean.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer does not support nlmer fits".to_string(),
        });
    }
    if matches!(method, BootLmerMethod::Residual) {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer residual bootstrap is not supported for GLMMs (use parametric)"
                .to_string(),
        });
    }
    if family == Family::Gaussian {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer: use boot_lmer for Gaussian LMMs".to_string(),
        });
    }
    if family == Family::Binomial {
        let y_obs_binary = fit.residuals.iter().zip(fit.fitted.iter()).all(|(&r, &m)| {
            let y = r + m;
            (y - 0.0).abs() < 1e-9 || (y - 1.0).abs() < 1e-9
        });
        if !y_obs_binary {
            return Err(LmeError::NotImplemented {
                feature: "boot_glmer parametric binomial currently supports 0/1 responses only (not weighted trials)"
                    .to_string(),
            });
        }
    }

    let link = match fit.link_name.as_deref() {
        Some(name) => Link::parse(name)?,
        None => Link::default_for(family),
    };
    let fixed_names = fit.fixed_names.clone().unwrap_or_default();
    if fixed_names.is_empty() {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer requires fixed-effect names on the reference fit".to_string(),
        });
    }
    let p = fit.coefficients.len();
    if p != fixed_names.len() {
        return Err(LmeError::NotImplemented {
            feature: "boot_glmer: coefficient length does not match fixed_names".to_string(),
        });
    }

    let prepared = prepare_glmer_weighted_with_link(formula_str, data, family, link, 1, None)?;
    if prepared.matrices.y.len() != fit.num_obs {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "boot_glmer: data has {} observations but fit has {}",
                prepared.matrices.y.len(),
                fit.num_obs
            ),
        });
    }

    let bootstrap_y = generate_bootstrap_responses(fit, BootLmerMethod::Parametric, nsim, seed)?;
    let prepared = Arc::new(prepared);
    let workers = resolve_n_jobs(n_jobs, nsim);

    let replicates = if workers == 1 {
        run_glmm_bootstrap_sequential(&prepared, &bootstrap_y)
    } else {
        pin_competing_threadpools_single_thread();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("boot_glmer failed to build thread pool: {e}"),
            })?;
        pool.install(|| run_glmm_bootstrap_parallel(&prepared, &bootstrap_y))
    };

    let n_conv = replicates.iter().filter(|r| r.converged).count();
    Ok(BootLmerResult {
        method: BootLmerMethod::Parametric,
        nsim,
        fixed_names,
        t0: fit.coefficients.clone(),
        t0_sigma2: fit.sigma2,
        prop_converged: n_conv as f64 / nsim as f64,
        replicates,
    })
}

fn run_glmm_bootstrap_sequential(
    prepared: &GlmerPrepared,
    bootstrap_y: &[Array1<f64>],
) -> Vec<BootReplicate> {
    bootstrap_y
        .iter()
        .enumerate()
        .map(|(index, y)| run_one_glmm_replicate(prepared, index, y.clone()))
        .collect()
}

fn run_glmm_bootstrap_parallel(
    prepared: &Arc<GlmerPrepared>,
    bootstrap_y: &[Array1<f64>],
) -> Vec<BootReplicate> {
    bootstrap_y
        .par_iter()
        .enumerate()
        .map(|(index, y)| run_one_glmm_replicate(prepared.as_ref(), index, y.clone()))
        .collect()
}

fn run_one_glmm_replicate(prepared: &GlmerPrepared, index: usize, y: Array1<f64>) -> BootReplicate {
    match fit_prepared_glmer_with_response(prepared, Some(y)) {
        Ok(fit) => BootReplicate {
            index,
            coefficients: fit.coefficients,
            theta: fit.theta,
            sigma2: fit.sigma2,
            converged: fit.converged.unwrap_or(false),
        },
        Err(_) => BootReplicate {
            index,
            coefficients: Array1::zeros(prepared.matrices.fixed_names.len()),
            theta: None,
            sigma2: None,
            converged: false,
        },
    }
}

fn generate_bootstrap_responses(
    fit: &LmeFit,
    method: BootLmerMethod,
    nsim: usize,
    seed: Option<u64>,
) -> Result<Vec<Array1<f64>>> {
    match method {
        BootLmerMethod::Parametric => simulate::simulate_range(fit, 0, nsim, Some(1), seed)
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("boot_lmer parametric simulation failed: {e}"),
            }),
        BootLmerMethod::Residual => {
            let n = fit.residuals.len();
            if n == 0 {
                return Err(LmeError::NotImplemented {
                    feature: "boot_lmer residual method requires residuals on the reference fit"
                        .to_string(),
                });
            }
            let mut out = Vec::with_capacity(nsim);
            if let Some(base) = seed {
                for i in 0..nsim {
                    let mut rng = StdRng::seed_from_u64(base.wrapping_add(i as u64));
                    out.push(draw_residual_bootstrap_y(fit, &mut rng));
                }
            } else {
                let mut rng = rand::rng();
                for _ in 0..nsim {
                    out.push(draw_residual_bootstrap_y(fit, &mut rng));
                }
            }
            Ok(out)
        }
    }
}

fn draw_residual_bootstrap_y<R: Rng + ?Sized>(fit: &LmeFit, rng: &mut R) -> Array1<f64> {
    let n = fit.residuals.len();
    let mut y = fit.fitted.clone();
    for i in 0..n {
        let j = rng.random_range(0..n);
        y[i] += fit.residuals[j];
    }
    y
}

fn run_bootstrap_sequential(
    prepared: &LmerPrepared,
    bootstrap_y: &[Array1<f64>],
    reml: bool,
) -> Vec<BootReplicate> {
    bootstrap_y
        .iter()
        .enumerate()
        .map(|(index, y)| run_one_replicate(prepared, index, y.clone(), reml))
        .collect()
}

fn run_bootstrap_parallel(
    prepared: &Arc<LmerPrepared>,
    bootstrap_y: &[Array1<f64>],
    reml: bool,
    _workers: usize,
) -> Vec<BootReplicate> {
    bootstrap_y
        .par_iter()
        .enumerate()
        .map(|(index, y)| run_one_replicate(prepared.as_ref(), index, y.clone(), reml))
        .collect()
}

fn run_one_replicate(
    prepared: &LmerPrepared,
    index: usize,
    y: Array1<f64>,
    reml: bool,
) -> BootReplicate {
    match fit_prepared_with_response(prepared, Some(y), reml) {
        Ok(fit) => BootReplicate {
            index,
            coefficients: fit.coefficients,
            theta: fit.theta,
            sigma2: fit.sigma2,
            converged: fit.converged.unwrap_or(false),
        },
        Err(_) => BootReplicate {
            index,
            coefficients: Array1::zeros(
                prepared
                    .matrices
                    .fixed_names
                    .len()
                    .max(prepared.lmm.x.ncols()),
            ),
            theta: None,
            sigma2: None,
            converged: false,
        },
    }
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn resolve_n_jobs(n_jobs: Option<usize>, n_tasks: usize) -> usize {
    let requested = n_jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    requested.max(1).min(n_tasks.max(1))
}

fn pin_competing_threadpools_single_thread() {
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
}
