//! Profile-likelihood confidence intervals for fixed effects.
//!
//! For each coefficient βⱼ, constrain that coefficient, re-optimize θ (and remaining β),
//! and find endpoints where the profile deviance equals the MLE deviance plus χ²(1)
//! at the requested level. Wald intervals remain the default via [`LmeFit::confint`].

use ndarray::{Array1, Array2};
use polars::prelude::DataFrame;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::sync::Arc;

use crate::family::Link;
use crate::math::LmmData;
use crate::optimizer::{self, OptimizeResult};
use crate::{
    prepare_glmer_weighted_with_link, prepare_lmer_weighted, ConfintResult, GlmerPrepared, LmeFit,
    LmerPrepared,
};

/// Method for [`LmeFit::confint_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfintMethod {
    /// β̂ ± critical × SE (default).
    Wald,
    /// Profile-likelihood intervals (requires original `data`).
    Profile,
}

impl LmeFit {
    /// Profile-likelihood confidence intervals for fixed-effect coefficients.
    ///
    /// Requires the same `data` used to fit the model (formula is read from the fit).
    /// Slower than Wald: each endpoint refits θ many times. Supported for Gaussian LMMs
    /// and GLMMs; not supported for `nlmer`.
    pub fn confint_profile(&self, level: f64, data: &DataFrame) -> anyhow::Result<ConfintResult> {
        profile_confint(self, level, data)
    }

    /// Confidence intervals with an explicit method.
    ///
    /// For [`ConfintMethod::Profile`], `data` is required.
    pub fn confint_with(
        &self,
        level: f64,
        method: ConfintMethod,
        data: Option<&DataFrame>,
    ) -> anyhow::Result<ConfintResult> {
        match method {
            ConfintMethod::Wald => self.confint(level),
            ConfintMethod::Profile => {
                let df = data.ok_or_else(|| {
                    anyhow::anyhow!("confint_with(Profile) requires data=Some(...)")
                })?;
                self.confint_profile(level, df)
            }
        }
    }
}

/// Compute profile-likelihood CIs for all fixed effects on `fit`.
pub fn profile_confint(
    fit: &LmeFit,
    level: f64,
    data: &DataFrame,
) -> anyhow::Result<ConfintResult> {
    if level <= 0.0 || level >= 1.0 {
        return Err(anyhow::anyhow!(
            "Confidence level must be in (0, 1), got {}",
            level
        ));
    }
    if fit.nlmm_mean.is_some() {
        return Err(anyhow::anyhow!(
            "confint_profile is not supported for nlmer fits"
        ));
    }
    let formula = fit
        .formula
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("confint_profile requires formula on the fit"))?;
    let se = fit.beta_se.as_ref().ok_or_else(|| {
        anyhow::anyhow!("confint_profile requires standard errors on the reference fit")
    })?;
    let d0 = fit
        .deviance
        .ok_or_else(|| anyhow::anyhow!("confint_profile requires deviance on the reference fit"))?;
    let p = fit.coefficients.len();
    if p == 0 {
        return Err(anyhow::anyhow!("confint_profile: no fixed effects"));
    }
    if se.len() != p {
        return Err(anyhow::anyhow!(
            "confint_profile: beta_se length does not match coefficients"
        ));
    }

    let names = fit
        .fixed_names
        .clone()
        .unwrap_or_else(|| (0..p).map(|i| format!("beta_{}", i)).collect());

    let mut lower = Array1::zeros(p);
    let mut upper = Array1::zeros(p);

    if let Some(family) = fit.family {
        let chi2 = ChiSquared::new(1.0).map_err(|e| anyhow::anyhow!("ChiSquared: {e}"))?;
        let target = d0 + chi2.inverse_cdf(level);
        let link = match fit.link_name.as_deref() {
            Some(name) => Link::parse(name).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => Link::default_for(family),
        };
        let prepared =
            prepare_glmer_weighted_with_link(formula, data, family, link, 1, fit.weights.clone())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        if prepared.matrices.y.len() != fit.num_obs {
            return Err(anyhow::anyhow!(
                "confint_profile: data has {} observations but fit has {}",
                prepared.matrices.y.len(),
                fit.num_obs
            ));
        }
        let init_theta = fit
            .theta
            .clone()
            .unwrap_or_else(|| prepared.init_theta.clone());
        for j in 0..p {
            let (lo, hi) = profile_bounds_glmm(
                &prepared,
                j,
                fit.coefficients[j],
                se[j],
                target,
                &init_theta,
            )?;
            lower[j] = lo;
            upper[j] = hi;
        }
    } else {
        // Profile fixed effects under ML even if the reference fit used REML:
        // REML criteria are not comparable across models with different `p`.
        let prepared = prepare_lmer_weighted(formula, data, fit.weights.clone())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if prepared.lmm.y.len() != fit.num_obs {
            return Err(anyhow::anyhow!(
                "confint_profile: data has {} observations but fit has {}",
                prepared.lmm.y.len(),
                fit.num_obs
            ));
        }
        let ml_fit = crate::fit_prepared(&prepared, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        let d0_ml = ml_fit
            .deviance
            .ok_or_else(|| anyhow::anyhow!("confint_profile: ML refit missing deviance"))?;
        let chi2 = ChiSquared::new(1.0).map_err(|e| anyhow::anyhow!("ChiSquared: {e}"))?;
        let target_ml = d0_ml + chi2.inverse_cdf(level);
        let init_theta = ml_fit
            .theta
            .clone()
            .unwrap_or_else(|| prepared.init_theta.clone());
        let se_ml = ml_fit.beta_se.as_ref().unwrap_or(se);
        for j in 0..p {
            let (lo, hi) = profile_bounds_lmm(
                &prepared,
                j,
                ml_fit.coefficients[j],
                se_ml[j],
                target_ml,
                false,
                &init_theta,
            )?;
            lower[j] = lo;
            upper[j] = hi;
        }
    }

    Ok(ConfintResult {
        lower,
        upper,
        names,
        level,
    })
}

fn drop_column(x: &Array2<f64>, j: usize) -> Array2<f64> {
    let (n, p) = x.dim();
    assert!(j < p);
    if p == 1 {
        return Array2::zeros((n, 0));
    }
    let mut out = Array2::zeros((n, p - 1));
    for i in 0..n {
        let mut c = 0;
        for k in 0..p {
            if k == j {
                continue;
            }
            out[[i, c]] = x[[i, k]];
            c += 1;
        }
    }
    out
}

fn profile_deviance_lmm(
    prepared: &LmerPrepared,
    j: usize,
    beta_j: f64,
    reml: bool,
    init_theta: &Array1<f64>,
) -> anyhow::Result<f64> {
    let x_full = &prepared.lmm.x;
    let y_full = &prepared.lmm.y;
    let n = y_full.len();
    let mut y_adj = y_full.clone();
    for i in 0..n {
        y_adj[i] -= beta_j * x_full[[i, j]];
    }
    let x_red = drop_column(x_full, j);
    let lmm = Arc::new(LmmData::new_weighted(
        x_red,
        prepared.lmm.zt.clone(),
        y_adj,
        prepared.lmm.re_blocks.clone(),
        prepared.lmm.weights.clone(),
    ));
    let opt: OptimizeResult = optimizer::optimize_theta_lmm(lmm, init_theta.clone(), reml)
        .map_err(|e| anyhow::anyhow!("profile LMM θ optimize failed: {e}"))?;
    Ok(opt.final_cost)
}

fn profile_deviance_glmm(
    prepared: &GlmerPrepared,
    j: usize,
    beta_j: f64,
    init_theta: &Array1<f64>,
) -> anyhow::Result<f64> {
    let x_full = &prepared.matrices.x;
    let n = prepared.matrices.y.len();
    let mut offset = prepared
        .matrices
        .offset
        .clone()
        .unwrap_or_else(|| Array1::zeros(n));
    for i in 0..n {
        offset[i] += beta_j * x_full[[i, j]];
    }
    let x_red = drop_column(x_full, j);
    let fam = prepared
        .family
        .build_with_link(prepared.link)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let opt = optimizer::optimize_theta_glmm_with_maps(
        x_red,
        prepared.matrices.zt.clone(),
        prepared.matrices.y.clone(),
        prepared.matrices.re_blocks.clone(),
        init_theta.clone(),
        fam,
        Some(offset),
        prepared.weights.clone(),
        prepared.zt_z.clone(),
        prepared.zt_w_z_map.clone(),
        1,
    )
    .map_err(|e| anyhow::anyhow!("profile GLMM θ optimize failed: {e}"))?;
    Ok(opt.final_cost)
}

fn profile_bounds_lmm(
    prepared: &LmerPrepared,
    j: usize,
    beta_hat: f64,
    se: f64,
    target: f64,
    reml: bool,
    init_theta: &Array1<f64>,
) -> anyhow::Result<(f64, f64)> {
    let eval = |b: f64| profile_deviance_lmm(prepared, j, b, reml, init_theta);
    find_profile_interval(beta_hat, se, target, eval)
}

fn profile_bounds_glmm(
    prepared: &GlmerPrepared,
    j: usize,
    beta_hat: f64,
    se: f64,
    target: f64,
    init_theta: &Array1<f64>,
) -> anyhow::Result<(f64, f64)> {
    let eval = |b: f64| profile_deviance_glmm(prepared, j, b, init_theta);
    find_profile_interval(beta_hat, se, target, eval)
}

fn find_profile_interval<F>(
    beta_hat: f64,
    se: f64,
    target: f64,
    mut eval: F,
) -> anyhow::Result<(f64, f64)>
where
    F: FnMut(f64) -> anyhow::Result<f64>,
{
    let step0 = if se.is_finite() && se > 0.0 {
        se.max(1e-8)
    } else {
        1.0
    };
    let lower = find_one_bound(beta_hat, -1.0, step0, target, &mut eval)?;
    let upper = find_one_bound(beta_hat, 1.0, step0, target, &mut eval)?;
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(anyhow::anyhow!(
            "profile CI failed to bracket: lower={lower}, upper={upper}"
        ));
    }
    Ok((lower, upper))
}

fn find_one_bound<F>(
    beta_hat: f64,
    direction: f64,
    step0: f64,
    target: f64,
    eval: &mut F,
) -> anyhow::Result<f64>
where
    F: FnMut(f64) -> anyhow::Result<f64>,
{
    let d_hat = eval(beta_hat)?;
    if !d_hat.is_finite() {
        return Err(anyhow::anyhow!(
            "profile deviance at MLE beta is non-finite"
        ));
    }
    let mut inner = beta_hat;
    let mut d_inner = d_hat;
    let mut step = step0;
    let mut outer = beta_hat;
    let mut d_outer = d_hat;
    let mut found = false;
    for _ in 0..40 {
        outer = beta_hat + direction * step;
        d_outer = eval(outer)?;
        if d_outer.is_finite() && d_outer >= target {
            found = true;
            break;
        }
        if d_outer.is_finite() && d_outer < target {
            inner = outer;
            d_inner = d_outer;
        }
        step *= 1.6;
    }
    if !found {
        return Err(anyhow::anyhow!(
            "profile CI: could not find deviance crossing (direction={direction})"
        ));
    }
    let mut a = inner;
    let mut b = outer;
    let mut da = d_inner;
    let mut db = d_outer;
    if a > b {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut da, &mut db);
    }
    for _ in 0..50 {
        let mid = 0.5 * (a + b);
        let dm = eval(mid)?;
        if !dm.is_finite() {
            b = mid;
            continue;
        }
        if dm < target {
            a = mid;
            da = dm;
        } else {
            b = mid;
            db = dm;
        }
        if (b - a).abs() < 1e-5 * step0.max(1e-3) {
            break;
        }
        let _ = (da, db);
    }
    Ok(0.5 * (a + b))
}
