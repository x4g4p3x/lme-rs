//! Profile-likelihood confidence intervals for fixed effects and variance components.
//!
//! For each parameter, constrain that value, re-optimize the remaining free parameters,
//! and find endpoints where the profile deviance equals the MLE deviance plus χ²(1)
//! at the requested level. Wald intervals remain the default via [`LmeFit::confint`].
//!
//! [`LmeFit::confint_profile`] covers fixed effects. [`LmeFit::confint_profile_vc`]
//! adds lme4-style variance-component intervals (`.sig01`… and `.sigma`).

use ndarray::{Array1, Array2};
use polars::prelude::DataFrame;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::sync::Arc;

use crate::family::Link;
use crate::glmm_math::GlmmData;
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

/// Which parameters a profile or bootstrap interval should include.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfintScope {
    /// Fixed-effect coefficients only (default).
    Fixed,
    /// Variance components: `.sig01`… (RE SD `θ·σ(θ)` when θ is scalar; otherwise θ) and `.sigma` for LMMs.
    Variance,
    /// Variance components followed by fixed effects (lme4 `confint.merMod` order).
    All,
}

impl LmeFit {
    /// Profile-likelihood confidence intervals for fixed-effect coefficients.
    ///
    /// Requires the same `data` used to fit the model (formula is read from the fit).
    /// Slower than Wald: each endpoint refits θ many times. Supported for Gaussian LMMs
    /// and GLMMs; not supported for `nlmer`.
    pub fn confint_profile(&self, level: f64, data: &DataFrame) -> anyhow::Result<ConfintResult> {
        profile_confint(self, level, data, None)
    }

    /// Profile-likelihood CIs for a subset of fixed effects (`parms` = 0-based indices).
    ///
    /// When `parms` is empty, all coefficients are profiled (same as [`Self::confint_profile`]).
    /// Subsetting avoids profiling unused coefficients and is the main speed lever.
    pub fn confint_profile_parms(
        &self,
        level: f64,
        data: &DataFrame,
        parms: &[usize],
    ) -> anyhow::Result<ConfintResult> {
        let subset = if parms.is_empty() { None } else { Some(parms) };
        profile_confint(self, level, data, subset)
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

    /// Like [`Self::confint_with`], with optional `parms` for profile (ignored for Wald).
    pub fn confint_with_parms(
        &self,
        level: f64,
        method: ConfintMethod,
        data: Option<&DataFrame>,
        parms: Option<&[usize]>,
    ) -> anyhow::Result<ConfintResult> {
        match method {
            ConfintMethod::Wald => {
                let mut ci = self.confint(level)?;
                if let Some(idxs) = parms {
                    if !idxs.is_empty() {
                        ci = subset_confint(ci, idxs)?;
                    }
                }
                Ok(ci)
            }
            ConfintMethod::Profile => {
                let df = data.ok_or_else(|| {
                    anyhow::anyhow!("confint_with_parms(Profile) requires data=Some(...)")
                })?;
                match parms {
                    Some(idxs) if !idxs.is_empty() => self.confint_profile_parms(level, df, idxs),
                    _ => self.confint_profile(level, df),
                }
            }
        }
    }

    /// Profile-likelihood CIs for variance components (lme4 `oldNames=TRUE` scale).
    ///
    /// For a scalar random intercept, `.sig01` is the RE standard deviation (θ·σ(θ)
    /// along the θ profile, matching lme4) and `.sigma` is the residual SD. Vector θ
    /// is profiled on the relative Cholesky scale as `.sig01`, `.sig02`, …. GLMMs omit
    /// `.sigma` unless the family has a free dispersion. LMM profiles use the ML
    /// deviance even if the reference fit was REML.
    pub fn confint_profile_vc(
        &self,
        level: f64,
        data: &DataFrame,
    ) -> anyhow::Result<ConfintResult> {
        profile_confint_vc(self, level, data)
    }

    /// Profile-likelihood CIs for variance components followed by fixed effects.
    ///
    /// Matches the row order of `lme4::confint.merMod(..., method = "profile")`.
    pub fn confint_profile_all(
        &self,
        level: f64,
        data: &DataFrame,
    ) -> anyhow::Result<ConfintResult> {
        let vc = self.confint_profile_vc(level, data)?;
        let fe = self.confint_profile(level, data)?;
        concat_confint(vc, fe)
    }

    /// Profile-likelihood CIs for [`ConfintScope::Fixed`], [`ConfintScope::Variance`],
    /// or [`ConfintScope::All`].
    pub fn confint_profile_scope(
        &self,
        level: f64,
        data: &DataFrame,
        scope: ConfintScope,
    ) -> anyhow::Result<ConfintResult> {
        match scope {
            ConfintScope::Fixed => self.confint_profile(level, data),
            ConfintScope::Variance => self.confint_profile_vc(level, data),
            ConfintScope::All => self.confint_profile_all(level, data),
        }
    }
}

fn subset_confint(ci: ConfintResult, parms: &[usize]) -> anyhow::Result<ConfintResult> {
    let p = ci.lower.len();
    let mut lower = Vec::with_capacity(parms.len());
    let mut upper = Vec::with_capacity(parms.len());
    let mut names = Vec::with_capacity(parms.len());
    for &j in parms {
        if j >= p {
            return Err(anyhow::anyhow!(
                "confint parms index {j} out of range (p={p})"
            ));
        }
        lower.push(ci.lower[j]);
        upper.push(ci.upper[j]);
        names.push(ci.names[j].clone());
    }
    Ok(ConfintResult {
        lower: Array1::from_vec(lower),
        upper: Array1::from_vec(upper),
        names,
        level: ci.level,
    })
}

fn concat_confint(first: ConfintResult, second: ConfintResult) -> anyhow::Result<ConfintResult> {
    if (first.level - second.level).abs() > 1e-12 {
        return Err(anyhow::anyhow!("confint level mismatch when concatenating"));
    }
    let mut names = first.names;
    names.extend(second.names);
    let mut lower = first.lower.to_vec();
    lower.extend(second.lower.iter().copied());
    let mut upper = first.upper.to_vec();
    upper.extend(second.upper.iter().copied());
    Ok(ConfintResult {
        lower: Array1::from_vec(lower),
        upper: Array1::from_vec(upper),
        names,
        level: first.level,
    })
}

/// Compute profile-likelihood CIs for fixed effects on `fit`.
///
/// When `parms` is `Some`, only those 0-based coefficient indices are profiled
/// (result length matches `parms`).
pub fn profile_confint(
    fit: &LmeFit,
    level: f64,
    data: &DataFrame,
    parms: Option<&[usize]>,
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

    let all_names = fit
        .fixed_names
        .clone()
        .unwrap_or_else(|| (0..p).map(|i| format!("beta_{}", i)).collect());
    let indices: Vec<usize> = match parms {
        Some(idxs) if !idxs.is_empty() => {
            for &j in idxs {
                if j >= p {
                    return Err(anyhow::anyhow!(
                        "confint_profile parms index {j} out of range (p={p})"
                    ));
                }
            }
            idxs.to_vec()
        }
        _ => (0..p).collect(),
    };

    let mut lower = Array1::zeros(indices.len());
    let mut upper = Array1::zeros(indices.len());
    let mut names = Vec::with_capacity(indices.len());

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
        for (out_i, &j) in indices.iter().enumerate() {
            let (lo, hi) = profile_bounds_glmm(
                &prepared,
                j,
                fit.coefficients[j],
                se[j],
                target,
                &init_theta,
            )?;
            lower[out_i] = lo;
            upper[out_i] = hi;
            names.push(all_names[j].clone());
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
        for (out_i, &j) in indices.iter().enumerate() {
            let (lo, hi) = profile_bounds_lmm(
                &prepared,
                j,
                ml_fit.coefficients[j],
                se_ml[j],
                target_ml,
                false,
                &init_theta,
            )?;
            lower[out_i] = lo;
            upper[out_i] = hi;
            names.push(all_names[j].clone());
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

/// Profile-likelihood CIs for variance components.
pub fn profile_confint_vc(
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
            "confint_profile_vc is not supported for nlmer fits"
        ));
    }
    let formula = fit
        .formula
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("confint_profile_vc requires formula on the fit"))?;
    let chi2 = ChiSquared::new(1.0).map_err(|e| anyhow::anyhow!("ChiSquared: {e}"))?;
    let delta = chi2.inverse_cdf(level);

    if let Some(family) = fit.family {
        let link = match fit.link_name.as_deref() {
            Some(name) => Link::parse(name).map_err(|e| anyhow::anyhow!("{e}"))?,
            None => Link::default_for(family),
        };
        let prepared =
            prepare_glmer_weighted_with_link(formula, data, family, link, 1, fit.weights.clone())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        if prepared.matrices.y.len() != fit.num_obs {
            return Err(anyhow::anyhow!(
                "confint_profile_vc: data has {} observations but fit has {}",
                prepared.matrices.y.len(),
                fit.num_obs
            ));
        }
        profile_confint_vc_glmm(fit, &prepared, level, delta)
    } else {
        let prepared = prepare_lmer_weighted(formula, data, fit.weights.clone())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if prepared.lmm.y.len() != fit.num_obs {
            return Err(anyhow::anyhow!(
                "confint_profile_vc: data has {} observations but fit has {}",
                prepared.lmm.y.len(),
                fit.num_obs
            ));
        }
        profile_confint_vc_lmm(&prepared, level, delta)
    }
}

fn profile_confint_vc_lmm(
    prepared: &LmerPrepared,
    level: f64,
    delta: f64,
) -> anyhow::Result<ConfintResult> {
    // Profile under ML so VC intervals are comparable to lme4 `confint` on an ML fit
    // and to the fixed-effect profile path.
    let ml_fit = crate::fit_prepared(prepared, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let theta_hat = ml_fit
        .theta
        .clone()
        .ok_or_else(|| anyhow::anyhow!("confint_profile_vc: ML refit missing theta"))?;
    let sigma2_hat = ml_fit
        .sigma2
        .ok_or_else(|| anyhow::anyhow!("confint_profile_vc: ML refit missing sigma2"))?;
    if !(sigma2_hat > 0.0 && sigma2_hat.is_finite()) {
        return Err(anyhow::anyhow!(
            "confint_profile_vc: non-positive residual variance"
        ));
    }
    let sigma_hat = sigma2_hat.sqrt();
    let lmm = prepared.lmm.as_ref();
    let scalar_re_sd = theta_hat.len() == 1;

    let mut names = Vec::new();
    let mut hats = Vec::new();
    if scalar_re_sd {
        names.push(".sig01".to_string());
        hats.push(theta_hat[0] * sigma_hat);
    } else {
        for i in 0..theta_hat.len() {
            names.push(format!(".sig{:02}", i + 1));
            hats.push(theta_hat[i]);
        }
    }
    names.push(".sigma".to_string());
    hats.push(sigma_hat);

    let mut lower = Array1::zeros(hats.len());
    let mut upper = Array1::zeros(hats.len());

    if scalar_re_sd {
        // lme4 profiles θ (relative Cholesky) and reports `.sig01` = θ·σ(θ)
        // on that profile, not a direct profile of the RE SD.
        let eval_th = |th: f64| -> anyhow::Result<f64> {
            finite_deviance(lmm.log_reml_deviance(&[th], false))
        };
        let d0 = eval_th(theta_hat[0])?;
        let (th_lo, th_hi) =
            find_profile_interval_log(theta_hat[0].max(1e-8), d0 + delta, eval_th)?;
        let mut sd_lo = re_sd_at_theta(lmm, th_lo, false)?;
        let mut sd_hi = re_sd_at_theta(lmm, th_hi, false)?;
        if sd_lo > sd_hi {
            std::mem::swap(&mut sd_lo, &mut sd_hi);
        }
        lower[0] = sd_lo;
        upper[0] = sd_hi;
    } else {
        let bounds = optimizer::compute_theta_lower_bounds(&lmm.re_blocks);
        for j in 0..theta_hat.len() {
            let eval_th = |th: f64| -> anyhow::Result<f64> {
                let d = profile_deviance_theta_held(lmm, j, th, false, &theta_hat);
                finite_deviance(d)
            };
            let d0 = eval_th(theta_hat[j])?;
            let (lo, hi) = if bounds.get(j).copied().unwrap_or(0.0).is_finite()
                && bounds.get(j).copied().unwrap_or(0.0) >= 0.0
                && theta_hat[j] > 0.0
            {
                find_profile_interval_log(theta_hat[j].max(1e-8), d0 + delta, eval_th)?
            } else {
                let se = theta_hat[j].abs().max(0.05);
                find_profile_interval(theta_hat[j], se, d0 + delta, eval_th)?
            };
            lower[j] = lo;
            upper[j] = hi;
        }
    }

    let sigma_idx = hats.len() - 1;
    let eval_sigma = |sig: f64| -> anyhow::Result<f64> {
        let d = profile_deviance_sigma(lmm, sig, false, &theta_hat);
        finite_deviance(d)
    };
    let d0_sigma = eval_sigma(sigma_hat)?;
    let (lo, hi) = find_profile_interval_log(sigma_hat, d0_sigma + delta, eval_sigma)?;
    lower[sigma_idx] = lo;
    upper[sigma_idx] = hi;

    Ok(ConfintResult {
        lower,
        upper,
        names,
        level,
    })
}

fn profile_confint_vc_glmm(
    fit: &LmeFit,
    prepared: &GlmerPrepared,
    level: f64,
    delta: f64,
) -> anyhow::Result<ConfintResult> {
    let theta_hat = fit
        .theta
        .clone()
        .unwrap_or_else(|| prepared.init_theta.clone());
    if theta_hat.is_empty() {
        return Err(anyhow::anyhow!("confint_profile_vc: no theta on GLMM fit"));
    }
    let mut glmm = glmm_data_from_prepared(prepared)?;
    let offset = prepared.matrices.offset.clone();
    let n_agq = prepared.n_agq;
    let bounds = optimizer::compute_theta_lower_bounds(&prepared.matrices.re_blocks);

    let mut names = Vec::with_capacity(theta_hat.len());
    for i in 0..theta_hat.len() {
        names.push(format!(".sig{:02}", i + 1));
    }
    let mut lower = Array1::zeros(theta_hat.len());
    let mut upper = Array1::zeros(theta_hat.len());

    for j in 0..theta_hat.len() {
        let mut eval_th = |th: f64| -> anyhow::Result<f64> {
            let d = profile_deviance_theta_held_glmm(
                &mut glmm,
                j,
                th,
                &theta_hat,
                offset.as_ref(),
                n_agq,
                &bounds,
            );
            finite_deviance(d)
        };
        let hat = theta_hat[j];
        let d0 = eval_th(hat)?;
        let (lo, hi) = if bounds.get(j).copied().unwrap_or(0.0).is_finite()
            && bounds.get(j).copied().unwrap_or(0.0) >= 0.0
            && hat > 0.0
        {
            find_profile_interval_log(hat.max(1e-8), d0 + delta, eval_th)?
        } else {
            let se = hat.abs().max(0.05);
            find_profile_interval(hat, se, d0 + delta, eval_th)?
        };
        lower[j] = lo;
        upper[j] = hi;
    }

    Ok(ConfintResult {
        lower,
        upper,
        names,
        level,
    })
}

fn glmm_data_from_prepared(prepared: &GlmerPrepared) -> anyhow::Result<GlmmData> {
    let fam = prepared
        .family
        .build_with_link(prepared.link)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(GlmmData::from_structural_parts(
        prepared.matrices.x.clone(),
        prepared.matrices.zt.clone(),
        prepared.matrices.y.clone(),
        prepared.matrices.re_blocks.clone(),
        fam,
        prepared.weights.clone(),
        prepared.zt_z.clone(),
        prepared.zt_w_z_map.clone(),
    ))
}

fn finite_deviance(d: f64) -> anyhow::Result<f64> {
    if d.is_finite() {
        Ok(d)
    } else {
        Err(anyhow::anyhow!("profile deviance is non-finite"))
    }
}

fn re_sd_at_theta(lmm: &LmmData, theta: f64, reml: bool) -> anyhow::Result<f64> {
    let coefs = lmm
        .try_evaluate(&[theta], reml)
        .map_err(|_| anyhow::anyhow!("evaluate failed at theta={theta}"))?;
    if coefs.sigma2 > 0.0 && coefs.sigma2.is_finite() {
        Ok(theta * coefs.sigma2.sqrt())
    } else {
        Err(anyhow::anyhow!("non-positive sigma2 at theta={theta}"))
    }
}

fn lmm_unprofiled_deviance(lmm: &LmmData, theta: &[f64], sigma2: f64, reml: bool) -> f64 {
    if !(sigma2 > 0.0 && sigma2.is_finite()) {
        return f64::MAX;
    }
    let Ok(coefs) = lmm.try_evaluate(theta, reml) else {
        return f64::MAX;
    };
    let d_prof = coefs.reml_crit;
    let s2_hat = coefs.sigma2;
    if !(d_prof.is_finite() && s2_hat > 0.0 && s2_hat.is_finite()) {
        return f64::MAX;
    }
    let n = lmm.y.len() as f64;
    let p = lmm.x.ncols() as f64;
    let reml_df = if reml { n - p } else { n };
    let twopi = std::f64::consts::TAU;
    let r2 = s2_hat * reml_df;
    let base_term = d_prof - reml_df * (twopi * s2_hat).ln() - reml_df;
    reml_df * (twopi * sigma2).ln() + base_term + r2 / sigma2
}

fn profile_deviance_sigma(lmm: &LmmData, sigma: f64, reml: bool, theta0: &Array1<f64>) -> f64 {
    if !(sigma > 0.0 && sigma.is_finite()) {
        return f64::MAX;
    }
    let sigma2 = sigma * sigma;
    if theta0.len() == 1 {
        minimize_positive(
            |th| lmm_unprofiled_deviance(lmm, &[th], sigma2, reml),
            theta0[0].max(1e-6),
        )
        .1
    } else {
        min_unprofiled_theta_vec(lmm, sigma2, reml, theta0)
    }
}

fn profile_deviance_theta_held(
    lmm: &LmmData,
    j: usize,
    theta_j: f64,
    reml: bool,
    init: &Array1<f64>,
) -> f64 {
    if init.len() == 1 {
        return lmm.log_reml_deviance(&[theta_j], reml);
    }
    let mut theta = init.clone();
    theta[j] = theta_j;
    let bounds = optimizer::compute_theta_lower_bounds(&lmm.re_blocks);
    for _ in 0..6 {
        for i in 0..init.len() {
            if i == j {
                continue;
            }
            let cur = theta[i];
            let cost = |v: f64| {
                theta[i] = v;
                lmm.log_reml_deviance(theta.as_slice().unwrap(), reml)
            };
            let best = if bounds.get(i).copied().unwrap_or(0.0).is_finite()
                && bounds.get(i).copied().unwrap_or(0.0) >= 0.0
            {
                minimize_positive(cost, cur.max(1e-6)).0
            } else {
                minimize_unbounded(cost, cur).0
            };
            theta[i] = best;
        }
    }
    theta[j] = theta_j;
    lmm.log_reml_deviance(theta.as_slice().unwrap(), reml)
}

fn min_unprofiled_theta_vec(lmm: &LmmData, sigma2: f64, reml: bool, init: &Array1<f64>) -> f64 {
    let mut theta = init.clone();
    let bounds = optimizer::compute_theta_lower_bounds(&lmm.re_blocks);
    for _ in 0..6 {
        for i in 0..init.len() {
            let cur = theta[i];
            let cost = |v: f64| {
                theta[i] = v;
                lmm_unprofiled_deviance(lmm, theta.as_slice().unwrap(), sigma2, reml)
            };
            let best = if bounds.get(i).copied().unwrap_or(0.0).is_finite()
                && bounds.get(i).copied().unwrap_or(0.0) >= 0.0
            {
                minimize_positive(cost, cur.max(1e-6)).0
            } else {
                minimize_unbounded(cost, cur).0
            };
            theta[i] = best;
        }
    }
    lmm_unprofiled_deviance(lmm, theta.as_slice().unwrap(), sigma2, reml)
}

#[allow(clippy::too_many_arguments)]
fn profile_deviance_theta_held_glmm(
    glmm: &mut GlmmData,
    j: usize,
    theta_j: f64,
    init: &Array1<f64>,
    offset: Option<&Array1<f64>>,
    n_agq: usize,
    bounds: &[f64],
) -> f64 {
    if init.len() == 1 {
        return glmm.laplace_deviance(&[theta_j], offset, n_agq);
    }
    let mut theta = init.clone();
    theta[j] = theta_j;
    for _ in 0..6 {
        for i in 0..init.len() {
            if i == j {
                continue;
            }
            let cur = theta[i];
            let cost = |v: f64| {
                theta[i] = v;
                glmm.laplace_deviance(theta.as_slice().unwrap(), offset, n_agq)
            };
            let best = if bounds.get(i).copied().unwrap_or(0.0).is_finite()
                && bounds.get(i).copied().unwrap_or(0.0) >= 0.0
            {
                minimize_positive(cost, cur.max(1e-6)).0
            } else {
                minimize_unbounded(cost, cur).0
            };
            theta[i] = best;
        }
    }
    theta[j] = theta_j;
    glmm.laplace_deviance(theta.as_slice().unwrap(), offset, n_agq)
}

fn minimize_positive<F>(mut f: F, x0: f64) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    const N_GRID: usize = 14;
    const GS_ITERS: u64 = 20;
    let x0 = if x0.is_finite() && x0 > 0.0 { x0 } else { 1.0 };
    let lo0 = (x0 / 24.0).max(1e-8);
    let hi0 = (x0 * 24.0).max(lo0 * 2.0);
    let log_lo = lo0.ln();
    let log_hi = hi0.ln();
    let mut best_x = x0;
    let mut best_f = finite_or_max(f(x0));
    let mut grid_x = Vec::with_capacity(N_GRID);
    let mut grid_f = Vec::with_capacity(N_GRID);
    for i in 0..N_GRID {
        let t = i as f64 / (N_GRID - 1) as f64;
        let x = (log_lo + t * (log_hi - log_lo)).exp();
        let v = finite_or_max(f(x));
        grid_x.push(x);
        grid_f.push(v);
        if v < best_f {
            best_f = v;
            best_x = x;
        }
    }
    let idx = grid_f
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let lo = grid_x[idx.saturating_sub(1)];
    let hi = grid_x[(idx + 1).min(N_GRID - 1)];
    let (x, cost, _) = golden_section_min(&mut f, lo, hi, 1e-6, GS_ITERS);
    if cost < best_f {
        (x, cost)
    } else {
        (best_x, best_f)
    }
}

fn minimize_unbounded<F>(mut f: F, x0: f64) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    let x0 = if x0.is_finite() { x0 } else { 0.0 };
    let span = x0.abs().max(0.5);
    let lo = x0 - 8.0 * span;
    let hi = x0 + 8.0 * span;
    let (x, cost, _) = golden_section_min(&mut f, lo, hi, 1e-6, 24);
    (x, cost)
}

fn golden_section_min<F>(f: &mut F, lo: f64, hi: f64, tol: f64, max_iters: u64) -> (f64, f64, u64)
where
    F: FnMut(f64) -> f64,
{
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        let mid = 0.5 * (lo + hi);
        return (mid, finite_or_max(f(mid)), 1);
    }
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = finite_or_max(f(c));
    let mut fd = finite_or_max(f(d));
    let mut iters = 0u64;
    while (b - a).abs() > tol && iters < max_iters {
        iters += 1;
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / phi;
            fc = finite_or_max(f(c));
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / phi;
            fd = finite_or_max(f(d));
        }
    }
    let mid = 0.5 * (a + b);
    let cost = finite_or_max(f(mid));
    (mid, cost, iters + 1)
}

fn finite_or_max(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::MAX
    }
}

fn find_profile_interval_log<F>(hat: f64, target: f64, mut eval: F) -> anyhow::Result<(f64, f64)>
where
    F: FnMut(f64) -> anyhow::Result<f64>,
{
    let hat = if hat.is_finite() && hat > 0.0 {
        hat
    } else {
        1.0
    };
    let lower = find_one_bound_log(hat, -1.0, target, &mut eval)?;
    let upper = find_one_bound_log(hat, 1.0, target, &mut eval)?;
    if !upper.is_finite() || lower >= upper {
        return Err(anyhow::anyhow!(
            "profile CI failed to bracket positive parameter: lower={lower}, upper={upper}"
        ));
    }
    Ok((lower, upper))
}

fn find_one_bound_log<F>(hat: f64, direction: f64, target: f64, eval: &mut F) -> anyhow::Result<f64>
where
    F: FnMut(f64) -> anyhow::Result<f64>,
{
    let d_hat = eval(hat)?;
    if !d_hat.is_finite() {
        return Err(anyhow::anyhow!("profile deviance at MLE is non-finite"));
    }
    let log_hat = hat.ln();
    let mut log_inner = log_hat;
    let mut step = 0.08_f64;
    let mut found = false;
    let mut log_outer = log_hat;
    const LOG_FLOOR: f64 = -16.0;
    const LOG_CEIL: f64 = 8.0;
    for _ in 0..40 {
        log_outer = log_hat + direction * step;
        if direction < 0.0 && log_outer < LOG_FLOOR {
            log_outer = LOG_FLOOR;
        }
        if direction > 0.0 && log_outer > LOG_CEIL {
            log_outer = LOG_CEIL;
        }
        let d_outer = eval(log_outer.exp())?;
        if d_outer.is_finite() && d_outer >= target {
            found = true;
            break;
        }
        if d_outer.is_finite() && d_outer < target {
            log_inner = log_outer;
        }
        if (direction < 0.0 && log_outer <= LOG_FLOOR + 1e-12)
            || (direction > 0.0 && log_outer >= LOG_CEIL - 1e-12)
        {
            break;
        }
        step *= 1.4;
    }
    if !found {
        if direction < 0.0 {
            return Ok(0.0);
        }
        return Err(anyhow::anyhow!(
            "profile CI: could not find deviance crossing on log scale (direction={direction})"
        ));
    }
    // `inner` is the last point with d < target (toward the MLE);
    // `outer` is the first point with d >= target.
    for _ in 0..60 {
        let mid = 0.5 * (log_inner + log_outer);
        let dm = eval(mid.exp())?;
        if !dm.is_finite() {
            log_outer = mid;
            continue;
        }
        if dm < target {
            log_inner = mid;
        } else {
            log_outer = mid;
        }
        if (log_outer - log_inner).abs() < 1e-6 {
            break;
        }
    }
    Ok((0.5 * (log_inner + log_outer)).exp())
}
