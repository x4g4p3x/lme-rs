//! Nonlinear mixed-model fitting (Laplace / penalized Gauss–Newton, optional scalar AGQ).

use std::sync::Arc;

use crate::model_matrix::ReBlock;
use crate::nlmm::formula::{re_param_indices, NlmerFormula};
use crate::nlmm::mean_fn::{eval_mean_with_re, NlmmMeanEval};
use crate::nlmm::re_cov::{
    log_det_sigma, re_penalty, sigma_from_theta, sigma_inv_from_theta, theta_len,
};
use crate::optimizer::{compute_theta_lower_bounds, nelder_mead_optimize};
use crate::quadrature::{gh_rule, log_sum_exp, resolve_gh_order};
use crate::{LmeError, LmeFit};
use argmin::core::CostFunction;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Solve, UPLO};

/// Starting values for fixed nonlinear parameters (by name).
pub type NlmmStart = std::collections::HashMap<String, f64>;

/// Options for [`fit_nlmer`](crate::nlmm::fit_nlmer).
#[derive(Debug, Clone)]
pub struct NlmerOptions {
    /// Use REML profiling for the variance component (default: ML).
    pub reml: bool,
    /// Starting values for fixed nonlinear parameters.
    pub start: NlmmStart,
    /// Optional lower bounds on **population** nonlinear parameters (by name).
    pub lower: Option<NlmmStart>,
    /// Optional upper bounds on **population** nonlinear parameters (by name).
    pub upper: Option<NlmmStart>,
    /// Maximum penalized Gauss–Newton iterations per RE-variance evaluation.
    pub max_inner: usize,
    /// Reserved for future multi-θ optimizers.
    pub max_outer_iters: u64,
    /// Adaptive Gauss–Hermite quadrature order for scalar random effects (`k = 1`).
    /// `1` (default) uses Laplace only; values `≥ 2` enable AGQ (mirrors `nAGQ` in `lme4`).
    pub n_agq: usize,
}

impl Default for NlmerOptions {
    fn default() -> Self {
        Self {
            reml: false,
            start: NlmmStart::new(),
            lower: None,
            upper: None,
            max_inner: 120,
            max_outer_iters: 500,
            n_agq: 1,
        }
    }
}

struct NlmmProblem {
    y: Array1<f64>,
    x: Array1<f64>,
    group: Vec<usize>,
    m: usize,
    mean: Arc<dyn NlmmMeanEval>,
    param_names: Vec<String>,
    re_indices: Vec<usize>,
    k_re: usize,
    n_fix: usize,
    /// Per-parameter lower bounds (`-∞` when unconstrained).
    lower: Vec<f64>,
    /// Per-parameter upper bounds (`+∞` when unconstrained).
    upper: Vec<f64>,
}

impl NlmmProblem {
    fn project_params(&self, params: &mut [f64]) {
        for (i, p) in params.iter_mut().enumerate() {
            if *p < self.lower[i] {
                *p = self.lower[i];
            } else if *p > self.upper[i] {
                *p = self.upper[i];
            }
        }
    }

    fn b_index(&self, group: usize, re_slot: usize) -> usize {
        group * self.k_re + re_slot
    }

    fn re_offsets_for_group(&self, b: &Array1<f64>, group: usize) -> Vec<f64> {
        (0..self.k_re).map(|r| b[self.b_index(group, r)]).collect()
    }

    fn predict(&self, params: &[f64], b: &Array1<f64>) -> Array1<f64> {
        let n = self.y.len();
        let mut mu = Array1::<f64>::zeros(n);
        for i in 0..n {
            let g = self.group[i];
            let re_off = self.re_offsets_for_group(b, g);
            mu[i] = eval_mean_with_re(
                self.mean.as_ref(),
                self.x[i],
                params,
                &self.re_indices,
                &re_off,
            )
            .0;
        }
        mu
    }

    fn penalized_rss(&self, params: &[f64], b: &Array1<f64>, theta: &[f64]) -> f64 {
        let mu = self.predict(params, b);
        let rss: f64 = self
            .y
            .iter()
            .zip(mu.iter())
            .map(|(&y, &m)| (y - m).powi(2))
            .sum();
        let mut b_pen = 0.0;
        for g in 0..self.m {
            let bg: Vec<f64> = (0..self.k_re).map(|r| b[self.b_index(g, r)]).collect();
            b_pen += re_penalty(self.k_re, theta, &bg);
        }
        rss + b_pen
    }

    fn random_effect_logdet(&self, params: &[f64], b: &Array1<f64>, theta: &[f64]) -> f64 {
        let n = self.y.len();
        let q = self.m * self.k_re;
        let mut j = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            let g = self.group[i];
            let re_off = self.re_offsets_for_group(b, g);
            let (_, grad) = eval_mean_with_re(
                self.mean.as_ref(),
                self.x[i],
                params,
                &self.re_indices,
                &re_off,
            );
            for r_slot in 0..self.k_re {
                let param_idx = self.re_indices[r_slot];
                j[[i, self.b_index(g, r_slot)]] = grad[param_idx];
            }
        }

        let mut h = j.t().dot(&j);
        let inv = sigma_inv_from_theta(self.k_re, theta);
        for g in 0..self.m {
            for r in 0..self.k_re {
                for s in 0..self.k_re {
                    let row = self.b_index(g, r);
                    let col = self.b_index(g, s);
                    h[[row, col]] += inv[[r, s]];
                }
            }
        }

        match h.cholesky(UPLO::Lower) {
            Ok(chol) => {
                let logdet_h: f64 = (0..q).map(|i| chol[[i, i]].max(1e-12).ln()).sum::<f64>() * 2.0;
                logdet_h + self.m as f64 * log_det_sigma(self.k_re, theta)
            }
            Err(_) => f64::INFINITY,
        }
    }

    fn add_re_prior_terms(
        &self,
        jtj: &mut Array2<f64>,
        rhs: &mut Array1<f64>,
        b: &Array1<f64>,
        theta: &[f64],
        col_offset: usize,
    ) {
        let inv = sigma_inv_from_theta(self.k_re, theta);
        for g in 0..self.m {
            for r in 0..self.k_re {
                let row = col_offset + self.b_index(g, r);
                let mut prior_grad = 0.0;
                for s in 0..self.k_re {
                    let col = col_offset + self.b_index(g, s);
                    jtj[[row, col]] += inv[[r, s]];
                    prior_grad += inv[[r, s]] * b[self.b_index(g, s)];
                }
                // Preserve the established scalar nlmer profiling path; the coupled
                // prior-gradient correction is needed for correlated multivariate RE.
                if self.k_re > 1 {
                    rhs[row] -= prior_grad;
                }
            }
        }
    }

    fn inner_gauss_newton(
        &self,
        theta: &[f64],
        start: &NlmmStart,
        max_iter: usize,
    ) -> (Vec<f64>, Array1<f64>, f64) {
        let mut params: Vec<f64> = self.mean.default_start_values(&self.param_names);
        for (name, value) in start {
            if let Some(idx) = self.param_names.iter().position(|n| n == name) {
                params[idx] = *value;
            }
        }
        if self.mean.needs_positive_scal() {
            let scal_idx = self
                .param_names
                .iter()
                .position(|n| n == "scal")
                .unwrap_or(2);
            if params[scal_idx].abs() < 1e-8 {
                params[scal_idx] = 350.0;
            }
        }
        self.project_params(&mut params);

        let mut b = Array1::<f64>::zeros(self.m * self.k_re);
        let n = self.y.len();
        let p_fix = self.n_fix;
        let p = p_fix + self.m * self.k_re;
        let mut lambda_lm = if self.k_re == 1 && self.mean.uses_scalar_rss_sigma() {
            1e-2
        } else {
            1e-4
        };
        for _ in 0..max_iter {
            let mut j = Array2::<f64>::zeros((n, p));
            let mut r = Array1::<f64>::zeros(n);

            for i in 0..n {
                let g = self.group[i];
                let re_off = self.re_offsets_for_group(&b, g);
                let (mui, grad) = eval_mean_with_re(
                    self.mean.as_ref(),
                    self.x[i],
                    &params,
                    &self.re_indices,
                    &re_off,
                );
                r[i] = self.y[i] - mui;
                for j_fix in 0..p_fix {
                    j[[i, j_fix]] = grad[j_fix];
                }
                for r_slot in 0..self.k_re {
                    let param_idx = self.re_indices[r_slot];
                    j[[i, p_fix + self.b_index(g, r_slot)]] = grad[param_idx];
                }
            }

            let mut jtj = j.t().dot(&j);
            let mut rhs = j.t().dot(&r);
            self.add_re_prior_terms(&mut jtj, &mut rhs, &b, theta, p_fix);
            let old_obj = self.penalized_rss(&params, &b, theta);

            let mut accepted = false;
            let mut step_norm = 0.0f64;
            for _attempt in 0..12 {
                let mut damped = jtj.clone();
                for i in 0..p {
                    damped[[i, i]] += lambda_lm * damped[[i, i]].max(1e-8);
                }
                let delta = match damped.cholesky(UPLO::Lower).and_then(|c| c.solve(&rhs)) {
                    Ok(d) => d,
                    Err(_) => break,
                };

                let mut alpha = 1.0;
                while alpha >= 1e-6 {
                    let mut new_params = params.clone();
                    for j_fix in 0..p_fix {
                        new_params[j_fix] += alpha * delta[j_fix];
                    }
                    if self.mean.needs_positive_scal() {
                        if let Some(scal_idx) = self.param_names.iter().position(|n| n == "scal") {
                            if new_params[scal_idx] < 1e-6 {
                                new_params[scal_idx] = 1e-6;
                            }
                        }
                    }
                    self.project_params(&mut new_params);
                    let mut nb = b.clone();
                    for g in 0..self.m {
                        for r_slot in 0..self.k_re {
                            let col = p_fix + self.b_index(g, r_slot);
                            nb[self.b_index(g, r_slot)] += alpha * delta[col];
                        }
                    }
                    let new_obj = self.penalized_rss(&new_params, &nb, theta);
                    if new_obj < old_obj {
                        step_norm = (alpha * delta).iter().map(|v| v.abs()).fold(0.0, f64::max);
                        params = new_params;
                        b = nb;
                        lambda_lm = (lambda_lm * 0.3).max(1e-8);
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if accepted {
                    break;
                }
                lambda_lm = (lambda_lm * 10.0).min(1e6);
            }
            if !accepted {
                break;
            }
            if step_norm < 1e-10 {
                break;
            }
        }

        let mu = self.predict(&params, &b);
        let rss: f64 = self
            .y
            .iter()
            .zip(mu.iter())
            .map(|(&y, &m)| (y - m).powi(2))
            .sum();
        (params, b, rss)
    }

    fn profile_objective(
        &self,
        theta: &[f64],
        start: &NlmmStart,
        reml: bool,
        max_inner: usize,
        n_agq: usize,
    ) -> (f64, Vec<f64>, f64, Array1<f64>) {
        let (params, b, rss) = self.inner_gauss_newton(theta, start, max_inner);
        let n = self.y.len() as f64;
        let p = self.n_fix as f64;
        let mut b_pen = 0.0;
        for g in 0..self.m {
            let bg: Vec<f64> = (0..self.k_re).map(|r| b[self.b_index(g, r)]).collect();
            b_pen += re_penalty(self.k_re, theta, &bg);
        }
        let pwrss = rss + b_pen;
        let df = if reml { (n - p).max(1.0) } else { n };
        let scalar_ssasymp = self.k_re == 1 && self.mean.uses_scalar_rss_sigma();
        let sigma2 = if scalar_ssasymp {
            (rss / df).max(1e-12)
        } else {
            (pwrss / df).max(1e-12)
        };
        let re_logdet = if self.k_re > 1 {
            self.random_effect_logdet(&params, &b, theta)
        } else {
            self.m as f64 * log_det_sigma(self.k_re, theta)
        };
        let mut crit = if scalar_ssasymp {
            let twopi = std::f64::consts::PI * 2.0;
            let mut crit = df * (twopi * sigma2).ln() + rss / sigma2 + b_pen + re_logdet;
            if reml {
                crit += (self.m as f64 - p) * (1.0 + sigma2.ln());
            }
            crit
        } else if reml {
            let twopi = std::f64::consts::PI * 2.0;
            df * (twopi * sigma2).ln()
                + pwrss / sigma2
                + re_logdet
                + (self.m as f64 * self.k_re as f64 - p) * (1.0 + sigma2.ln())
        } else {
            n * pwrss.ln() + re_logdet
        };
        if let Some(agq_q) = self.agq_correction(n_agq, &params, &b, theta, sigma2) {
            crit += agq_q;
        }
        (crit, params, sigma2, b)
    }

    /// Scalar AGQ correction to the deviance (`k = 1` random effects only).
    fn agq_correction(
        &self,
        n_agq: usize,
        params: &[f64],
        b: &Array1<f64>,
        theta: &[f64],
        sigma2: f64,
    ) -> Option<f64> {
        if n_agq < 2 || self.k_re != 1 {
            return None;
        }
        let order = resolve_gh_order(n_agq)?;
        let (z, w) = gh_rule(order)?;
        let inv_sigma = sigma_inv_from_theta(1, theta)[[0, 0]];

        let mut group_obs: Vec<Vec<usize>> = vec![vec![]; self.m];
        for (i, &g) in self.group.iter().enumerate() {
            group_obs[g].push(i);
        }

        let mut total_q = 0.0;
        for g in 0..self.m {
            let b_hat_g = b[g];
            let obs_idx = &group_obs[g];
            if obs_idx.is_empty() {
                continue;
            }

            let mut a_gg = inv_sigma;
            for &i in obs_idx {
                let re_off = self.re_offsets_for_group(b, g);
                let (_, grad) = eval_mean_with_re(
                    self.mean.as_ref(),
                    self.x[i],
                    params,
                    &self.re_indices,
                    &re_off,
                );
                let dmu_db = grad[self.re_indices[0]];
                a_gg += dmu_db * dmu_db / sigma2;
            }
            if a_gg <= f64::EPSILON || !a_gg.is_finite() {
                return None;
            }
            let scale = a_gg.sqrt().recip();

            let mu_hat = self.predict(params, b);
            let mut rss_g_hat = 0.0;
            for &i in obs_idx {
                let r = self.y[i] - mu_hat[i];
                rss_g_hat += r * r;
            }

            let mut log_terms = Vec::with_capacity(z.len());
            let mut b_trial = b.clone();
            for (ki, &z_k) in z.iter().enumerate() {
                let b_quad = b_hat_g + scale * z_k;
                b_trial[g] = b_quad;
                let mu_quad = self.predict(params, &b_trial);
                let mut rss_g_quad = 0.0;
                for &i in obs_idx {
                    let r = self.y[i] - mu_quad[i];
                    rss_g_quad += r * r;
                }
                let rss_diff = (rss_g_quad - rss_g_hat) / sigma2;
                let pen_quad = re_penalty(1, theta, &[b_quad]);
                let pen_hat = re_penalty(1, theta, &[b_hat_g]);
                let pen_diff = pen_quad - pen_hat;
                let delta = -0.5 * rss_diff - 0.5 * pen_diff;
                log_terms.push(w[ki].ln() + delta);
            }
            let log_inner = log_sum_exp(&log_terms);
            if !log_inner.is_finite() {
                return None;
            }
            total_q += -2.0 * log_inner;
        }
        Some(total_q)
    }
}

/// Golden-section search for a scalar Cholesky diagonal (k = 1).
fn optimize_theta_golden(
    problem: &NlmmProblem,
    start: &NlmmStart,
    reml: bool,
    max_inner: usize,
    n_agq: usize,
    lo: f64,
    hi: f64,
) -> (f64, f64, u64) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = problem.profile_objective(&[c], start, reml, max_inner, n_agq);
    let mut fd = problem.profile_objective(&[d], start, reml, max_inner, n_agq);
    let mut fc_cost = fc.0;
    let mut fd_cost = fd.0;
    let mut iters = 0u64;
    while (b - a).abs() > 1e-4 && iters < 80 {
        iters += 1;
        if fc_cost < fd_cost {
            b = d;
            d = c;
            fd = fc;
            fd_cost = fc_cost;
            c = b - (b - a) / phi;
            fc = problem.profile_objective(&[c], start, reml, max_inner, n_agq);
            fc_cost = fc.0;
        } else {
            a = c;
            c = d;
            fc = fd;
            fc_cost = fd_cost;
            d = a + (b - a) / phi;
            fd = problem.profile_objective(&[d], start, reml, max_inner, n_agq);
            fd_cost = fd.0;
        }
    }
    let theta0 = (a + b) / 2.0;
    let final_cost = problem
        .profile_objective(&[theta0], start, reml, max_inner, n_agq)
        .0;
    (theta0, final_cost, iters)
}

struct ThetaObjective<'a> {
    problem: &'a NlmmProblem,
    start: &'a NlmmStart,
    reml: bool,
    max_inner: usize,
    n_agq: usize,
}

fn clamp_nlmm_theta(theta: &mut [f64], k: usize) {
    let lower = theta_lower_bounds(k);
    let mut idx = 0usize;
    for col in 0..k {
        for row in col..k {
            theta[idx] = theta[idx].max(lower[idx]);
            if row == col {
                theta[idx] = theta[idx].min(12.0);
            } else {
                theta[idx] = theta[idx].clamp(-10.0, 10.0);
            }
            idx += 1;
        }
    }
}

impl CostFunction for ThetaObjective<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut th = theta.as_slice().unwrap().to_vec();
        clamp_nlmm_theta(&mut th, self.problem.k_re);
        let cost = self
            .problem
            .profile_objective(&th, self.start, self.reml, self.max_inner, self.n_agq)
            .0;
        if cost.is_finite() {
            Ok(cost)
        } else {
            Ok(f64::MAX)
        }
    }
}

fn theta_lower_bounds(k: usize) -> Vec<f64> {
    let mut bounds = compute_theta_lower_bounds(&[ReBlock {
        m: 1,
        k,
        theta_len: theta_len(k),
        group_name: String::new(),
        effect_names: vec![],
        group_map: Default::default(),
    }]);
    // Keep Cholesky diagonals off zero (singular Λ).
    let mut idx = 0usize;
    for col in 0..k {
        bounds[idx] = bounds[idx].max(0.05);
        idx += k - col;
    }
    bounds
}

fn optimize_theta_nelder_mead(
    problem: &NlmmProblem,
    start: &NlmmStart,
    reml: bool,
    max_inner: usize,
    n_agq: usize,
    init: Array1<f64>,
    max_outer_iters: u64,
) -> (Array1<f64>, u64) {
    let lower_bounds = theta_lower_bounds(problem.k_re);
    let cost = ThetaObjective {
        problem,
        start,
        reml,
        max_inner,
        n_agq,
    };
    let result = nelder_mead_optimize(init.clone(), &lower_bounds, max_outer_iters, cost)
        .unwrap_or_else(|_| crate::optimizer::OptimizeResult {
            theta: init,
            converged: false,
            iterations: 0,
            final_cost: f64::MAX,
        });
    let mut theta = result.theta;
    if let Some(slice) = theta.as_slice_mut() {
        clamp_nlmm_theta(slice, problem.k_re);
    }
    (theta, result.iterations)
}

fn default_theta_init(k_re: usize) -> Array1<f64> {
    match k_re {
        // lme4 relative Cholesky diagonals (σ-scaled RE SD ≈ θ·σ).
        1 => Array1::from_vec(vec![4.0]),
        2 => Array1::from_vec(vec![4.6, 3.8, 3.0]),
        _ => Array1::from_elem(theta_len(k_re), 1.0),
    }
}

fn default_start_map(mean: &dyn NlmmMeanEval, param_names: &[String]) -> NlmmStart {
    let values = mean.default_start_values(param_names);
    param_names
        .iter()
        .zip(values.iter())
        .map(|(name, value)| (name.clone(), *value))
        .collect()
}

fn bound_vectors(
    param_names: &[String],
    lower: &Option<NlmmStart>,
    upper: &Option<NlmmStart>,
) -> crate::Result<(Vec<f64>, Vec<f64>)> {
    let mut lo = vec![f64::NEG_INFINITY; param_names.len()];
    let mut hi = vec![f64::INFINITY; param_names.len()];
    if let Some(map) = lower {
        for (name, value) in map {
            let Some(idx) = param_names.iter().position(|n| n == name) else {
                return Err(LmeError::NotImplemented {
                    feature: format!("nlmer lower bound unknown parameter '{name}'"),
                });
            };
            lo[idx] = *value;
        }
    }
    if let Some(map) = upper {
        for (name, value) in map {
            let Some(idx) = param_names.iter().position(|n| n == name) else {
                return Err(LmeError::NotImplemented {
                    feature: format!("nlmer upper bound unknown parameter '{name}'"),
                });
            };
            hi[idx] = *value;
        }
    }
    for (i, name) in param_names.iter().enumerate() {
        if lo[i] > hi[i] {
            return Err(LmeError::NotImplemented {
                feature: format!(
                    "nlmer bounds for '{name}': lower {} > upper {}",
                    lo[i], hi[i]
                ),
            });
        }
    }
    Ok((lo, hi))
}

fn start_candidates(
    opts: &NlmerOptions,
    mean: &dyn NlmmMeanEval,
    y: &Array1<f64>,
    x: &Array1<f64>,
    param_names: &[String],
) -> Vec<NlmmStart> {
    if !opts.start.is_empty() {
        return vec![opts.start.clone()];
    }
    let kind_start = mean.self_start_values(y, x, param_names);
    vec![kind_start, default_start_map(mean, param_names)]
}

struct NlmmOptimized {
    thetas: Array1<f64>,
    params: Vec<f64>,
    b: Array1<f64>,
    deviance: f64,
    outer_iters: u64,
}

fn optimize_nlmm_at_start(
    problem: &NlmmProblem,
    start: &NlmmStart,
    k_re: usize,
    opts: &NlmerOptions,
) -> NlmmOptimized {
    let (thetas, outer_iters) = if k_re == 1 {
        let (theta0, _cost, iters) = optimize_theta_golden(
            problem,
            start,
            opts.reml,
            opts.max_inner,
            opts.n_agq,
            0.2,
            20.0,
        );
        (Array1::from_vec(vec![theta0]), iters)
    } else {
        let inits = vec![
            default_theta_init(k_re),
            Array1::from_vec(match k_re {
                2 => vec![3.5, 2.5, 2.0],
                _ => vec![2.0; theta_len(k_re)],
            }),
            Array1::from_vec(match k_re {
                2 => vec![5.5, 0.0, 4.0],
                _ => vec![6.0; theta_len(k_re)],
            }),
        ];
        let mut best_theta = inits[0].clone();
        let mut best_cost = f64::MAX;
        let mut total_iters = 0u64;
        for init in inits {
            let (theta, iters) = optimize_theta_nelder_mead(
                problem,
                start,
                opts.reml,
                opts.max_inner,
                opts.n_agq,
                init,
                opts.max_outer_iters.max(600),
            );
            let cost = problem
                .profile_objective(
                    theta.as_slice().unwrap(),
                    start,
                    opts.reml,
                    opts.max_inner,
                    opts.n_agq,
                )
                .0;
            if cost < best_cost {
                best_cost = cost;
                best_theta = theta;
            }
            total_iters += iters;
        }
        (best_theta, total_iters)
    };

    let theta_slice = thetas.as_slice().unwrap();
    let (deviance, params, _sigma2_inner, b) =
        problem.profile_objective(theta_slice, start, opts.reml, opts.max_inner, opts.n_agq);

    NlmmOptimized {
        thetas,
        params,
        b,
        deviance,
        outer_iters,
    }
}

/// Fit a nonlinear mixed model from a parsed formula and data frame.
pub fn fit_nlmer(
    parsed: &NlmerFormula,
    mean: Arc<dyn NlmmMeanEval>,
    data: &polars::prelude::DataFrame,
    formula_str: &str,
    opts: &NlmerOptions,
) -> crate::Result<LmeFit> {
    let re_indices = re_param_indices(parsed)?;
    let k_re = re_indices.len();
    let n_fix = mean.n_params();
    if parsed.fixed_param_names.len() != n_fix {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Nonlinear mean requires {n_fix} fixed parameters, got {}",
                parsed.fixed_param_names.len()
            ),
        });
    }

    let y = column_f64(data, &parsed.response)?;
    let x = column_f64(data, &parsed.covariate)?;
    let groups = column_str(data, &parsed.re_group)?;
    let mut level_map = std::collections::HashMap::<String, usize>::new();
    let mut group = Vec::with_capacity(y.len());
    for g in groups {
        let m = level_map.len();
        let idx = *level_map.entry(g).or_insert(m);
        group.push(idx);
    }
    let m = level_map.len();
    if m == 0 {
        return Err(LmeError::NotImplemented {
            feature: "No random-effect groups".to_string(),
        });
    }

    let candidates = start_candidates(opts, mean.as_ref(), &y, &x, &parsed.fixed_param_names);
    let (lower, upper) = bound_vectors(&parsed.fixed_param_names, &opts.lower, &opts.upper)?;

    let problem = NlmmProblem {
        y,
        x,
        group,
        m,
        mean: mean.clone(),
        param_names: parsed.fixed_param_names.clone(),
        re_indices,
        k_re,
        n_fix,
        lower,
        upper,
    };

    let t_len = theta_len(k_re);
    let _lower = compute_theta_lower_bounds(&[ReBlock {
        m,
        k: k_re,
        theta_len: t_len,
        group_name: parsed.re_group.clone(),
        effect_names: parsed.re_params.clone(),
        group_map: level_map.clone(),
    }]);

    let mut best: Option<(NlmmStart, NlmmOptimized)> = None;
    for start in candidates {
        let optimized = optimize_nlmm_at_start(&problem, &start, k_re, opts);
        let replace = best
            .as_ref()
            .is_none_or(|(_, prev)| optimized.deviance < prev.deviance);
        if replace {
            best = Some((start, optimized));
        }
    }
    let (_resolved_start, optimized) = best.expect("at least one start candidate");
    let NlmmOptimized {
        thetas,
        params,
        b,
        deviance,
        outer_iters,
    } = optimized;

    let theta_slice = thetas.as_slice().unwrap();

    let fitted = problem.predict(&params, &b);
    let residuals = &problem.y - &fitted;
    let n = problem.y.len();
    let rss_nl: f64 = residuals.iter().map(|r| r * r).sum();
    let mut b_pen = 0.0;
    for g in 0..m {
        let bg: Vec<f64> = (0..k_re).map(|r| b[problem.b_index(g, r)]).collect();
        b_pen += re_penalty(k_re, theta_slice, &bg);
    }
    let n_f = n as f64;
    let p = n_fix as f64;
    let df = if opts.reml { (n_f - p).max(1.0) } else { n_f };
    let pwrss = rss_nl + b_pen;
    let scalar_ssasymp = k_re == 1 && mean.uses_scalar_rss_sigma();
    let sigma2 = if scalar_ssasymp {
        (rss_nl / df).max(1e-12)
    } else {
        (pwrss / df).max(1e-12)
    };
    let loglik = -deviance / 2.0;

    let coefficients = Array1::from_vec(params);
    let mut ranef_rows = Vec::new();
    for (label, &idx) in &level_map {
        for (r_slot, re_name) in parsed.re_params.iter().enumerate() {
            ranef_rows.push((
                parsed.re_group.clone(),
                label.clone(),
                re_name.clone(),
                b[problem.b_index(idx, r_slot)],
            ));
        }
    }
    let ranef_df = build_ranef_df(&ranef_rows);
    let var_corr_df = build_nlmm_varcorr(
        &parsed.re_group,
        &parsed.re_params,
        theta_slice,
        k_re,
        sigma2,
    );

    let n_params = p + t_len as f64 + 1.0;
    let aic = deviance + 2.0 * n_params;
    let bic = deviance + n_params * (n as f64).ln();

    Ok(LmeFit {
        coefficients,
        residuals,
        fitted,
        ranef: Some(ranef_df),
        var_corr: Some(var_corr_df),
        theta: Some(thetas),
        sigma2: Some(sigma2),
        reml: if opts.reml { Some(deviance) } else { None },
        log_likelihood: Some(loglik),
        aic: Some(aic),
        bic: Some(bic),
        deviance: Some(deviance),
        b: Some(b),
        u: None,
        beta_se: None,
        beta_t: None,
        formula: Some(formula_str.to_string()),
        fixed_names: Some(parsed.fixed_param_names.clone()),
        fixed_term_assign: None,
        fixed_design_x: None,
        re_blocks: Some(vec![ReBlock {
            m,
            k: k_re,
            theta_len: t_len,
            group_name: parsed.re_group.clone(),
            effect_names: parsed.re_params.clone(),
            group_map: level_map,
        }]),
        num_obs: n,
        converged: Some(true),
        iterations: Some(outer_iters),
        family_name: Some("nlmm".to_string()),
        link_name: None,
        family: None,
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: None,
        robust: None,
        categorical_levels: None,
        nlmm_mean: Some(mean),
        nlmm_formula: Some(parsed.clone()),
    })
}

pub(crate) fn column_f64(
    df: &polars::prelude::DataFrame,
    name: &str,
) -> crate::Result<Array1<f64>> {
    let s = df.column(name).map_err(|e| LmeError::NotImplemented {
        feature: format!("Column '{name}': {e}"),
    })?;
    if let Ok(ca) = s.f64() {
        return Ok(Array1::from_iter(
            ca.into_iter().map(|v| v.unwrap_or(f64::NAN)),
        ));
    }
    if let Ok(ca) = s.i64() {
        return Ok(Array1::from_iter(
            ca.into_iter().map(|v| v.unwrap_or(0) as f64),
        ));
    }
    if let Ok(ca) = s.f32() {
        return Ok(Array1::from_iter(
            ca.into_iter().map(|v| v.unwrap_or(f32::NAN) as f64),
        ));
    }
    Err(LmeError::NotImplemented {
        feature: format!("Column '{name}' must be numeric"),
    })
}

pub(crate) fn column_str(
    df: &polars::prelude::DataFrame,
    name: &str,
) -> crate::Result<Vec<String>> {
    let s = df.column(name).map_err(|e| LmeError::NotImplemented {
        feature: format!("Column '{name}': {e}"),
    })?;
    if let Ok(ca) = s.str() {
        return Ok(ca
            .into_iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect());
    }
    if let Ok(ca) = s.f64() {
        return Ok(ca
            .into_iter()
            .map(|v| v.map(|x| x.to_string()).unwrap_or_default())
            .collect());
    }
    if let Ok(ca) = s.i64() {
        return Ok(ca
            .into_iter()
            .map(|v| v.map(|x| x.to_string()).unwrap_or_default())
            .collect());
    }
    let cast =
        s.cast(&polars::prelude::DataType::String)
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("Column '{name}' could not be cast to string: {e}"),
            })?;
    let ca = cast.str().map_err(|e| LmeError::NotImplemented {
        feature: format!("Column '{name}' string cast failed: {e}"),
    })?;
    Ok(ca
        .into_iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect())
}

fn build_ranef_df(rows: &[(String, String, String, f64)]) -> polars::prelude::DataFrame {
    use polars::prelude::*;
    let groups: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    let levels: Vec<String> = rows.iter().map(|r| r.1.clone()).collect();
    let names: Vec<String> = rows.iter().map(|r| r.2.clone()).collect();
    let vals: Vec<f64> = rows.iter().map(|r| r.3).collect();
    DataFrame::new(vec![
        Column::new("group".into(), &groups),
        Column::new("level".into(), &levels),
        Column::new("term".into(), &names),
        Column::new("condval".into(), &vals),
    ])
    .unwrap_or_default()
}

fn build_nlmm_varcorr(
    group: &str,
    re_params: &[String],
    theta: &[f64],
    k_re: usize,
    sigma2: f64,
) -> polars::prelude::DataFrame {
    use polars::prelude::*;
    let sigma_re = sigma_from_theta(k_re, theta).mapv(|v| v * sigma2);
    let mut grps = Vec::new();
    let mut var1 = Vec::new();
    let mut vcov = Vec::new();
    let mut sdcor = Vec::new();
    for (i, name) in re_params.iter().enumerate() {
        let var = sigma_re[[i, i]];
        grps.push(group.to_string());
        var1.push(name.clone());
        vcov.push(var);
        sdcor.push(var.sqrt());
    }
    grps.push("Residual".to_string());
    var1.push(String::new());
    vcov.push(sigma2);
    sdcor.push(sigma2.sqrt());
    DataFrame::new(vec![
        Column::new("grp".into(), &grps),
        Column::new("var1".into(), &var1),
        Column::new("vcov".into(), &vcov),
        Column::new("sdcor".into(), &sdcor),
    ])
    .unwrap_or_default()
}

#[cfg(test)]
mod orange_inner {
    use super::*;
    use crate::nlmm::formula::{parse_nlmer_formula, NlmmMeanKind};
    use crate::nlmm::mean_fn::builtin_mean;
    use polars::prelude::SerReader;
    use std::fs::File;

    #[test]
    fn inner_at_r_tau_matches_reference() {
        let mut file = File::open("tests/data/orange.csv").unwrap();
        let df = polars::prelude::CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(&mut file)
            .finish()
            .unwrap();
        let (parsed, mean) =
            parse_nlmer_formula("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree")
                .unwrap();
        let y = column_f64(&df, &parsed.response).unwrap();
        let x = column_f64(&df, &parsed.covariate).unwrap();
        let groups = column_str(&df, &parsed.re_group).unwrap();
        let mut level_map = std::collections::HashMap::<String, usize>::new();
        let mut group = Vec::new();
        for g in groups {
            let m = level_map.len();
            let idx = *level_map.entry(g).or_insert(m);
            group.push(idx);
        }
        let mut start = NlmmStart::new();
        start.insert("Asym".to_string(), 200.0);
        start.insert("xmid".to_string(), 725.0);
        start.insert("scal".to_string(), 350.0);
        let re_indices = re_param_indices(&parsed).unwrap();
        let problem = NlmmProblem {
            y,
            x,
            group,
            m: 5,
            mean: builtin_mean(mean),
            param_names: parsed.fixed_param_names.clone(),
            re_indices,
            k_re: 1,
            n_fix: 3,
            lower: vec![f64::NEG_INFINITY; 3],
            upper: vec![f64::INFINITY; 3],
        };
        let (params, _, _) = problem.inner_gauss_newton(&[4.03497223047614], &start, 200);
        assert!(
            (params[0] - 192.0528).abs() < 3.0,
            "asym={} xmid={} scal={}",
            params[0],
            params[1],
            params[2]
        );
        assert!((params[1] - 727.9045).abs() < 5.0, "xmid={}", params[1]);
        assert_eq!(mean, NlmmMeanKind::Sslogis);
    }

    /// Inner GN at R's converged relative θ (lme4 `getME(., "theta")`).
    #[test]
    fn multi_re_inner_at_r_theta_matches_reference() {
        let mut file = File::open("tests/data/orange.csv").unwrap();
        let df = polars::prelude::CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(&mut file)
            .finish()
            .unwrap();
        let (parsed, mean) = parse_nlmer_formula(
            "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree",
        )
        .unwrap();
        let y = column_f64(&df, &parsed.response).unwrap();
        let x = column_f64(&df, &parsed.covariate).unwrap();
        let groups = column_str(&df, &parsed.re_group).unwrap();
        let mut level_map = std::collections::HashMap::<String, usize>::new();
        let mut group = Vec::new();
        for g in groups {
            let m = level_map.len();
            let idx = *level_map.entry(g).or_insert(m);
            group.push(idx);
        }
        let mut start = NlmmStart::new();
        start.insert("Asym".to_string(), 200.0);
        start.insert("xmid".to_string(), 725.0);
        start.insert("scal".to_string(), 350.0);
        let re_indices = re_param_indices(&parsed).unwrap();
        let problem = NlmmProblem {
            y,
            x,
            group,
            m: 5,
            mean: builtin_mean(mean),
            param_names: parsed.fixed_param_names.clone(),
            re_indices,
            k_re: 2,
            n_fix: 3,
            lower: vec![f64::NEG_INFINITY; 3],
            upper: vec![f64::INFINITY; 3],
        };
        let theta_rel = [4.59538334289034, 3.80974418701676, 2.98859274071634];
        let (params, b, _) = problem.inner_gauss_newton(&theta_rel, &start, 400);
        let pen = problem.penalized_rss(&params, &b, &theta_rel);
        assert!(pen < 2200.0, "inner PLS pen={pen}");
        assert!((params[0] - 191.3665).abs() < 8.0, "Asym={}", params[0]);
        assert!((params[1] - 717.5343).abs() < 8.0, "xmid={}", params[1]);
        assert!((params[2] - 346.8667).abs() < 8.0, "scal={}", params[2]);
    }
}
