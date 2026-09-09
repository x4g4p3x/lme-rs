//! Nonlinear mixed-model fitting (Laplace / penalized Gauss–Newton, optional AGQ).

use std::sync::Arc;

use crate::model_matrix::ReBlock;
use crate::nlmm::formula::{re_param_indices, NlmerFormula};
use crate::nlmm::mean_fn::{eval_mean_with_re, NlmmMeanEval};
use crate::nlmm::re_cov::{
    log_det_sigma, re_penalty, sigma_from_theta, sigma_inv_from_theta, theta_len,
};
use crate::optimizer::{compute_theta_lower_bounds, nelder_mead_optimize};
use crate::quadrature::{gh_rule, log_sum_exp, resolve_gh_order, resolve_gh_order_product};
use crate::{LmeError, LmeFit};
use argmin::core::CostFunction;
use ndarray::Array1;
use ndarray_linalg::{Cholesky, Inverse, UPLO};

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
    /// Optional lower bounds on **group-level** parameters `β + b` (by name).
    ///
    /// Only parameters that appear in the random-effects formula are constrained;
    /// after each Gauss–Newton trial, `b` is adjusted so `β_j + b_{g,j}` stays in range.
    pub group_lower: Option<NlmmStart>,
    /// Optional upper bounds on **group-level** parameters `β + b` (by name).
    pub group_upper: Option<NlmmStart>,
    /// Maximum penalized Gauss–Newton iterations per RE-variance evaluation.
    pub max_inner: usize,
    /// Maximum iterations per variance search (each restart uses this budget).
    pub max_outer_iters: u64,
    /// Adaptive Gauss–Hermite quadrature order. `1` (default) uses Laplace only;
    /// values `≥ 2` enable AGQ on the θ profile (scalar `k = 1`, or product quadrature
    /// for `k_re > 1` when the node-count cap allows). Scalar AGQ mirrors `nAGQ` in `lme4`.
    pub n_agq: usize,
}

impl Default for NlmerOptions {
    fn default() -> Self {
        Self {
            reml: false,
            start: NlmmStart::new(),
            lower: None,
            upper: None,
            group_lower: None,
            group_upper: None,
            max_inner: 120,
            max_outer_iters: 500,
            n_agq: 1,
        }
    }
}

#[derive(Clone)]
struct NlmmProblem {
    y: Arc<Array1<f64>>,
    x: Arc<Array1<f64>>,
    group: Arc<Vec<usize>>,
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
    /// Per-parameter group-level (`β + b`) lower bounds.
    group_lower: Vec<f64>,
    /// Per-parameter group-level (`β + b`) upper bounds.
    group_upper: Vec<f64>,
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

    /// Project `b` so each group's `β_j + b_{g,j}` respects [`Self::group_lower`]/`group_upper`].
    fn project_group_params(&self, params: &[f64], b: &mut Array1<f64>) {
        for r_slot in 0..self.k_re {
            let param_idx = self.re_indices[r_slot];
            let lo = self.group_lower[param_idx];
            let hi = self.group_upper[param_idx];
            if !lo.is_finite() && !hi.is_finite() {
                continue;
            }
            let beta = params[param_idx];
            for g in 0..self.m {
                let idx = self.b_index(g, r_slot);
                let mut phi = beta + b[idx];
                if phi < lo {
                    phi = lo;
                } else if phi > hi {
                    phi = hi;
                }
                b[idx] = phi - beta;
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
        let k = self.k_re;
        let inv = sigma_inv_from_theta(k, theta);
        let mut blocks = vec![inv; self.m];
        for i in 0..self.y.len() {
            let g = self.group[i];
            let re_off = self.re_offsets_for_group(b, g);
            let (_, grad) = eval_mean_with_re(
                self.mean.as_ref(),
                self.x[i],
                params,
                &self.re_indices,
                &re_off,
            );
            for r in 0..k {
                for c in 0..k {
                    blocks[g][[r, c]] += grad[self.re_indices[r]] * grad[self.re_indices[c]];
                }
            }
        }
        let mut total = self.m as f64 * log_det_sigma(k, theta);
        for block in blocks {
            let Ok(chol) = block.cholesky(UPLO::Lower) else {
                return f64::INFINITY;
            };
            total += 2.0 * chol.diag().iter().map(|x| x.ln()).sum::<f64>();
        }
        total
    }

    fn inner_gauss_newton_status(
        &self,
        theta: &[f64],
        start: &NlmmStart,
        max_iter: usize,
    ) -> (Vec<f64>, Array1<f64>, f64, bool, usize) {
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
        let mut lambda_lm = if self.k_re == 1 && self.mean.uses_scalar_rss_sigma() {
            1e-2
        } else {
            1e-4
        };
        let mut converged = false;
        let mut iterations = 0;
        for iteration in 0..max_iter {
            iterations = iteration + 1;
            let mut normal = super::block_solve::BlockNormal::new(p_fix, self.k_re, self.m);
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
                let random: Vec<f64> = self.re_indices.iter().map(|&j| grad[j]).collect();
                let fixed: Vec<f64> = grad
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        if self.lower[j] == self.upper[j] {
                            0.0
                        } else {
                            v
                        }
                    })
                    .collect();
                normal.accumulate(g, &fixed, &random, self.y[i] - mui);
            }
            normal.add_prior(&sigma_inv_from_theta(self.k_re, theta), &b, true);
            let old_obj = self.penalized_rss(&params, &b, theta);

            let mut accepted = false;
            let mut step_norm = 0.0f64;
            for _attempt in 0..12 {
                let delta = match normal.solve(lambda_lm) {
                    Some(delta) => delta,
                    None => break,
                };
                let proposed_step = delta.iter().map(|v| v.abs()).fold(0.0, f64::max);
                if proposed_step < 1e-10 {
                    converged = true;
                    break;
                }

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
                    self.project_group_params(&new_params, &mut nb);
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
                converged = true;
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
        (params, b, rss, converged, iterations)
    }

    fn inner_gauss_newton(
        &self,
        theta: &[f64],
        start: &NlmmStart,
        max_iter: usize,
    ) -> (Vec<f64>, Array1<f64>, f64) {
        let (params, b, rss, _, _) = self.inner_gauss_newton_status(theta, start, max_iter);
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
        let re_logdet = if !scalar_ssasymp {
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
            n * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln()) + re_logdet
        };
        if let Some(agq_q) = self.agq_correction(n_agq, &params, &b, theta, sigma2) {
            crit += agq_q;
        }
        (crit, params, sigma2, b)
    }

    /// AGQ correction to the Laplace profile criterion.
    ///
    /// Scalar (`k_re = 1`) uses the same 1-D rule as `lme4::nlmer`. Vector RE uses a
    /// product Gauss–Hermite grid per group when the node-count cap allows it.
    fn agq_correction(
        &self,
        n_agq: usize,
        params: &[f64],
        b: &Array1<f64>,
        theta: &[f64],
        sigma2: f64,
    ) -> Option<f64> {
        if n_agq < 2 {
            return None;
        }
        if self.k_re != 1 {
            return self.agq_correction_product(n_agq, params, b, theta, sigma2);
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

    /// Product AGQ over a `k_re > 1` random-effect vector (one grouping factor).
    ///
    /// Uses the Gauss–Newton Hessian `A_g = Σ⁻¹ + Σ_i (∂μ/∂b)(∂μ/∂b)ᵀ / σ²` and
    /// `b_quad = b_hat + L z` with `L Lᵀ = A_g⁻¹`, matching the GLMM `k > 1` product rule.
    fn agq_correction_product(
        &self,
        n_agq: usize,
        params: &[f64],
        b: &Array1<f64>,
        theta: &[f64],
        sigma2: f64,
    ) -> Option<f64> {
        let k = self.k_re;
        let order = resolve_gh_order_product(n_agq, k)?;
        let (z, w) = gh_rule(order)?;
        let n_nodes = z.len();
        let ncomb = n_nodes.pow(k as u32);
        let inv_sigma = if k <= 2 {
            sigma_inv_from_theta(k, theta)
        } else {
            sigma_from_theta(k, theta).inv().ok()?
        };

        let mut group_obs: Vec<Vec<usize>> = vec![vec![]; self.m];
        for (i, &g) in self.group.iter().enumerate() {
            group_obs[g].push(i);
        }

        let mu_hat = self.predict(params, b);
        let mut total_q = 0.0;
        for g in 0..self.m {
            let obs_idx = &group_obs[g];
            if obs_idx.is_empty() {
                continue;
            }

            let re_off = self.re_offsets_for_group(b, g);
            let mut a_g = inv_sigma.clone();
            for &i in obs_idx {
                let (_, grad) = eval_mean_with_re(
                    self.mean.as_ref(),
                    self.x[i],
                    params,
                    &self.re_indices,
                    &re_off,
                );
                for r in 0..k {
                    let gr = grad[self.re_indices[r]];
                    for s in 0..k {
                        a_g[[r, s]] += gr * grad[self.re_indices[s]] / sigma2;
                    }
                }
            }
            let sigma_hess = a_g.inv().ok()?;
            let l_sigma = sigma_hess.cholesky(UPLO::Lower).ok()?;

            let mut b_hat_g = vec![0.0_f64; k];
            for r in 0..k {
                b_hat_g[r] = b[self.b_index(g, r)];
            }
            let pen_hat = re_penalty(k, theta, &b_hat_g);

            let mut rss_g_hat = 0.0;
            for &i in obs_idx {
                let resid = self.y[i] - mu_hat[i];
                rss_g_hat += resid * resid;
            }

            let mut log_terms = Vec::with_capacity(ncomb);
            let mut b_trial = b.clone();
            let mut z_vec = vec![0.0_f64; k];
            let mut b_quad = vec![0.0_f64; k];
            for t in 0..ncomb {
                let mut rem = t;
                let mut log_w = 0.0_f64;
                for z_dim in &mut z_vec {
                    let idx = rem % n_nodes;
                    rem /= n_nodes;
                    *z_dim = z[idx];
                    log_w += w[idx].ln();
                }
                for r in 0..k {
                    let mut s = 0.0_f64;
                    for j in 0..k {
                        s += l_sigma[[r, j]] * z_vec[j];
                    }
                    b_quad[r] = b_hat_g[r] + s;
                    b_trial[self.b_index(g, r)] = b_quad[r];
                }
                let mu_quad = self.predict(params, &b_trial);
                let mut rss_g_quad = 0.0;
                for &i in obs_idx {
                    let resid = self.y[i] - mu_quad[i];
                    rss_g_quad += resid * resid;
                }
                let rss_diff = (rss_g_quad - rss_g_hat) / sigma2;
                let pen_quad = re_penalty(k, theta, &b_quad);
                let delta = -0.5 * rss_diff - 0.5 * (pen_quad - pen_hat);
                log_terms.push(log_w + delta);
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
#[allow(clippy::too_many_arguments)]
fn optimize_theta_golden(
    problem: &NlmmProblem,
    start: &NlmmStart,
    reml: bool,
    max_inner: usize,
    n_agq: usize,
    lo: f64,
    hi: f64,
    max_outer_iters: u64,
) -> (f64, f64, u64, bool) {
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
    while (b - a).abs() > 1e-4 && iters < max_outer_iters {
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
    (
        theta0,
        final_cost,
        iters,
        (b - a).abs() <= 1e-4 && final_cost.is_finite(),
    )
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
) -> (Array1<f64>, u64, bool) {
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
    (
        theta,
        result.iterations,
        result.converged && result.final_cost < f64::MAX,
    )
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

fn validate_group_bounds_names(
    param_names: &[String],
    re_indices: &[usize],
    bounds: &Option<NlmmStart>,
) -> crate::Result<()> {
    let Some(map) = bounds else {
        return Ok(());
    };
    for name in map.keys() {
        let Some(idx) = param_names.iter().position(|n| n == name) else {
            return Err(LmeError::NotImplemented {
                feature: format!("nlmer group bound unknown parameter '{name}'"),
            });
        };
        if !re_indices.contains(&idx) {
            return Err(LmeError::NotImplemented {
                feature: format!(
                    "nlmer group bound '{name}' is not a random-effect parameter (β+b bounds apply only to RE terms)"
                ),
            });
        }
    }
    Ok(())
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
    converged: bool,
    inner_iters: usize,
}

fn optimize_nlmm_at_start(
    problem: &NlmmProblem,
    start: &NlmmStart,
    k_re: usize,
    opts: &NlmerOptions,
) -> NlmmOptimized {
    let (thetas, outer_iters, outer_converged) = if k_re == 1 {
        let (theta0, _cost, iters, converged) = optimize_theta_golden(
            problem,
            start,
            opts.reml,
            opts.max_inner,
            opts.n_agq,
            0.2,
            20.0,
            opts.max_outer_iters,
        );
        (Array1::from_vec(vec![theta0]), iters, converged)
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
        let mut best_converged = false;
        for init in inits {
            let (theta, iters, converged) = optimize_theta_nelder_mead(
                problem,
                start,
                opts.reml,
                opts.max_inner,
                opts.n_agq,
                init,
                opts.max_outer_iters,
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
                best_converged = converged;
            }
            total_iters += iters;
        }
        (best_theta, total_iters, best_converged)
    };

    let theta_slice = thetas.as_slice().unwrap();
    let (mut deviance, mut params, _sigma2_inner, mut b) =
        problem.profile_objective(theta_slice, start, opts.reml, opts.max_inner, opts.n_agq);

    let (_, _, _, mut inner_converged, mut inner_iters) =
        problem.inner_gauss_newton_status(theta_slice, start, opts.max_inner);
    let mut thetas = thetas;
    let mut outer_iters = outer_iters;
    let mut outer_converged = outer_converged;
    // Laplace likelihood depends on beta through the random-effect Jacobian.
    // Refining theta and beta jointly avoids treating the PWRSS beta as its MLE.
    if !opts.reml && !problem.mean.uses_scalar_rss_sigma() {
        let objective = JointObjective { problem, opts };
        let mut initial = thetas.to_vec();
        initial.extend(&params);
        let mut lower = theta_lower_bounds(k_re);
        lower.extend(&problem.lower);
        if let Ok(result) = nelder_mead_optimize(
            Array1::from_vec(initial),
            &lower,
            opts.max_outer_iters,
            objective,
        ) {
            let (cost, fixed, random, converged, inner) =
                JointObjective { problem, opts }.evaluate(&result.theta, true);
            outer_iters += result.iterations;
            if cost <= deviance {
                thetas = result
                    .theta
                    .slice(ndarray::s![..theta_len(k_re)])
                    .to_owned();
                clamp_nlmm_theta(thetas.as_slice_mut().unwrap(), k_re);
                deviance = cost;
                params = fixed;
                b = random;
                outer_converged = result.converged;
                inner_converged = converged;
                inner_iters = inner;
            }
        }
    }
    NlmmOptimized {
        converged: outer_converged && inner_converged && deviance.is_finite(),
        inner_iters,
        thetas,
        params,
        b,
        deviance,
        outer_iters,
    }
}

struct JointObjective<'a> {
    problem: &'a NlmmProblem,
    opts: &'a NlmerOptions,
}
impl JointObjective<'_> {
    fn evaluate(
        &self,
        values: &Array1<f64>,
        diagnostics: bool,
    ) -> (f64, Vec<f64>, Array1<f64>, bool, usize) {
        let nt = theta_len(self.problem.k_re);
        let mut theta = values.slice(ndarray::s![..nt]).to_vec();
        clamp_nlmm_theta(&mut theta, self.problem.k_re);
        let mut params = values.slice(ndarray::s![nt..]).to_vec();
        self.problem.project_params(&mut params);
        let mut conditional = self.problem.clone();
        conditional.lower.clone_from(&params);
        conditional.upper.clone_from(&params);
        let start = self
            .problem
            .param_names
            .iter()
            .cloned()
            .zip(params)
            .collect();
        let (cost, params, _, b) = conditional.profile_objective(
            &theta,
            &start,
            false,
            self.opts.max_inner,
            self.opts.n_agq,
        );
        let (converged, iterations) = if diagnostics {
            let (_, _, _, c, i) =
                conditional.inner_gauss_newton_status(&theta, &start, self.opts.max_inner);
            (c, i)
        } else {
            (false, 0)
        };
        (cost, params, b, converged, iterations)
    }
}
impl CostFunction for JointObjective<'_> {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, values: &Self::Param) -> Result<f64, argmin::core::Error> {
        let cost = self.evaluate(values, false).0;
        Ok(if cost.is_finite() { cost } else { f64::MAX })
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
    if opts.max_inner == 0 || opts.max_outer_iters == 0 {
        return Err(LmeError::InvalidInput {
            message: "nlmer iteration limits must be positive".into(),
        });
    }
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
    if y.is_empty() || opts.start.values().any(|v| !v.is_finite()) {
        return Err(LmeError::InvalidInput {
            message: "nonlinear fitting requires observations and finite starting parameters"
                .into(),
        });
    }
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
    let (group_lower, group_upper) = bound_vectors(
        &parsed.fixed_param_names,
        &opts.group_lower,
        &opts.group_upper,
    )?;
    validate_group_bounds_names(&parsed.fixed_param_names, &re_indices, &opts.group_lower)?;
    validate_group_bounds_names(&parsed.fixed_param_names, &re_indices, &opts.group_upper)?;

    let problem = NlmmProblem {
        y: Arc::new(y),
        x: Arc::new(x),
        group: Arc::new(group),
        m,
        mean: mean.clone(),
        param_names: parsed.fixed_param_names.clone(),
        re_indices,
        k_re,
        n_fix,
        lower,
        upper,
        group_lower,
        group_upper,
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
        converged,
        inner_iters,
    } = optimized;

    if !deviance.is_finite() || deviance == f64::MAX || !params.iter().all(|x| x.is_finite()) {
        return Err(LmeError::NonConvergence {
            message: "nonlinear objective is not finite".into(),
        });
    }
    let theta_slice = thetas.as_slice().unwrap();

    let fitted = problem.predict(&params, &b);
    let residuals = problem.y.as_ref() - &fitted;
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
        diagnostics: Some(crate::FitDiagnostics {
            termination: if converged {
                crate::TerminationReason::Converged
            } else if inner_iters >= opts.max_inner || outer_iters >= opts.max_outer_iters {
                crate::TerminationReason::IterationLimit
            } else {
                crate::TerminationReason::NoProgress
            },
            outer_iterations: outer_iters,
            inner_iterations: Some(inner_iters),
            objective: deviance,
            requested_n_agq: opts.n_agq,
            effective_n_agq: if problem
                .agq_correction(
                    opts.n_agq,
                    coefficients.as_slice().unwrap(),
                    &b,
                    theta_slice,
                    sigma2,
                )
                .is_some()
            {
                resolve_gh_order_product(opts.n_agq, k_re).unwrap_or(1)
            } else {
                1
            },
            warnings: Vec::new(),
        }),
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
        converged: Some(converged),
        iterations: Some(outer_iters),
        family_name: Some("nlmm".to_string()),
        link_name: None,
        family: None,
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: None,
        robust: None,
        categorical_levels: None,
        basis_encodings: None,
        nlmm_mean: Some(mean),
        nlmm_formula: Some(parsed.clone()),
        weights: None,
    })
}

pub(crate) fn column_f64(
    df: &polars::prelude::DataFrame,
    name: &str,
) -> crate::Result<Array1<f64>> {
    crate::model_matrix::numeric_column_f64(df, name)
}

pub(crate) fn column_str(
    df: &polars::prelude::DataFrame,
    name: &str,
) -> crate::Result<Vec<String>> {
    let s = df.column(name).map_err(|e| LmeError::NotImplemented {
        feature: format!("Column '{name}': {e}"),
    })?;
    if s.null_count() > 0 {
        return Err(LmeError::InvalidInput {
            message: format!("Grouping column '{name}' contains nulls"),
        });
    }
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
    fn inner_at_r_tau_is_stationary_for_penalized_least_squares() {
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
            y: Arc::new(y),
            x: Arc::new(x),
            group: Arc::new(group),
            m: 5,
            mean: builtin_mean(mean),
            param_names: parsed.fixed_param_names.clone(),
            re_indices,
            k_re: 1,
            n_fix: 3,
            lower: vec![f64::NEG_INFINITY; 3],
            upper: vec![f64::INFINITY; 3],
            group_lower: vec![f64::NEG_INFINITY; 3],
            group_upper: vec![f64::INFINITY; 3],
        };
        let theta = [4.03497223047614];
        let (params, b, _) = problem.inner_gauss_newton(&theta, &start, 200);
        // PWRSS is not the joint Laplace likelihood. Check its stationarity here;
        // integration tests compare the final likelihood fit with lme4.
        for j in 0..params.len() {
            let h = 1e-4;
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[j] += h;
            minus[j] -= h;
            let derivative = (problem.penalized_rss(&plus, &b, &theta)
                - problem.penalized_rss(&minus, &b, &theta))
                / (2.0 * h);
            assert!(
                derivative.abs() < 1e-3,
                "parameter {j}: gradient {derivative}"
            );
        }
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
            y: Arc::new(y),
            x: Arc::new(x),
            group: Arc::new(group),
            m: 5,
            mean: builtin_mean(mean),
            param_names: parsed.fixed_param_names.clone(),
            re_indices,
            k_re: 2,
            n_fix: 3,
            lower: vec![f64::NEG_INFINITY; 3],
            upper: vec![f64::INFINITY; 3],
            group_lower: vec![f64::NEG_INFINITY; 3],
            group_upper: vec![f64::INFINITY; 3],
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
