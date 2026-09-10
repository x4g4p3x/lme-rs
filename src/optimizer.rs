use crate::math::LmmData;
use crate::model_matrix::ReBlock;
use crate::quadrature::{resolve_gh_order_joint, resolve_gh_order_product};
use argmin::core::{CostFunction, Error};
#[cfg(not(feature = "basin"))]
use argmin::core::{Executor, State};
#[cfg(not(feature = "basin"))]
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};
use sprs::CsMat;
use std::sync::Arc;

#[cfg(feature = "basin")]
mod basin_backend;

/// Result of the Nelder-Mead optimization, including convergence diagnostics.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// The optimized theta vector (relative covariance parameters).
    pub theta: Array1<f64>,
    /// Whether the optimizer converged within the iteration limit.
    pub converged: bool,
    /// Number of iterations executed.
    pub iterations: u64,
    /// Final cost (deviance) at the optimized theta.
    pub final_cost: f64,
}

/// Compute lower bounds for each θ element based on the random-effect block structure.
///
/// In R's `lme4`, diagonal entries of the lower-triangular Cholesky factor Λ must be ≥ 0
/// (they represent standard deviations), while off-diagonal entries are unbounded (correlations).
/// The lower-triangular factor for a k×k block is stored column-major:
///   col 0: θ\[0\] (diagonal), θ\[1\] (off-diag), ..., θ\[k-1\] (off-diag)
///   col 1: θ\[k\] (diagonal), θ\[k+1\] (off-diag), ...
/// Diagonal positions within each column j are the first element: row index j.
pub fn compute_theta_lower_bounds(re_blocks: &[ReBlock]) -> Vec<f64> {
    let mut bounds = Vec::new();
    for block in re_blocks {
        let k = block.k;
        for j in 0..k {
            for i in j..k {
                if i == j {
                    // Diagonal: standard deviation, must be ≥ 0
                    bounds.push(0.0);
                } else {
                    // Off-diagonal: correlation parameter, unbounded
                    bounds.push(f64::NEG_INFINITY);
                }
            }
        }
    }
    bounds
}

/// Clamp a theta vector to respect lower bounds element-wise.
fn clamp_theta(theta: &mut Array1<f64>, lower_bounds: &[f64]) {
    for i in 0..theta.len() {
        if theta[i] < lower_bounds[i] {
            theta[i] = lower_bounds[i];
        }
    }
}

pub(crate) fn nelder_mead_optimize<C>(
    init_theta: Array1<f64>,
    lower_bounds: &[f64],
    max_iters: u64,
    cost: C,
) -> Result<OptimizeResult, anyhow::Error>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    nelder_mead_optimize_tolerance(init_theta, lower_bounds, max_iters, 1e-6, cost)
}

#[cfg(feature = "basin")]
pub(crate) use basin_backend::optimize as nelder_mead_optimize_tolerance;

#[cfg(not(feature = "basin"))]
pub(crate) fn nelder_mead_optimize_tolerance<C>(
    init_theta: Array1<f64>,
    lower_bounds: &[f64],
    max_iters: u64,
    tolerance: f64,
    cost: C,
) -> Result<OptimizeResult, anyhow::Error>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    let n = init_theta.len();
    let mut initial_simplex = vec![init_theta.clone()];

    for i in 0..n {
        let mut param = init_theta.clone();
        param[i] += 0.2;
        clamp_theta(&mut param, lower_bounds);
        initial_simplex.push(param);
    }

    let solver = NelderMead::new(initial_simplex).with_sd_tolerance(tolerance)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()?;

    let state = res.state();
    let mut best_theta = state.get_best_param().cloned().unwrap_or(init_theta);
    clamp_theta(&mut best_theta, lower_bounds);
    let best_cost = state.get_best_cost();
    let iterations = state.get_iter();
    let converged = iterations < max_iters && best_cost.is_finite() && best_cost < f64::MAX;

    Ok(OptimizeResult {
        theta: best_theta,
        converged,
        iterations,
        final_cost: best_cost,
    })
}

/// REML deviance objective shared by the optimizer backends.
struct LmmObjective {
    lmm: Arc<LmmData>,
    reml: bool,
    lower_bounds: Vec<f64>,
}

impl CostFunction for LmmObjective {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Clamp theta to respect lower bounds before evaluation
        let mut theta_clamped = theta.clone();
        clamp_theta(&mut theta_clamped, &self.lower_bounds);

        let val = self
            .lmm
            .log_reml_deviance(theta_clamped.as_slice().unwrap(), self.reml);
        if val.is_nan() {
            Ok(f64::MAX)
        } else {
            Ok(val)
        }
    }
}

/// Optimizes $\theta$ (the variance component vector) using Nelder-Mead on an un-gradiented search space.
///
/// Enforces lower bounds on θ: diagonal elements of the Cholesky factor ≥ 0.
pub fn optimize_theta_nd(
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    init_theta: Array1<f64>,
    reml: bool,
    weights: Option<Array1<f64>>,
) -> Result<OptimizeResult, anyhow::Error> {
    let lmm = Arc::new(LmmData::new_weighted(x, zt, y, re_blocks, weights));
    optimize_theta_lmm(lmm, init_theta, reml)
}

/// Like [`optimize_theta_nd`] but reuses a pre-built [`LmmData`] (avoids duplicate cross-product setup).
pub fn optimize_theta_lmm(
    lmm: Arc<LmmData>,
    init_theta: Array1<f64>,
    reml: bool,
) -> Result<OptimizeResult, anyhow::Error> {
    crate::perf_diag::scope(crate::perf_diag::Phase::LmerOptimize, || {
        optimize_theta_lmm_inner(lmm, init_theta, reml)
    })
}

/// Optimize using caller controls; defaults keep the specialized LMM search.
pub fn optimize_theta_lmm_control(
    lmm: Arc<LmmData>,
    init_theta: Array1<f64>,
    reml: bool,
    control: &crate::FitControl,
) -> Result<OptimizeResult, anyhow::Error> {
    control.validate(init_theta.len())?;
    if control.default_search() {
        return optimize_theta_lmm(lmm, init_theta, reml);
    }
    let bounds = compute_theta_lower_bounds(&lmm.re_blocks);
    let start = control.start.clone().unwrap_or(init_theta);
    let cost = LmmObjective {
        lmm,
        reml,
        lower_bounds: bounds.clone(),
    };
    nelder_mead_optimize_tolerance(
        start,
        &bounds,
        control.max_iterations,
        control.tolerance,
        cost,
    )
}

fn optimize_theta_lmm_inner(
    lmm: Arc<LmmData>,
    init_theta: Array1<f64>,
    reml: bool,
) -> Result<OptimizeResult, anyhow::Error> {
    let lower_bounds = compute_theta_lower_bounds(&lmm.re_blocks);

    if lmm.intercept_only_re() {
        return match init_theta.len() {
            1 => Ok(optimize_theta_intercept_profile(
                lmm,
                init_theta,
                reml,
                &lower_bounds,
            )),
            2 => optimize_theta_intercept_2d(lmm, init_theta, reml, &lower_bounds),
            _ => {
                let cost = LmmObjective {
                    lmm,
                    reml,
                    lower_bounds: lower_bounds.clone(),
                };
                nelder_mead_optimize(init_theta, &lower_bounds, 1000, cost)
            }
        };
    }

    let cost = LmmObjective {
        lmm,
        reml,
        lower_bounds: lower_bounds.clone(),
    };

    nelder_mead_optimize(init_theta, &lower_bounds, 1000, cost)
}

/// Golden-section profile search for intercept-only models with |θ| = 1.
fn optimize_theta_intercept_profile(
    lmm: Arc<LmmData>,
    init_theta: Array1<f64>,
    reml: bool,
    lower_bounds: &[f64],
) -> OptimizeResult {
    let mut theta = init_theta;
    clamp_theta(&mut theta, lower_bounds);
    let mut trial = theta.as_slice().unwrap().to_vec();
    let mut total_iters = 0u64;

    let mut best_cost = {
        trial.copy_from_slice(theta.as_slice().unwrap());
        clamp_theta_slice(&mut trial, lower_bounds);
        let val = lmm.log_reml_deviance(&trial, reml);
        if val.is_finite() {
            val
        } else {
            f64::MAX
        }
    };

    match theta.len() {
        1 => optimize_one_dim(
            &lmm,
            reml,
            lower_bounds,
            0,
            &mut theta,
            &mut trial,
            &mut best_cost,
            &mut total_iters,
        ),
        _ => unreachable!("intercept profile optimizer only handles |θ| = 1"),
    }

    clamp_theta(&mut theta, lower_bounds);
    best_cost = lmm.log_reml_deviance(theta.as_slice().unwrap(), reml);

    OptimizeResult {
        theta,
        converged: best_cost.is_finite() && best_cost < f64::MAX,
        iterations: total_iters,
        final_cost: best_cost,
    }
}

/// Low-evaluation 2D search for intercept-only crossed models.
///
/// ML and REML: 5×5 + local 4×4 log-grids, followed by Nelder–Mead
/// refinement to establish convergence instead of treating a grid minimum as converged.
fn optimize_theta_intercept_2d(
    lmm: Arc<LmmData>,
    init_theta: Array1<f64>,
    reml: bool,
    lower_bounds: &[f64],
) -> Result<OptimizeResult, anyhow::Error> {
    const THETA_HI: f64 = 12.0;
    const COARSE_N: usize = 5;
    const ML_FINE_N: usize = 4;
    const REML_FINE_N: usize = 4;
    const NM_POLISH_ITERS: u64 = 1000;

    let mut theta = init_theta;
    clamp_theta(&mut theta, lower_bounds);
    let mut trial = theta.as_slice().unwrap().to_vec();
    let mut grid_evals = 0u64;

    let mut eval = |trial: &mut [f64]| -> f64 {
        grid_evals += 1;
        clamp_theta_slice(trial, lower_bounds);
        let val = lmm.log_reml_deviance(trial, reml);
        if val.is_finite() {
            val
        } else {
            f64::MAX
        }
    };

    let theta_lo = |dim: usize| lower_bounds[dim].max(1e-6);

    let mut best_cost = {
        trial.copy_from_slice(theta.as_slice().unwrap());
        eval(&mut trial)
    };

    grid_search_2d(
        &mut theta,
        &mut trial,
        &mut best_cost,
        &mut eval,
        theta_lo(0),
        THETA_HI,
        theta_lo(1),
        THETA_HI,
        COARSE_N,
    );

    let local_lo0 = (theta[0] / 4.0).max(theta_lo(0));
    let local_hi0 = (theta[0] * 4.0)
        .min(THETA_HI)
        .max(local_lo0 * (1.0 + 1e-12));
    let local_lo1 = (theta[1] / 4.0).max(theta_lo(1));
    let local_hi1 = (theta[1] * 4.0)
        .min(THETA_HI)
        .max(local_lo1 * (1.0 + 1e-12));
    let fine_n = if reml { REML_FINE_N } else { ML_FINE_N };
    grid_search_2d(
        &mut theta,
        &mut trial,
        &mut best_cost,
        &mut eval,
        local_lo0,
        local_hi0,
        local_lo1,
        local_hi1,
        fine_n,
    );

    let cost = LmmObjective {
        lmm,
        reml,
        lower_bounds: lower_bounds.to_vec(),
    };
    let mut result = nelder_mead_optimize(theta, lower_bounds, NM_POLISH_ITERS, cost)?;
    result.iterations += grid_evals;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn grid_search_2d(
    theta: &mut Array1<f64>,
    trial: &mut [f64],
    best_cost: &mut f64,
    eval: &mut impl FnMut(&mut [f64]) -> f64,
    lo0: f64,
    hi0: f64,
    lo1: f64,
    hi1: f64,
    n: usize,
) {
    let grid0 = log_grid_1d(lo0, hi0, n);
    let grid1 = log_grid_1d(lo1, hi1, n);
    for &t0 in &grid0 {
        for &t1 in &grid1 {
            trial[0] = t0;
            trial[1] = t1;
            let cost = eval(trial);
            if cost < *best_cost {
                *best_cost = cost;
                theta[0] = t0;
                theta[1] = t1;
            }
        }
    }
}

/// Log-spaced grid on `[lo, hi]` (inclusive endpoints).
fn log_grid_1d(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(lo * hi).sqrt()];
    }
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (log_lo + t * (log_hi - log_lo)).exp()
        })
        .collect()
}

fn clamp_theta_slice(theta: &mut [f64], lower_bounds: &[f64]) {
    for (value, &bound) in theta.iter_mut().zip(lower_bounds.iter()) {
        if *value < bound {
            *value = bound;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn optimize_one_dim(
    lmm: &LmmData,
    reml: bool,
    lower_bounds: &[f64],
    dim: usize,
    theta: &mut Array1<f64>,
    trial: &mut [f64],
    best_cost: &mut f64,
    total_iters: &mut u64,
) {
    const COORD_TOL: f64 = 1e-6;
    const HI_CAP: f64 = 12.0;
    const GS_MAX_ITERS: u64 = 16;

    let lo = lower_bounds[dim].max(1e-6);
    let center = theta[dim].max(lo + 1e-6);
    let lo = lo.max(center / 8.0);
    let hi = (center * 8.0).min(HI_CAP).max(lo + 1e-6);
    let (value, cost, iters) = golden_section_min_coord(
        |trial_val| {
            trial.copy_from_slice(theta.as_slice().unwrap());
            trial[dim] = trial_val;
            clamp_theta_slice(trial, lower_bounds);
            let val = lmm.log_reml_deviance(trial, reml);
            if val.is_finite() {
                val
            } else {
                f64::MAX
            }
        },
        lo,
        hi,
        COORD_TOL,
        GS_MAX_ITERS,
    );
    *total_iters += iters;
    theta[dim] = value;
    if cost < *best_cost {
        *best_cost = cost;
    }
}

/// Minimize a unimodal scalar function on `[lo, hi]` via golden-section search.
fn golden_section_min_coord<F>(
    mut f: F,
    lo: f64,
    hi: f64,
    tol: f64,
    max_iters: u64,
) -> (f64, f64, u64)
where
    F: FnMut(f64) -> f64,
{
    if hi <= lo {
        let mid = (lo + hi) / 2.0;
        let cost = f(mid);
        return (mid, cost, 1);
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = f(c);
    let mut fd = f(d);
    let mut iters = 0u64;

    while (b - a).abs() > tol && iters < max_iters {
        iters += 1;
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / phi;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / phi;
            fd = f(d);
        }
    }

    let mid = (a + b) / 2.0;
    let cost = f(mid);
    (mid, cost, iters + 1)
}

// ─── GLMM Optimizer ───────────────────────────────────────────────────────────

use crate::family::GlmFamily;
use crate::glmm_math::{self, GlmmData};

/// GLMM Laplace / AGQ deviance objective shared by the optimizer backends.
#[derive(Clone)]
struct GlmmObjective {
    evaluator: Arc<std::sync::Mutex<GlmmData>>,
    offset: Option<Arc<Array1<f64>>>,
    lower_bounds: Vec<f64>,
    n_agq: usize,
    control: crate::FitControl,
}

impl CostFunction for GlmmObjective {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Clamp theta to respect lower bounds before evaluation
        let mut theta_clamped = theta.clone();
        clamp_theta(&mut theta_clamped, &self.lower_bounds);

        let mut glmm = self
            .evaluator
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned GLMM workspace"))?;
        let val = glmm.laplace_deviance(
            theta_clamped.as_slice().unwrap(),
            self.offset.as_deref(),
            self.n_agq,
        );
        if val.is_nan() {
            Ok(f64::MAX)
        } else {
            Ok(val)
        }
    }
}

/// Optimizes θ for a GLMM using Nelder-Mead on the Laplace- or AGQ-approximated deviance.
///
/// Enforces lower bounds on θ: diagonal elements of the Cholesky factor ≥ 0.
///
/// When `n_agq > 1` and PIRLS can apply AGQ (one RE block with a product Gauss–Hermite
/// grid, or multiple blocks with total `q` small enough for joint AGQ), the outer θ
/// objective uses AGQ deviance after a Laplace warm-start. Scalar `k = 1` matches
/// `lme4::glmer` / `nlmer`; vector and small-`q` joint AGQ-in-θ go beyond lme4 (which
/// stays Laplace for multivariate RE). Otherwise the outer objective stays Laplace.
#[allow(clippy::too_many_arguments)]
pub fn optimize_theta_glmm(
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    init_theta: Array1<f64>,
    family: Box<dyn GlmFamily>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    n_agq: usize,
) -> Result<OptimizeResult, anyhow::Error> {
    let (zt_z, zt_w_z_map) = glmm_math::build_glmm_structural_maps(&zt);
    optimize_theta_glmm_with_maps(
        x, zt, y, re_blocks, init_theta, family, offset, weights, zt_z, zt_w_z_map, n_agq,
    )
}

/// Like [`optimize_theta_glmm`], but reuses precomputed structural maps.
#[allow(clippy::too_many_arguments)]
pub fn optimize_theta_glmm_with_maps(
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    init_theta: Array1<f64>,
    family: Box<dyn GlmFamily>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    zt_z: CsMat<f64>,
    zt_w_z_map: Vec<(usize, usize, f64)>,
    n_agq: usize,
) -> Result<OptimizeResult, anyhow::Error> {
    optimize_theta_glmm_with_maps_control(
        x,
        zt,
        y,
        re_blocks,
        init_theta,
        family,
        offset,
        weights,
        zt_z,
        zt_w_z_map,
        n_agq,
        &crate::FitControl::default(),
    )
}

/// Optimize a GLMM with explicit outer and inner iteration controls.
#[allow(clippy::too_many_arguments)]
pub fn optimize_theta_glmm_with_maps_control(
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    init_theta: Array1<f64>,
    family: Box<dyn GlmFamily>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    zt_z: CsMat<f64>,
    zt_w_z_map: Vec<(usize, usize, f64)>,
    n_agq: usize,
    control: &crate::FitControl,
) -> Result<OptimizeResult, anyhow::Error> {
    optimize_theta_glmm_data_control(
        GlmmData::from_structural_parts(x, zt, y, re_blocks, family, weights, zt_z, zt_w_z_map),
        init_theta,
        offset,
        n_agq,
        control,
    )
}

pub(crate) fn optimize_theta_glmm_data_control(
    data: GlmmData,
    init_theta: Array1<f64>,
    offset: Option<Array1<f64>>,
    n_agq: usize,
    control: &crate::FitControl,
) -> Result<OptimizeResult, anyhow::Error> {
    control.validate(init_theta.len())?;
    let init_theta = control.start.clone().unwrap_or(init_theta);
    let lower_bounds = compute_theta_lower_bounds(&data.re_blocks);
    let n_agq_obj = n_agq_for_theta_objective(n_agq, &data.re_blocks);
    let evaluator = Arc::new(std::sync::Mutex::new(data));
    evaluator
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned GLMM workspace"))?
        .set_control(control.clone());
    let laplace = GlmmObjective {
        evaluator,
        offset: offset.map(Arc::new),
        lower_bounds: lower_bounds.clone(),
        n_agq: 1,
        control: control.clone(),
    };
    let laplace_result = optimize_glmm_theta(init_theta, &lower_bounds, &laplace)?;
    if n_agq_obj <= 1 {
        return Ok(laplace_result);
    }
    let agq = GlmmObjective {
        n_agq: n_agq_obj,
        ..laplace
    };
    let agq_result = optimize_glmm_theta(laplace_result.theta.clone(), &lower_bounds, &agq)?;
    // Reject pathological AGQ refinements (discontinuous AGQ/Laplace fallback landscape).
    let theta_ok = agq_result.theta.iter().all(|t| t.is_finite())
        && agq_result
            .theta
            .iter()
            .zip(laplace_result.theta.iter())
            .all(|(&a, &l)| (a - l).abs() < (5.0_f64).max(10.0 * l.abs().max(1e-8)));
    if agq_result.final_cost.is_finite() && theta_ok {
        Ok(OptimizeResult {
            theta: agq_result.theta,
            converged: laplace_result.converged && agq_result.converged,
            iterations: laplace_result.iterations + agq_result.iterations,
            final_cost: agq_result.final_cost,
        })
    } else {
        Ok(laplace_result)
    }
}

/// Scalar GLMM θ uses a log-grid plus golden-section search so optima near the
/// zero bound (typical for Gamma after φ is profiled) are not skipped by
/// Nelder–Mead started at 1. Vector θ keeps Nelder–Mead.
fn optimize_glmm_theta(
    init_theta: Array1<f64>,
    lower_bounds: &[f64],
    cost: &GlmmObjective,
) -> Result<OptimizeResult, anyhow::Error> {
    if init_theta.len() == 1 && cost.control.default_search() {
        optimize_theta_glmm_1d(init_theta, lower_bounds, cost)
    } else {
        nelder_mead_optimize_tolerance(
            init_theta,
            lower_bounds,
            cost.control.max_iterations,
            cost.control.tolerance,
            cost.clone(),
        )
    }
}

fn optimize_theta_glmm_1d(
    init_theta: Array1<f64>,
    lower_bounds: &[f64],
    cost: &GlmmObjective,
) -> Result<OptimizeResult, anyhow::Error> {
    const LO: f64 = 1e-4;
    const HI: f64 = 8.0;
    const GRID_N: usize = 11;
    const GS_MAX: u64 = 24;
    const GS_TOL: f64 = 1e-6;

    let floor = lower_bounds.first().copied().unwrap_or(0.0).max(LO);
    let mut evals = 0u64;
    let mut eval = |t: f64| -> f64 {
        evals += 1;
        let x = Array1::from_vec(vec![t.max(floor)]);
        match cost.cost(&x) {
            Ok(v) if v.is_finite() => v,
            _ => f64::MAX,
        }
    };

    let mut best_t = init_theta[0].clamp(floor, HI);
    let mut best_c = eval(best_t);
    for &t in &log_grid_1d(floor, HI, GRID_N) {
        let c = eval(t);
        if c < best_c {
            best_c = c;
            best_t = t;
        }
    }

    let lo = (best_t / 4.0).max(floor);
    let hi = (best_t * 4.0).min(HI).max(lo * (1.0 + 1e-12));
    let (value, gs_cost, iters) = golden_section_min_coord(&mut eval, lo, hi, GS_TOL, GS_MAX);
    evals += iters;
    if gs_cost < best_c {
        best_c = gs_cost;
        best_t = value;
    }

    Ok(OptimizeResult {
        theta: Array1::from_vec(vec![best_t]),
        converged: best_c.is_finite() && best_c < f64::MAX,
        iterations: evals,
        final_cost: best_c,
    })
}

/// AGQ inside the θ search when PIRLS can apply the same quadrature rule.
fn n_agq_for_theta_objective(n_agq: usize, re_blocks: &[ReBlock]) -> usize {
    if n_agq <= 1 || re_blocks.is_empty() {
        return 1;
    }
    if re_blocks.len() == 1 {
        return match resolve_gh_order_product(n_agq, re_blocks[0].k) {
            Some(_) => n_agq,
            None => 1,
        };
    }
    let q: usize = re_blocks.iter().map(|b| b.m.saturating_mul(b.k)).sum();
    match resolve_gh_order_joint(n_agq, q) {
        Some(_) => n_agq,
        None => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::PoissonFamily;
    use crate::model_matrix::ReBlock;
    use ndarray::{array, Array2};
    use sprs::TriMat;

    struct TestObjective<F>(F);

    impl<F> CostFunction for TestObjective<F>
    where
        F: Fn(&Array1<f64>) -> Result<f64, Error>,
    {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, theta: &Self::Param) -> Result<f64, Error> {
            (self.0)(theta)
        }
    }

    #[test]
    fn nelder_mead_recovers_interior_and_negative_unbounded_coordinates() {
        let objective = |x: &Array1<f64>| Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2));
        let result = nelder_mead_optimize_tolerance(
            array![1.0, 0.0],
            &[0.0, f64::NEG_INFINITY],
            1000,
            1e-12,
            TestObjective(objective),
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert!((result.theta[0] - 2.0).abs() < 1e-5);
        assert!((result.theta[1] + 1.0).abs() < 1e-5);
        assert_eq!(result.final_cost, objective(&result.theta).unwrap());
    }

    #[test]
    fn nelder_mead_propagates_objective_errors() {
        let evaluations = std::cell::Cell::new(0);
        let result = nelder_mead_optimize(
            array![1.0],
            &[0.0],
            100,
            TestObjective(|x: &Array1<f64>| {
                evaluations.set(evaluations.get() + 1);
                if evaluations.get() > 2 {
                    anyhow::bail!("evaluation failed");
                }
                Ok(x[0].powi(2))
            }),
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("evaluation failed"));
    }

    #[cfg(feature = "basin")]
    #[test]
    fn basin_propagates_initialization_errors() {
        let result = nelder_mead_optimize(
            array![1.0],
            &[0.0],
            100,
            TestObjective(|_: &Array1<f64>| anyhow::bail!("initialization failed")),
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("initialization failed"));
    }

    #[test]
    fn nelder_mead_reports_exhaustion_with_finite_best_iterate() {
        let objective = |x: &Array1<f64>| Ok(x[0].powi(2));
        let result =
            nelder_mead_optimize(array![1.0], &[0.0], 1, TestObjective(objective)).unwrap();
        assert!(!result.converged, "{result:?}");
        assert_eq!(result.iterations, 1);
        assert!(result.final_cost.is_finite() && result.final_cost < 1.0);
        assert_eq!(result.final_cost, objective(&result.theta).unwrap());
    }

    #[cfg(feature = "basin")]
    #[test]
    fn basin_invalid_objectives_and_zero_budget_do_not_converge() {
        for value in [f64::NAN, f64::NEG_INFINITY] {
            let result = nelder_mead_optimize(
                array![1.0],
                &[0.0],
                2,
                TestObjective(|_: &Array1<f64>| Ok(value)),
            )
            .unwrap();
            assert!(!result.converged);
            assert_eq!(result.final_cost, f64::MAX);
            assert_eq!(result.iterations, 2);
        }
        let result = nelder_mead_optimize(
            array![1.0],
            &[0.0],
            0,
            TestObjective(|_: &Array1<f64>| Ok(1.0)),
        )
        .unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn nelder_mead_rejects_invalid_final_costs() {
        for value in [f64::MAX, f64::INFINITY] {
            let result = nelder_mead_optimize(
                array![1.0],
                &[0.0],
                2,
                TestObjective(|_: &Array1<f64>| Ok(value)),
            )
            .unwrap();
            assert!(!result.converged, "{result:?}");
            assert_eq!(result.iterations, 2);
        }
    }

    #[cfg(feature = "basin")]
    #[test]
    fn basin_projects_every_evaluation_and_recovers_boundary_minima() {
        for target in [0.0, 1e-5, 2.0] {
            let objective = |x: &Array1<f64>| {
                assert!(x[0] >= 0.0, "infeasible evaluation: {x:?}");
                Ok((x[0] - target).powi(2) + (x[1] + 1.0).powi(2))
            };
            let result = nelder_mead_optimize_tolerance(
                array![-2.0, 0.0],
                &[0.0, f64::NEG_INFINITY],
                1000,
                1e-16,
                TestObjective(objective),
            )
            .unwrap();
            assert!(result.converged, "target={target}: {result:?}");
            assert!((result.theta[0] - target).abs() < 1e-6, "{result:?}");
            assert!((result.theta[1] + 1.0).abs() < 1e-6, "{result:?}");
            assert_eq!(result.final_cost, objective(&result.theta).unwrap());
        }
    }

    #[test]
    fn test_nan_deviance_cost() {
        // Create an objective that will generate NaN deviance.
        // Poisson family with y = -1.0 will produce NaN deviance residuals.
        let y = array![-1.0, 0.0]; // invalid for Poisson

        let x = Array2::<f64>::ones((2, 2));
        let mut zt_tri = TriMat::new((2, 2));
        zt_tri.add_triplet(0, 0, 1.0);
        zt_tri.add_triplet(1, 1, 1.0);
        let zt = zt_tri.to_csr();

        let re_blocks = vec![ReBlock {
            m: 2,
            k: 1,
            theta_len: 1,
            group_name: "G".to_string(),
            effect_names: vec!["(Intercept)".to_string()],
            group_map: std::collections::HashMap::new(),
        }];

        let family = Box::new(PoissonFamily::new());
        let lower_bounds = compute_theta_lower_bounds(&re_blocks);
        let (zt_z, zt_w_z_map) = glmm_math::build_glmm_structural_maps(&zt);

        let cost_fn = GlmmObjective {
            evaluator: Arc::new(std::sync::Mutex::new(
                glmm_math::GlmmData::from_structural_parts(
                    x, zt, y, re_blocks, family, None, zt_z, zt_w_z_map,
                ),
            )),
            offset: None,
            lower_bounds,
            n_agq: 1,
            control: crate::FitControl::default(),
        };

        let theta = array![1.0];
        let cost = cost_fn.cost(&theta).unwrap();
        assert_eq!(cost, f64::MAX);
    }

    #[test]
    fn nelder_mead_marks_not_converged_when_iteration_budget_exhausted() {
        let y = array![1.0_f64, 2.0];
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 2.0]).unwrap();
        let mut zt_tri = TriMat::new((2, 2));
        zt_tri.add_triplet(0, 0, 1.0);
        zt_tri.add_triplet(1, 1, 1.0);
        let zt = zt_tri.to_csr();
        let re_blocks = vec![ReBlock {
            m: 2,
            k: 1,
            theta_len: 1,
            group_name: "G".to_string(),
            effect_names: vec!["(Intercept)".to_string()],
            group_map: std::collections::HashMap::new(),
        }];
        let lower_bounds = compute_theta_lower_bounds(&re_blocks);
        let cost = LmmObjective {
            lmm: Arc::new(LmmData::new_weighted(x, zt, y, re_blocks, None)),
            reml: true,
            lower_bounds: lower_bounds.clone(),
        };
        let res = nelder_mead_optimize(array![1.0], &lower_bounds, 1, cost).unwrap();
        assert!(
            !res.converged,
            "expected max-iteration exhaustion to set converged=false, got {:?}",
            res
        );
        assert!(res.iterations >= 1, "iterations={}", res.iterations);
    }

    fn dummy_re_block(m: usize, k: usize) -> ReBlock {
        ReBlock {
            m,
            k,
            theta_len: k * (k + 1) / 2,
            group_name: "G".to_string(),
            effect_names: (0..k).map(|i| format!("e{i}")).collect(),
            group_map: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn n_agq_for_theta_uses_product_and_joint_when_grids_fit() {
        assert_eq!(n_agq_for_theta_objective(7, &[dummy_re_block(10, 1)]), 7);
        assert_eq!(n_agq_for_theta_objective(5, &[dummy_re_block(8, 2)]), 5);
        assert_eq!(
            n_agq_for_theta_objective(5, &[dummy_re_block(2, 1), dummy_re_block(2, 1)]),
            5
        );
        assert_eq!(
            n_agq_for_theta_objective(5, &[dummy_re_block(20, 1), dummy_re_block(20, 1)]),
            1
        );
        assert_eq!(n_agq_for_theta_objective(1, &[dummy_re_block(8, 2)]), 1);
        assert_eq!(
            n_agq_for_theta_objective(3, &[dummy_re_block(2, 8)]),
            1,
            "product grid for k=8 exceeds the node cap"
        );
    }

    #[test]
    fn log_grid_1d_spans_endpoints() {
        let grid = log_grid_1d(0.01, 10.0, 5);
        assert_eq!(grid.len(), 5);
        assert!((grid[0] - 0.01).abs() < 1e-12);
        assert!((grid[4] - 10.0).abs() < 1e-9);
        assert!(grid.windows(2).all(|w| w[1] > w[0]));
    }
}
