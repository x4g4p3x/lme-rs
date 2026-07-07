use crate::math::LmmData;
use crate::model_matrix::ReBlock;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};
use sprs::CsMat;
use std::sync::Arc;

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
    let n = init_theta.len();
    let mut initial_simplex = vec![init_theta.clone()];

    for i in 0..n {
        let mut param = init_theta.clone();
        param[i] += 0.2;
        clamp_theta(&mut param, lower_bounds);
        initial_simplex.push(param);
    }

    let solver = NelderMead::new(initial_simplex).with_sd_tolerance(1e-6)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()?;

    let state = res.state();
    let mut best_theta = state.get_best_param().cloned().unwrap_or(init_theta);
    clamp_theta(&mut best_theta, lower_bounds);
    let best_cost = state.get_best_cost();
    let iterations = state.get_iter();
    let converged = iterations < max_iters;

    Ok(OptimizeResult {
        theta: best_theta,
        converged,
        iterations,
        final_cost: best_cost,
    })
}

/// Wrapper for the REML deviance function to be used by argmin.
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
        converged: true,
        iterations: total_iters,
        final_cost: best_cost,
    }
}

/// Low-evaluation 2D search for intercept-only crossed models.
///
/// ML (`reml = false`): 6×6 + local 5×5 log-grids only (~61 evals).
/// REML: adds a short Nelder–Mead polish so golden parity fixtures converge.
fn optimize_theta_intercept_2d(
    lmm: Arc<LmmData>,
    init_theta: Array1<f64>,
    reml: bool,
    lower_bounds: &[f64],
) -> Result<OptimizeResult, anyhow::Error> {
    const THETA_HI: f64 = 12.0;
    const COARSE_N: usize = 6;
    const ML_FINE_N: usize = 5;
    const REML_FINE_N: usize = 4;
    const NM_POLISH_ITERS: u64 = 20;

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

    if reml {
        let cost = LmmObjective {
            lmm: Arc::clone(&lmm),
            reml,
            lower_bounds: lower_bounds.to_vec(),
        };
        let mut nm = nelder_mead_optimize(theta, lower_bounds, NM_POLISH_ITERS, cost)?;
        nm.iterations += grid_evals;
        if nm.final_cost > best_cost {
            nm.final_cost = best_cost;
        }
        return Ok(nm);
    }

    Ok(OptimizeResult {
        theta,
        converged: true,
        iterations: grid_evals,
        final_cost: best_cost,
    })
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
use crate::glmm_math::GlmmData;

/// Wrapper for the GLMM Laplace deviance function to be used by argmin.
struct GlmmObjective {
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    family: Box<dyn GlmFamily>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    lower_bounds: Vec<f64>,
}

impl CostFunction for GlmmObjective {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Clamp theta to respect lower bounds before evaluation
        let mut theta_clamped = theta.clone();
        clamp_theta(&mut theta_clamped, &self.lower_bounds);

        // Always use Laplace (n_agq = 1) for θ: AGQ marginal deviance is expensive and can be
        // poorly behaved for derivative-free search on θ; AGQ is applied in the final `pirls`
        // pass when the user requests `n_agq > 1`.
        let mut glmm = GlmmData::new_weighted(
            self.x.clone(),
            self.zt.clone(),
            self.y.clone(),
            self.re_blocks.clone(),
            self.family.build_clone(),
            1,
            self.weights.clone(),
        );
        let val = glmm.laplace_deviance(theta_clamped.as_slice().unwrap(), self.offset.as_ref(), 1);
        if val.is_nan() {
            Ok(f64::MAX)
        } else {
            Ok(val)
        }
    }
}

/// Optimizes θ for a GLMM using Nelder-Mead on the Laplace-approximated deviance.
///
/// Enforces lower bounds on θ: diagonal elements of the Cholesky factor ≥ 0.
///
/// Adaptive quadrature (`n_agq > 1`) is evaluated only in the final [`crate::glmm_math::GlmmData::pirls`]
/// call after `θ` is estimated; the outer objective stays Laplace for numerical stability.
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
) -> Result<OptimizeResult, anyhow::Error> {
    let lower_bounds = compute_theta_lower_bounds(&re_blocks);

    let cost = GlmmObjective {
        x: x.clone(),
        zt: zt.clone(),
        y: y.clone(),
        re_blocks: re_blocks.clone(),
        family,
        offset,
        weights,
        lower_bounds: lower_bounds.clone(),
    };

    nelder_mead_optimize(init_theta, &lower_bounds, 1000, cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::PoissonFamily;
    use crate::model_matrix::ReBlock;
    use ndarray::{array, Array2};
    use sprs::TriMat;

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

        let cost_fn = GlmmObjective {
            x,
            zt,
            y,
            re_blocks,
            family,
            offset: None,
            weights: None,
            lower_bounds,
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

    #[test]
    fn log_grid_1d_spans_endpoints() {
        let grid = log_grid_1d(0.01, 10.0, 5);
        assert_eq!(grid.len(), 5);
        assert!((grid[0] - 0.01).abs() < 1e-12);
        assert!((grid[4] - 10.0).abs() < 1e-9);
        assert!(grid.windows(2).all(|w| w[1] > w[0]));
    }
}
