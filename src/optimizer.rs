use crate::math::LmmData;
use crate::model_matrix::ReBlock;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};
use sprs::CsMat;

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
///   col 0: θ[0] (diagonal), θ[1] (off-diag), ..., θ[k-1] (off-diag)
///   col 1: θ[k] (diagonal), θ[k+1] (off-diag), ...
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

/// Wrapper for the REML deviance function to be used by argmin.
struct LmmObjective {
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    reml: bool,
    lower_bounds: Vec<f64>,
    weights: Option<Array1<f64>>,
}

impl CostFunction for LmmObjective {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Clamp theta to respect lower bounds before evaluation
        let mut theta_clamped = theta.clone();
        clamp_theta(&mut theta_clamped, &self.lower_bounds);

        let lmm = LmmData::new_weighted(
            self.x.clone(),
            self.zt.clone(),
            self.y.clone(),
            self.re_blocks.clone(),
            self.weights.clone(),
        );
        let val = lmm.log_reml_deviance(theta_clamped.as_slice().unwrap(), self.reml);
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
    let lower_bounds = compute_theta_lower_bounds(&re_blocks);

    let cost = LmmObjective {
        x: x.clone(),
        zt: zt.clone(),
        y: y.clone(),
        re_blocks: re_blocks.clone(),
        reml,
        lower_bounds: lower_bounds.clone(),
        weights,
    };

    let n = init_theta.len();
    let max_iters = 1000u64;
    let mut initial_simplex = vec![init_theta.clone()];

    // Create an initial simplex by perturbing each dimension, respecting bounds
    for i in 0..n {
        let mut param = init_theta.clone();
        param[i] += 0.2;
        clamp_theta(&mut param, &lower_bounds);
        initial_simplex.push(param);
    }

    let solver = NelderMead::new(initial_simplex).with_sd_tolerance(1e-6)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()?;

    let state = res.state();
    let mut best_theta = state.get_best_param().cloned().unwrap_or(init_theta);
    clamp_theta(&mut best_theta, &lower_bounds);
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
        let mut glmm = GlmmData::new(
            self.x.clone(),
            self.zt.clone(),
            self.y.clone(),
            self.re_blocks.clone(),
            self.family.build_clone(),
            1,
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
) -> Result<OptimizeResult, anyhow::Error> {
    let lower_bounds = compute_theta_lower_bounds(&re_blocks);

    let cost = GlmmObjective {
        x: x.clone(),
        zt: zt.clone(),
        y: y.clone(),
        re_blocks: re_blocks.clone(),
        family,
        offset,
        lower_bounds: lower_bounds.clone(),
    };

    let n = init_theta.len();
    let max_iters = 1000u64;
    let mut initial_simplex = vec![init_theta.clone()];

    for i in 0..n {
        let mut param = init_theta.clone();
        param[i] += 0.2;
        clamp_theta(&mut param, &lower_bounds);
        initial_simplex.push(param);
    }

    let solver = NelderMead::new(initial_simplex).with_sd_tolerance(1e-6)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()?;

    let state = res.state();
    let mut best_theta = state.get_best_param().cloned().unwrap_or(init_theta);
    clamp_theta(&mut best_theta, &lower_bounds);
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
            lower_bounds,
        };

        let theta = array![1.0];
        let cost = cost_fn.cost(&theta).unwrap();
        assert_eq!(cost, f64::MAX);
    }
}
