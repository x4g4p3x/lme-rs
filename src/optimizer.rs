use crate::math::LmmData;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};
use sprs::CsMat;
use crate::model_matrix::ReBlock;

/// Wrapper for the REML deviance function to be used by argmin.
struct LmmObjective {
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    reml: bool,
}

impl CostFunction for LmmObjective {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Evaluate deviance at theta.
        // Return infinity or a very large number if the deviance is NaN (invalid region).
        let lmm = LmmData::new(
            self.x.clone(),
            self.zt.clone(),
            self.y.clone(),
            self.re_blocks.clone(),
        );
        let val = lmm.log_reml_deviance(theta.as_slice().unwrap(), self.reml);
        if val.is_nan() {
            Ok(f64::MAX)
        } else {
            Ok(val)
        }
    }
}

/// Optimizes $\theta$ (the variance component vector) using Nelder-Mead on an un-gradiented search space.
pub fn optimize_theta_nd(
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    re_blocks: Vec<ReBlock>,
    init_theta: Array1<f64>,
    reml: bool,
) -> Result<Array1<f64>, anyhow::Error> {
    let cost = LmmObjective {
        x: x.clone(),
        zt: zt.clone(),
        y: y.clone(),
        re_blocks: re_blocks.clone(),
        reml,
    };

    let n = init_theta.len();
    let mut initial_simplex = vec![init_theta.clone()];
    
    // Create an initial simplex by perturbing each dimension
    for i in 0..n {
        let mut param = init_theta.clone();
        param[i] += 0.2; // simple perturbation factor
        initial_simplex.push(param);
    }

    let solver = NelderMead::new(initial_simplex)
        .with_sd_tolerance(1e-6)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(1000))
        .run()?;

    Ok(res.state().get_best_param().cloned().unwrap_or(init_theta))
}
