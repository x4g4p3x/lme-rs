use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::brent::BrentOpt;
use ndarray::{Array1, Array2};
use crate::math::LmmData;

/// Wrapper for the REML deviance function to be used by argmin.
struct DevfunCost<'a> {
    pub lmm_data: &'a LmmData,
}

impl<'a> CostFunction for DevfunCost<'a> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Evaluate deviance at theta
        Ok(self.lmm_data.log_reml_deviance(*theta))
    }
}

/// Optimize theta (the variance component) for a simple intercept-only model
/// using Brent's method on the interval [min_theta, max_theta].
pub fn optimize_theta_1d(
    x: Array2<f64>,
    zt: Array2<f64>,
    y: Array1<f64>,
    min_theta: f64,
    max_theta: f64,
) -> Result<f64, Error> {
    let lmm_data = LmmData::new(x, zt, y);
    let cost = DevfunCost { lmm_data: &lmm_data };

    // BrentOpt needs an interval
    let solver = BrentOpt::new(min_theta, max_theta);

    // Run the solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100))
        .run()?;

    // Return the best parameter found
    Ok(res.state().get_best_param().copied().unwrap_or(0.0))
}
