use crate::math::LmmData;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, Array2};
use sprs::CsMat;

/// Wrapper for the REML deviance function to be used by argmin.
struct DevfunCost<'a> {
    pub lmm_data: &'a LmmData,
}

impl<'a> CostFunction for DevfunCost<'a> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
        // Evaluate deviance at theta.
        // Return infinity or a very large number if the deviance is NaN (invalid region).
        let val = self.lmm_data.log_reml_deviance(theta.as_slice().unwrap());
        if val.is_nan() {
            Ok(f64::MAX)
        } else {
            Ok(val)
        }
    }
}

/// Optimize theta (the variance component vector) using Nelder-Mead.
pub fn optimize_theta_nd(
    x: Array2<f64>,
    zt: CsMat<f64>,
    y: Array1<f64>,
    init_theta: Array1<f64>,
) -> Result<Array1<f64>, Error> {
    let lmm_data = LmmData::new(x, zt, y);
    let cost = DevfunCost { lmm_data: &lmm_data };

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
