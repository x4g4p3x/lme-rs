use super::{clamp_theta, CostFunction, OptimizeResult};
use basin::{run_loop, BasicSimplexState, BoxConstraints, NelderMead, Problem, SimplexState};
use basin::{TerminationCriterion, TerminationReason};
use ndarray::Array1;

struct BoundedObjective<C> {
    cost: C,
    lower: Array1<f64>,
    upper: Array1<f64>,
}

impl<C> basin::CostFunction for BoundedObjective<C>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    type Param = Array1<f64>;
    type Output = f64;
    type Error = anyhow::Error;

    fn cost(&self, theta: &Self::Param) -> Result<f64, Self::Error> {
        let value = self.cost.cost(theta)?;
        // Invalid points must sort behind valid candidates without introducing
        // NaNs into simplex ordering or the convergence criterion.
        Ok(if value.is_finite() { value } else { f64::MAX })
    }
}

impl<C> BoxConstraints for BoundedObjective<C>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    fn lower(&self) -> &Self::Param {
        &self.lower
    }

    fn upper(&self) -> &Self::Param {
        &self.upper
    }
}

struct CostStandardDeviation(f64);

impl CostStandardDeviation {
    fn converged(&self, costs: &[f64]) -> bool {
        if costs.len() < 2 || costs.iter().any(|c| !c.is_finite() || *c == f64::MAX) {
            return false;
        }
        // FitControl specifies Argmin's sample standard deviation, rather than
        // Basin's built-in joint position-and-cost simplex-collapse criterion.
        let n = costs.len() as f64;
        let mean = costs.iter().sum::<f64>() / n;
        let sd = (costs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        sd < self.0
    }
}

impl TerminationCriterion<BasicSimplexState<Array1<f64>>> for CostStandardDeviation {
    fn check(&mut self, state: &BasicSimplexState<Array1<f64>>) -> Option<TerminationReason> {
        self.converged(state.costs())
            .then_some(TerminationReason::SolverConverged)
    }
}

fn coordinate_simplex(theta: &Array1<f64>) -> Vec<Array1<f64>> {
    let mut simplex = vec![theta.clone()];
    for i in 0..theta.len() {
        let mut vertex = theta.clone();
        vertex[i] += 0.2;
        simplex.push(vertex);
    }
    simplex
}

fn boundary_restart<C>(
    problem: &mut Problem<BoundedObjective<C>>,
    state: &BasicSimplexState<Array1<f64>>,
) -> Result<Option<Vec<Array1<f64>>>, anyhow::Error>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    let best = &state.vertices()[0];
    for i in 0..best.len() {
        let lower = problem.inner().lower[i];
        if !lower.is_finite() || !state.vertices().iter().all(|vertex| vertex[i] == lower) {
            continue;
        }

        // Projection can remove every inward direction from the simplex.
        // Backtracking avoids skipping minima closer to the bound than the
        // original coordinate step, while limiting probes at numerical resolution.
        let scale = lower.abs().max(1.0);
        let mut step = 0.2 * scale;
        while step >= f64::EPSILON.sqrt() * scale {
            let mut probe = best.clone();
            probe[i] += step;
            if probe[i].is_finite() && problem.cost(&probe)? < state.costs()[0] {
                // Retaining the best point makes recovery monotone, and a full
                // simplex lets the other coordinates adjust to the inward move.
                let mut simplex = coordinate_simplex(best);
                simplex[i + 1] = probe;
                return Ok(Some(simplex));
            }
            step *= 0.5;
        }
    }
    Ok(None)
}

pub(crate) fn optimize<C>(
    mut init_theta: Array1<f64>,
    lower_bounds: &[f64],
    max_iters: u64,
    tolerance: f64,
    cost: C,
) -> Result<OptimizeResult, anyhow::Error>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    anyhow::ensure!(
        !init_theta.is_empty()
            && init_theta.len() == lower_bounds.len()
            && init_theta.iter().all(|v| v.is_finite())
            && lower_bounds
                .iter()
                .all(|v| v.is_finite() || *v == f64::NEG_INFINITY),
        "Nelder-Mead requires finite starting parameters and matching valid lower bounds"
    );
    anyhow::ensure!(
        tolerance.is_finite() && tolerance > 0.0,
        "Nelder-Mead tolerance must be positive and finite"
    );

    // Project before adding coordinate steps so an infeasible start does not
    // collapse every initial vertex onto the same boundary point.
    clamp_theta(&mut init_theta, lower_bounds);
    let n = init_theta.len();
    let mut simplex = coordinate_simplex(&init_theta);
    let mut problem = Problem::new(BoundedObjective {
        cost,
        lower: Array1::from_vec(lower_bounds.to_vec()),
        upper: Array1::from_elem(n, f64::INFINITY),
    });
    let mut iterations = 0;
    loop {
        let result = run_loop(
            &mut problem,
            BasicSimplexState::from_simplex(simplex),
            &mut NelderMead::new().projected(),
            &mut [Box::new(CostStandardDeviation(tolerance))],
            max_iters - iterations,
        )?;
        iterations += result.iter();
        let final_cost = result.best_cost();
        let converged = result.reason == TerminationReason::SolverConverged
            && final_cost.is_finite()
            && final_cost < f64::MAX;
        if converged {
            if let Some(restarted) = boundary_restart(&mut problem, &result.state)? {
                simplex = restarted;
                // Charge recovery to the same budget, including when a run
                // reaches its cost tolerance before taking an iteration.
                iterations += 1;
                continue;
            }
        }
        return Ok(OptimizeResult {
            theta: result.best_param().clone(),
            converged,
            iterations,
            final_cost,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct CoupledQuadratic {
        target: f64,
        lower: f64,
    }

    impl CostFunction for CoupledQuadratic {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, theta: &Self::Param) -> Result<f64, anyhow::Error> {
            assert!(theta[0] >= self.lower, "infeasible evaluation: {theta:?}");
            let u = theta[0] - self.lower - self.target;
            let v = theta[1] + 5.0;
            Ok(u * u + u * v + v * v)
        }
    }

    #[test]
    fn collapsed_boundary_simplex_recovers_interior_minimum() {
        for lower in [0.0, -2.0, 3.0] {
            for target in [0.05, 1e-5] {
                let result = optimize(
                    array![lower + 1.0, 0.0],
                    &[lower, f64::NEG_INFINITY],
                    1000,
                    1e-16,
                    CoupledQuadratic { target, lower },
                )
                .unwrap();
                assert!(result.converged, "{result:?}");
                assert!(result.final_cost < 1e-14, "{result:?}");
                assert!(
                    (result.theta[0] - lower - target).abs() < 1e-6,
                    "{result:?}"
                );
                assert!((result.theta[1] + 5.0).abs() < 1e-6, "{result:?}");
            }
        }
    }

    #[test]
    fn collapsed_boundary_simplex_can_converge_at_boundary_minimum() {
        let result = optimize(
            array![1.0, 0.0],
            &[0.0, f64::NEG_INFINITY],
            1000,
            1e-12,
            CoupledQuadratic {
                target: -0.05,
                lower: 0.0,
            },
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert_eq!(result.theta[0], 0.0);
        assert!((result.theta[1] + 5.025).abs() < 1e-5, "{result:?}");
        assert!((result.final_cost - 0.001875).abs() < 1e-11, "{result:?}");
    }

    #[test]
    fn boundary_restarts_share_the_iteration_budget() {
        let result = optimize(
            array![1.0, 0.0],
            &[0.0, f64::NEG_INFINITY],
            60,
            1e-12,
            CoupledQuadratic {
                target: 0.05,
                lower: 0.0,
            },
        )
        .unwrap();
        assert!(!result.converged, "{result:?}");
        assert_eq!(result.iterations, 60);
        assert!(result.final_cost < 0.001875, "{result:?}");
    }

    #[test]
    fn random_slopes_recover_nonzero_variance_from_difficult_starts() {
        use polars::prelude::*;

        let data = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some("tests/data/sleepstudy.csv".into()))
            .unwrap()
            .finish()
            .unwrap();
        let prepared = crate::prepare_lmer("Reaction ~ Days + (Days | Subject)", &data).unwrap();
        let bounds = [0.0, f64::NEG_INFINITY, 0.0];
        for start in [
            array![1.0, -5.0, 1.0],
            array![0.01, -5.0, 0.01],
            array![10.0, 5.0, 0.01],
        ] {
            let result = optimize(
                start,
                &bounds,
                1000,
                1e-10,
                super::super::LmmObjective {
                    lmm: prepared.lmm.clone(),
                    reml: true,
                    lower_bounds: bounds.to_vec(),
                },
            )
            .unwrap();
            assert!(result.converged, "{result:?}");
            for (actual, expected) in result.theta.iter().zip([0.9667, 0.0151, 0.2309]) {
                assert!((actual - expected).abs() < 1e-3, "{result:?}");
            }
        }
    }

    #[test]
    fn sample_standard_deviation_uses_strict_tolerance() {
        assert!(!CostStandardDeviation(1.0).converged(&[1.0, 2.0, 3.0]));
        assert!(CostStandardDeviation(1.001).converged(&[1.0, 2.0, 3.0]));
        assert!(!CostStandardDeviation(1.1).converged(&[1.0, 3.0]));
        assert!(CostStandardDeviation(1e-6).converged(&[-10.0, -10.0]));
    }

    #[test]
    fn invalid_simplex_costs_cannot_establish_convergence() {
        for value in [f64::MAX, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(!CostStandardDeviation(1e-6).converged(&[value, value]));
            assert!(!CostStandardDeviation(1e-6).converged(&[0.0, value]));
        }
    }
}
