use super::{clamp_theta, covariance_columns, CostFunction, CovarianceRestart, OptimizeResult};
use basin::{
    run_loop_with_control, ApplicationStop, BoxBounds, ConstraintDescription, ConstraintSet,
    ConstraintSolveError, Constraints, NelderMead, Problem, RunControl, SimplexProgress,
    Termination,
};
use ndarray::Array1;
use std::cell::RefCell;
use std::ops::Range;

struct BoundedObjective<C> {
    cost: C,
    lower: Array1<f64>,
    upper: Array1<f64>,
    // A successful LMM recovery probe becomes the first initial vertex.
    // Consume its known cost once; never memoize arbitrary objective history.
    initial_cost: RefCell<Option<(Array1<f64>, f64)>>,
}

impl<C> basin::CostFunction for BoundedObjective<C>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    type Param = Array1<f64>;
    type Output = f64;
    type Error = anyhow::Error;

    fn cost(&self, theta: &Self::Param) -> Result<f64, Self::Error> {
        if let Some((point, value)) = self.initial_cost.borrow_mut().take() {
            if point.len() == theta.len()
                && point
                    .iter()
                    .zip(theta)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Ok(value);
            }
        }
        let value = self.cost.cost(theta)?;
        // Invalid points must sort behind valid candidates without introducing
        // NaNs into simplex ordering or the convergence criterion.
        Ok(if value.is_finite() { value } else { f64::MAX })
    }
}

impl<C> Constraints for BoundedObjective<C>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    type Set = ConstraintSet<BoxBounds<Array1<f64>>>;

    fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new().with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
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

fn coordinate_simplex(theta: &Array1<f64>) -> Vec<Array1<f64>> {
    coordinate_simplex_with_step(theta, 0.2)
}

fn coordinate_simplex_with_step(theta: &Array1<f64>, step: f64) -> Vec<Array1<f64>> {
    let mut simplex = vec![theta.clone()];
    for i in 0..theta.len() {
        let mut vertex = theta.clone();
        vertex[i] += step;
        simplex.push(vertex);
    }
    simplex
}

fn covariance_restart<C>(
    problem: &mut Problem<BoundedObjective<C>>,
    vertices: &[Array1<f64>],
    best_cost: f64,
    columns: &[Range<usize>],
    tolerance: f64,
) -> Result<Option<CovarianceRestart>, anyhow::Error>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    super::covariance_restart_with_eval(vertices, best_cost, columns, tolerance, |point| {
        problem.cost(point)
    })
}

pub(super) fn optimize_lmm(
    init_theta: Array1<f64>,
    lower_bounds: &[f64],
    max_iters: u64,
    tolerance: f64,
    cost: super::LmmObjective,
) -> Result<OptimizeResult, anyhow::Error> {
    let columns = covariance_columns(&cost.lmm.re_blocks);
    optimize_with_columns(
        init_theta,
        lower_bounds,
        max_iters,
        tolerance,
        cost,
        &columns,
    )
}

fn boundary_restart<C>(
    problem: &mut Problem<BoundedObjective<C>>,
    state: &SimplexProgress<Array1<f64>>,
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
    init_theta: Array1<f64>,
    lower_bounds: &[f64],
    max_iters: u64,
    tolerance: f64,
    cost: C,
) -> Result<OptimizeResult, anyhow::Error>
where
    C: CostFunction<Param = Array1<f64>, Output = f64>,
{
    optimize_with_columns(init_theta, lower_bounds, max_iters, tolerance, cost, &[])
}

fn optimize_with_columns<C>(
    mut init_theta: Array1<f64>,
    lower_bounds: &[f64],
    max_iters: u64,
    tolerance: f64,
    cost: C,
    columns: &[Range<usize>],
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
        initial_cost: RefCell::new(None),
    });
    let mut iterations = 0;
    loop {
        let mut control = RunControl::new()
            .max_iter(max_iters - iterations)
            .stop_when(move |state: &SimplexProgress<Array1<f64>>| {
                CostStandardDeviation(tolerance)
                    .converged(state.costs())
                    .then(|| ApplicationStop::new("lme.strict_cost_sd"))
            });
        let result = run_loop_with_control(
            &mut problem,
            SimplexProgress::from_simplex(simplex),
            // Basin 3 enables extra simplex stopping rules by default. The
            // FitControl contract uses only the strict cost-SD hook above.
            &mut NelderMead::new()
                .with_absolute_simplex_size_tolerance(None)
                .with_relative_simplex_size_tolerance(None)
                .with_absolute_simplex_cost_tolerance(None)
                .with_relative_simplex_cost_tolerance(None)
                .projected(),
            &mut control,
        )
        .map_err(|error| match error {
            ConstraintSolveError::Evaluation(error) => error,
            ConstraintSolveError::InvalidConstraints(error) => anyhow::Error::new(error),
            other => anyhow::anyhow!("constraint solver failed: {other}"),
        })?;
        iterations += result.iter();
        let final_cost = result.best_cost();
        let converged = matches!(&result.report.termination, Termination::Application(stop) if stop.code == "lme.strict_cost_sd")
            && final_cost.is_finite()
            && final_cost < f64::MAX;
        if converged && iterations < max_iters {
            if !columns.is_empty() {
                if let Some(restart) = covariance_restart(
                    &mut problem,
                    result.state.vertices(),
                    final_cost,
                    columns,
                    tolerance,
                )? {
                    iterations += 1;
                    if iterations == max_iters {
                        return Ok(OptimizeResult {
                            theta: restart.point,
                            final_cost: restart.cost,
                            converged: false,
                            iterations,
                        });
                    }
                    simplex = coordinate_simplex_with_step(&restart.point, restart.step);
                    *problem.inner().initial_cost.borrow_mut() =
                        Some((restart.point, restart.cost));
                    continue;
                }
            } else if let Some(restarted) = boundary_restart(&mut problem, &result.state)? {
                simplex = restarted;
                // Charge recovery to the same budget, including when a run
                // reaches its cost tolerance before taking an iteration.
                iterations += 1;
                if iterations == max_iters {
                    return Ok(OptimizeResult {
                        theta: result.best_param().clone(),
                        final_cost,
                        converged: false,
                        iterations,
                    });
                }
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

    struct StrictToleranceQuadratic;

    impl CostFunction for StrictToleranceQuadratic {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, theta: &Self::Param) -> Result<f64, anyhow::Error> {
            Ok((theta[0] - 1.3).powi(2) + (theta[1] + 2.0).powi(2))
        }
    }

    #[test]
    fn tight_cost_tolerance_is_not_preempted_by_simplex_collapse() {
        let result = optimize(
            array![0.0, 0.0],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            1000,
            1e-20,
            StrictToleranceQuadratic,
        )
        .unwrap();
        assert!(result.converged, "{result:?}");
        assert!(result.final_cost < 1e-18, "{result:?}");
    }

    struct InitialSpreadExactlyOne;

    impl CostFunction for InitialSpreadExactlyOne {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, theta: &Self::Param) -> Result<f64, anyhow::Error> {
            Ok(5.0 * theta[0] + 10.0 * theta[1])
        }
    }

    #[test]
    fn exact_tolerance_at_initial_simplex_does_not_stop() {
        // The initial objective values are 0, 1, 2; their sample SD is 1.
        let result = optimize(
            array![0.0, 0.0],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            1,
            1.0,
            InitialSpreadExactlyOne,
        )
        .unwrap();
        assert!(!result.converged, "{result:?}");
        assert_eq!(result.iterations, 1);
    }

    #[derive(Debug)]
    struct ObjectiveSentinel;

    impl std::fmt::Display for ObjectiveSentinel {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("objective sentinel")
        }
    }

    impl std::error::Error for ObjectiveSentinel {}

    struct FailingObjective;

    impl CostFunction for FailingObjective {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, _: &Self::Param) -> Result<f64, anyhow::Error> {
            Err(anyhow::Error::new(ObjectiveSentinel))
        }
    }

    #[test]
    fn objective_error_preserves_its_original_type() {
        let error =
            optimize(array![1.0, 1.0], &[0.0, 0.0], 10, 1e-6, FailingObjective).unwrap_err();
        assert!(error.is::<ObjectiveSentinel>(), "{error:#}");
    }

    struct CovarianceQuadratic([f64; 3]);

    impl CostFunction for CovarianceQuadratic {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, theta: &Array1<f64>) -> Result<f64, anyhow::Error> {
            assert!(theta[0] >= 0.0 && theta[2] >= 0.0);
            let covariance = [
                theta[0].powi(2),
                theta[0] * theta[1],
                theta[1].powi(2) + theta[2].powi(2),
            ];
            Ok(covariance
                .iter()
                .zip(self.0)
                .map(|(a, b)| (a - b).powi(2))
                .sum())
        }
    }

    fn covariance_problem(target: [f64; 3]) -> Problem<BoundedObjective<CovarianceQuadratic>> {
        Problem::new(BoundedObjective {
            cost: CovarianceQuadratic(target),
            lower: array![0.0, f64::NEG_INFINITY, 0.0],
            upper: Array1::from_elem(3, f64::INFINITY),
            initial_cost: RefCell::new(None),
        })
    }

    #[test]
    fn covariance_recovery_checks_the_other_column_orientation() {
        let mut problem = covariance_problem([0.01, -0.05, 0.25]);
        let best = array![0.0, 0.5, 0.0];
        let best_cost = problem.cost(&best).unwrap();
        // The same covariance at the boundary has either off-diagonal sign.
        assert_eq!(best_cost, problem.cost(&array![0.0, -0.5, 0.0]).unwrap());
        let restart = covariance_restart(
            &mut problem,
            &vec![best; 4],
            best_cost,
            &[0..2, 2..3],
            1e-12,
        )
        .unwrap()
        .unwrap();
        assert!(restart.point[0] > 0.0 && restart.point[1] < 0.0);
        assert!(restart.cost < best_cost);
    }

    #[test]
    fn covariance_recovery_checks_a_numerically_collapsed_face() {
        let mut problem = covariance_problem([0.25, 0.0, 0.01]);
        let best = array![0.5, 0.0, 1e-16];
        let best_cost = problem.cost(&best).unwrap();
        let restart = covariance_restart(
            &mut problem,
            &vec![best; 4],
            best_cost,
            &[0..2, 2..3],
            1e-12,
        )
        .unwrap()
        .unwrap();
        assert!(restart.cost < 1e-20);
        assert!((restart.point[2] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn covariance_recovery_keeps_a_true_boundary_optimum() {
        let mut problem = covariance_problem([0.0; 3]);
        assert!(covariance_restart(
            &mut problem,
            &vec![Array1::zeros(3); 4],
            0.0,
            &[0..2, 2..3],
            1e-12
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn covariance_column_layout_includes_block_offsets() {
        let block = |k| crate::model_matrix::ReBlock {
            m: 2,
            k,
            theta_len: k * (k + 1) / 2,
            group_name: String::new(),
            effect_names: Vec::new(),
            group_map: Default::default(),
        };
        assert_eq!(
            covariance_columns(&[block(1), block(3), block(2)]),
            vec![0..1, 1..4, 4..6, 6..7, 7..9, 9..10]
        );
        assert!(covariance_columns(&[block(1), block(1)]).is_empty());
    }

    #[test]
    fn covariance_recovery_reverses_a_whole_three_row_column() {
        // One scalar block followed by a 3x3 block. Construct the covariance
        // independently to catch offsets or incomplete off-diagonal reversal.
        fn covariance(t: &Array1<f64>) -> [f64; 7] {
            [
                t[0].powi(2),
                t[1].powi(2),
                t[1] * t[2],
                t[1] * t[3],
                t[2].powi(2) + t[4].powi(2),
                t[2] * t[3] + t[4] * t[5],
                t[3].powi(2) + t[5].powi(2) + t[6].powi(2),
            ]
        }
        struct ThreeRowCost;
        impl CostFunction for ThreeRowCost {
            type Param = Array1<f64>;
            type Output = f64;
            fn cost(&self, theta: &Array1<f64>) -> Result<f64, anyhow::Error> {
                let target = covariance(&array![2.0, 0.1, -0.5, -0.3, 0.4, 0.2, 0.6]);
                Ok(covariance(theta)
                    .iter()
                    .zip(target)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum())
            }
        }
        let mut problem = Problem::new(BoundedObjective {
            cost: ThreeRowCost,
            lower: array![
                0.0,
                0.0,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                0.0,
                f64::NEG_INFINITY,
                0.0
            ],
            upper: Array1::from_elem(7, f64::INFINITY),
            initial_cost: RefCell::new(None),
        });
        let best = array![2.0, 0.0, 0.5, 0.3, 0.4, 0.2, 0.6];
        let cost = problem.cost(&best).unwrap();
        let restart = covariance_restart(
            &mut problem,
            &vec![best; 8],
            cost,
            &[0..1, 1..4, 4..6, 6..7],
            1e-12,
        )
        .unwrap()
        .unwrap();
        assert_eq!(restart.point[0], 2.0);
        assert!(restart.point[1] > 0.0);
        assert_eq!(restart.point[2], -0.5);
        assert_eq!(restart.point[3], -0.3);
        assert_eq!(restart.point.slice(ndarray::s![4..]), array![0.4, 0.2, 0.6]);
        assert!(restart.cost < cost);
    }

    #[test]
    fn known_initial_cost_is_consumed_once_and_only_for_the_same_point() {
        let mut problem = covariance_problem([0.0; 3]);
        let point = array![1.0, 0.0, 0.0];
        *problem.inner().initial_cost.borrow_mut() = Some((point.clone(), 42.0));
        assert_eq!(problem.cost(&point).unwrap(), 42.0);
        assert_eq!(problem.cost(&point).unwrap(), 1.0);
        *problem.inner().initial_cost.borrow_mut() = Some((array![0.0, 0.0, 0.0], 42.0));
        assert_eq!(problem.cost(&point).unwrap(), 1.0);
    }

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
