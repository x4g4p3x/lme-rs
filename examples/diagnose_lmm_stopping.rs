//! Diagnostic-only replay of a retained LMM input; does not change fit semantics.
//! Input uses IEEE-754 bits so different JSON parsers preserve identical doubles.
use argmin::core::{
    CostFunction, Error, Executor, IterState, Problem, Solver, State, TerminationStatus, KV,
};
use argmin::solver::neldermead::NelderMead;
use lme_rs::{math::LmmData, FitControl};
use ndarray::Array1;
use polars::prelude::*;
use serde_json::{json, Value};
use std::{cell::RefCell, rc::Rc, sync::Arc};

type SearchState = IterState<Vec<f64>, (), (), (), (), f64>;
type Simplex = NelderMead<Vec<f64>, f64>;

#[derive(Clone)]
struct Objective {
    lmm: Arc<LmmData>,
    evaluations: Rc<RefCell<Vec<Value>>>,
}

impl CostFunction for Objective {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, param: &Vec<f64>) -> Result<f64, Error> {
        let mut theta = param.clone();
        theta[0] = theta[0].max(0.0);
        theta[2] = theta[2].max(0.0);
        let cost = self.lmm.log_reml_deviance(&theta, true);
        self.evaluations
            .borrow_mut()
            .push(json!({"theta": theta, "cost": cost}));
        Ok(cost)
    }
}

// The dependency keeps its simplex private. Query its unchanged terminate method
// on clones to bracket the actual sample SD; no extra objective calls occur.
fn sd_upper(solver: &Simplex, state: &SearchState) -> f64 {
    let (mut low, mut high) = (0.0, 1.0e6);
    for _ in 0..100 {
        let middle = (low + high) / 2.0;
        let mut probe = solver.clone().with_sd_tolerance(middle).unwrap();
        if matches!(
            <Simplex as Solver<Objective, SearchState>>::terminate(&mut probe, state),
            TerminationStatus::Terminated(_)
        ) {
            high = middle;
        } else {
            low = middle;
        }
    }
    high
}

struct TraceSolver {
    inner: Simplex,
    checkpoints: Rc<RefCell<Vec<Value>>>,
}

impl Solver<Objective, SearchState> for TraceSolver {
    fn name(&self) -> &str {
        "Diagnostic Argmin delegate"
    }
    fn init(
        &mut self,
        problem: &mut Problem<Objective>,
        state: SearchState,
    ) -> Result<(SearchState, Option<KV>), Error> {
        self.inner.init(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<Objective>,
        state: SearchState,
    ) -> Result<(SearchState, Option<KV>), Error> {
        self.inner.next_iter(problem, state)
    }
    fn terminate(&mut self, state: &SearchState) -> TerminationStatus {
        if [0, 100, 200, 500, 1000, 2000, 4000].contains(&state.get_iter()) {
            self.checkpoints.borrow_mut().push(json!({
                "iteration": state.get_iter(), "cost": state.get_best_cost(),
                "theta": state.get_best_param(), "sample_sd_upper": sd_upper(&self.inner, state)
            }));
        }
        <Simplex as Solver<Objective, SearchState>>::terminate(&mut self.inner, state)
    }
}

fn floats(input: &Value, key: &str) -> Vec<f64> {
    input[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let input: Value = serde_json::from_slice(&std::fs::read(
        std::env::args().nth(1).expect("input JSON path"),
    )?)?;
    let x = floats(&input, "x_bits");
    let y = floats(&input, "y_bits");
    let retained = floats(&input, "theta_bits");
    let groups: Vec<String> = input["group"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| format!("g{:03}", g.as_u64().unwrap()))
        .collect();
    let data = df!["x" => x, "y" => y, "group" => groups]?;
    let formula = "y ~ x + (1 + x | group)";
    let prepared = lme_rs::prepare_lmer(formula, &data)?;
    let matrices = lme_rs::model_matrix::try_build_fair_lmm_design(formula, &data)?.unwrap();
    let lmm = Arc::new(LmmData::new_weighted_with_zt_z(
        matrices.x,
        matrices.zt,
        matrices.y,
        matrices.re_blocks,
        None,
        matrices.precomputed_zt_z,
    ));
    let mut public_fits = Vec::new();
    for tolerance in [1e-8, 1e-10, 1e-12, 1e-14] {
        for start in [None, Some(Array1::from_vec(retained.clone()))] {
            let label = if start.is_some() {
                "retained"
            } else {
                "default"
            };
            let control = FitControl {
                tolerance,
                max_iterations: 5000,
                start,
                ..Default::default()
            };
            let fit = prepared.fit(None, true, &control)?;
            let diag = fit.diagnostics.unwrap();
            public_fits.push(json!({"start": label, "tolerance": tolerance, "converged": fit.converged, "iterations": diag.outer_iterations, "termination": format!("{:?}", diag.termination), "objective": diag.objective, "criterion": fit.reml, "sigma2": fit.sigma2, "theta": fit.theta.unwrap().to_vec()}));
        }
    }
    let evaluations = Rc::new(RefCell::new(Vec::new()));
    let checkpoints = Rc::new(RefCell::new(Vec::new()));
    let objective = Objective {
        lmm: Arc::clone(&lmm),
        evaluations: Rc::clone(&evaluations),
    };
    let simplex = vec![
        vec![1.0, 1.0, 1.0],
        vec![1.2, 1.0, 1.0],
        vec![1.0, 1.2, 1.0],
        vec![1.0, 1.0, 1.2],
    ];
    let solver = TraceSolver {
        inner: Simplex::new(simplex).with_sd_tolerance(1e-12)?,
        checkpoints: Rc::clone(&checkpoints),
    };
    let result = Executor::new(objective, solver)
        .configure(|s| s.max_iters(5000))
        .run()?;
    let final_trace = json!({"iterations": result.state.get_iter(), "termination": format!("{:?}", result.state.get_termination_reason()), "cost": result.state.get_best_cost(), "theta": result.state.get_best_param(), "sample_sd_upper": sd_upper(&result.solver.inner, &result.state)});
    let mut repeats = Vec::new();
    let mut neighbors = Vec::new();
    for _ in 0..100 {
        repeats.push(json!({"cached": lmm.log_reml_deviance(&retained, true), "full": lmm.evaluate(&retained, true).reml_crit}));
    }
    for step in [1e-8, 1e-7, 1e-6, 1e-4] {
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut point = retained.clone();
                point[axis] += step * sign;
                neighbors.push(json!({"step": step, "axis": axis, "sign": sign, "theta": point, "cached": lmm.log_reml_deviance(&point, true), "full": lmm.evaluate(&point, true).reml_crit}));
            }
        }
    }
    let repeat_after_neighbors = lmm.log_reml_deviance(&retained, true);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "scope": "post-study diagnostic only; original pilot unchanged", "basin_feature": cfg!(feature="basin"),
            "public_fits": public_fits, "argmin_checkpoints": *checkpoints.borrow(), "argmin_final": final_trace,
            "argmin_evaluations": *evaluations.borrow(), "repeats": repeats, "neighbors": neighbors,
            "repeat_after_neighbors": repeat_after_neighbors
        }))?
    );
    Ok(())
}
