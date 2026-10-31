//! Exact-bit, diagnostic-only fixed-point checks; never replaces a study fit.
use lme_rs::{math::LmmData, FitControl};
use polars::prelude::*;
use serde_json::{json, Value};

fn doubles(raw: &Value, key: &str) -> Vec<f64> {
    raw[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(v.as_u64().unwrap()))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let fixtures: Value = serde_json::from_slice(include_bytes!(
        "../tests/data/stopping_control_counterexample_bits.json"
    ))?;
    let mut cases = Vec::new();
    for raw in fixtures["fixtures"].as_array().unwrap() {
        let groups: Vec<_> = raw["group"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| format!("g{:03}", v.as_u64().unwrap()))
            .collect();
        let data =
            df!["x" => doubles(raw, "x_bits"), "y" => doubles(raw, "y_bits"), "group" => groups]?;
        let formula = "y ~ x + (1 + x | group)";
        let prepared = lme_rs::prepare_lmer(formula, &data)?;
        let design = lme_rs::model_matrix::try_build_fair_lmm_design(formula, &data)?.unwrap();
        let lmm = LmmData::new_weighted_with_zt_z(
            design.x,
            design.zt,
            design.y,
            design.re_blocks,
            None,
            design.precomputed_zt_z,
        );
        let control = FitControl {
            tolerance: 1e-10,
            max_iterations: 5000,
            ..Default::default()
        };
        let fit = prepared.fit(None, true, &control)?;
        let retained = doubles(raw, "theta_bits");
        let theta = fit.theta.as_ref().unwrap().to_vec();
        let mut points = vec![
            ("retained".to_owned(), retained.clone()),
            ("zero_column".to_owned(), vec![0.0, 0.0, retained[2]]),
            ("repaired".to_owned(), theta.clone()),
        ];
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut point = retained.clone();
                point[axis] += sign * f64::EPSILON.sqrt().sqrt() * (1.0 + point[axis].abs());
                if point[axis] < 0.0 && axis != 1 {
                    continue;
                }
                points.push((format!("retained_axis_{axis}_{sign}"), point));
            }
        }
        let evaluations: Vec<_> = points
            .into_iter()
            .map(|(label, theta)| {
                let cached = lmm.log_reml_deviance(&theta, true);
                let full = lmm.evaluate(&theta, true).reml_crit;
                json!({"label": label, "theta": theta, "cached": cached, "full": full})
            })
            .collect();
        cases.push(json!({"case": raw["case"], "replicate": raw["replicate"],
            "control": {"tolerance": control.tolerance, "max_iterations": control.max_iterations},
            "fit": {"theta": theta, "sigma2": fit.sigma2, "criterion": fit.reml,
                "converged": fit.converged, "iterations": fit.diagnostics.unwrap().outer_iterations},
            "evaluations": evaluations}));
    }
    println!("{}", serde_json::to_string_pretty(&cases)?);
    Ok(())
}
