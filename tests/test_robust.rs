use lme_rs::lmer;
use ndarray::Array1;
use polars::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct ModelOutput {
    outputs: Outputs,
}

#[derive(Deserialize)]
struct Outputs {
    robust_v_beta: Vec<f64>,
    robust_se: Vec<f64>,
    robust_t: Vec<f64>,
}

fn load_sleepstudy_data() -> DataFrame {
    let file = std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file)
        .finish()
        .unwrap()
}

#[test]
fn test_robust_se_numerical_parity() {
    let json_str = std::fs::read_to_string("tests/data/random_slopes.json").expect("Failed to read random_slopes.json");
    let test_data: ModelOutput = serde_json::from_str(&json_str).expect("Failed to parse JSON");
    let r_outputs = test_data.outputs;

    let df = load_sleepstudy_data();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Compute CR0 with Subject clustering
    fit.with_robust_se(&df, Some("Subject")).unwrap();
    let robust = fit.robust.as_ref().unwrap();

    let rust_robust_se = &robust.robust_se;
    let rust_robust_t = &robust.robust_t;
    let rust_v_beta = &robust.v_beta_robust;

    // Convert flat R variance matrix to Array2
    let p = 2; // Intercept, Days
    let r_v_beta = ndarray::Array2::from_shape_vec((p, p), r_outputs.robust_v_beta).unwrap();

    let tol = 1e-4;

    for i in 0..p {
        for j in 0..p {
            assert!(
                (rust_v_beta[[i, j]] - r_v_beta[[i, j]]).abs() < tol,
                "V_beta ({}, {}) mismatch. Rust: {}, R: {}",
                i, j, rust_v_beta[[i, j]], r_v_beta[[i, j]]
            );
        }

        assert!(
            (rust_robust_se[i] - r_outputs.robust_se[i]).abs() < tol,
            "Robust SE [{}] mismatch. Rust: {}, R: {}",
            i, rust_robust_se[i], r_outputs.robust_se[i]
        );

        assert!(
            (rust_robust_t[i] - r_outputs.robust_t[i]).abs() < tol,
            "Robust t-value [{}] mismatch. Rust: {}, R: {}",
            i, rust_robust_t[i], r_outputs.robust_t[i]
        );
    }
}
