#![allow(dead_code)]

use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Deserialize)]
struct TestData {
    pub model: String,
    pub inputs: Inputs,
    pub outputs: Outputs,
}

#[derive(Debug, Deserialize)]
struct Inputs {
    #[serde(rename = "X")]
    pub x: Vec<Vec<f64>>,
    #[serde(rename = "Zt")]
    pub zt: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Outputs {
    pub theta: Vec<f64>,
    pub beta: Vec<f64>,
    pub reml_crit: f64,
}

#[test]
fn test_load_intercept_only_data() {
    let file = File::open("tests/data/intercept_only.json")
        .expect("Failed to open JSON file. Ensure you run `Rscript tests/generate_test_data.R` first.");
    let reader = BufReader::new(file);
    let data: TestData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    assert_eq!(data.model, "Reaction ~ 1 + (1 | Subject)");
    
    // sleepstudy has 18 subjects, 10 days each = 180 observations
    assert_eq!(data.inputs.y.len(), 180);
    assert_eq!(data.inputs.x.len(), 180); // 180 rows
    assert_eq!(data.inputs.x[0].len(), 1); // 1 column (intercept)
    
    assert_eq!(data.outputs.theta.len(), 1);
    assert_eq!(data.outputs.beta.len(), 1);
    
    // Check reasonable expected values for beta
    assert!((data.outputs.beta[0] - 298.508).abs() < 0.1); 

    // Convert Vec<Vec<f64>> manually to Array2 since we want to evaluate LmmData
    use ndarray::{Array1, Array2};
    use lme_rs::math::LmmData;

    let x_arr = Array2::from_shape_vec(
        (data.inputs.x.len(), data.inputs.x[0].len()),
        data.inputs.x.into_iter().flatten().collect(),
    ).unwrap();

    let zt_arr = Array2::from_shape_vec(
        (data.inputs.zt.len(), data.inputs.zt[0].len()),
        data.inputs.zt.into_iter().flatten().collect(),
    ).unwrap();

    let y_arr = Array1::from_vec(data.inputs.y);

    // Provide the theta generated from R and check if our REML deviance matches
    let model = LmmData::new(x_arr.clone(), zt_arr.clone(), y_arr.clone());
    let deviance = model.log_reml_deviance(data.outputs.theta[0]);
    
    // Check against LME4 computed REML objective
    println!("lme4 reml_crit: {}, Rust deviance: {}", data.outputs.reml_crit, deviance);
    assert!((deviance - data.outputs.reml_crit).abs() < 1e-6);

    // Run the optimizer and check if we find the exact same theta!
    use lme_rs::optimizer::optimize_theta_1d;
    let best_theta = optimize_theta_1d(x_arr, zt_arr, y_arr, 1e-5, 10.0).expect("Optimizer failed");

    println!("lme4 theta: {}, Rust optimized theta: {}", data.outputs.theta[0], best_theta);
    // Tolerance for optimization is higher (e.g. 1e-4) because of flat region heuristics
    assert!((best_theta - data.outputs.theta[0]).abs() < 1e-4);
}
