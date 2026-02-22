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

    let mut zt_tri = sprs::TriMat::new((data.inputs.zt.len(), data.inputs.zt[0].len()));
    for (i, row) in data.inputs.zt.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val != 0.0 {
                zt_tri.add_triplet(i, j, val);
            }
        }
    }
    let zt_arr: sprs::CsMat<f64> = zt_tri.to_csr();

    let y_arr = Array1::from_vec(data.inputs.y);

    let re_blocks = vec![lme_rs::model_matrix::ReBlock { m: 18, k: 1, theta_len: 1 }];
    let model = LmmData::new(x_arr.clone(), zt_arr.clone(), y_arr.clone(), re_blocks.clone());
    let deviance = model.log_reml_deviance(&[data.outputs.theta[0]]);
    
    // Check against LME4 computed REML objective
    println!("lme4 reml_crit: {}, Rust deviance: {}", data.outputs.reml_crit, deviance);
    assert!((deviance - data.outputs.reml_crit).abs() < 1e-6);

    // Run the optimizer and check
    use lme_rs::optimizer::optimize_theta_nd;
    let b_vec = ndarray::Array1::from_vec(vec![1.0]);
    let best_th = optimize_theta_nd(x_arr.clone(), zt_arr.clone(), y_arr.clone(), re_blocks.clone(), b_vec).unwrap();
    println!("lme4 theta: {}, Rust optimized theta: {}", data.outputs.theta[0], best_th[0]);
    assert!((best_th[0] - data.outputs.theta[0]).abs() < 1e-4);

    // Evaluate coefficients
    let coefs = model.evaluate(&data.outputs.theta);
    println!("lme4 beta0: {}, Rust beta0: {}", data.outputs.beta[0], coefs.beta[0]);
    assert!((coefs.beta[0] - data.outputs.beta[0]).abs() < 1e-4);

    // standard errors
    println!("lme4 SE beta0: 9.0499, Rust SE beta0: {}", coefs.beta_se[0]);
    assert!((coefs.beta_se[0] - 9.0499).abs() < 1e-4);
    
    // t-values
    println!("lme4 t beta0: 32.985, Rust t beta0: {}", coefs.beta_t[0]);
    assert!((coefs.beta_t[0] - 32.985).abs() < 1e-3);
}
