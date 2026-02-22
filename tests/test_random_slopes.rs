use lme_rs::math::LmmData;
use ndarray::{Array1, Array2};
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
fn test_load_random_slopes_data() {
    let file = File::open("tests/data/random_slopes.json")
        .expect("Failed to open JSON file. Ensure you run `Rscript tests/generate_test_data.R` first.");
    let reader = BufReader::new(file);
    let data: TestData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    assert_eq!(data.model, "Reaction ~ Days + (Days | Subject)");
    
    // Check reasonable expected values for theta (now a vector of length 3)
    // lme4 produces: [0.9667, 0.0151, 0.2309]
    assert_eq!(data.outputs.theta.len(), 3);
    assert!((data.outputs.theta[0] - 0.9667).abs() < 0.1); 

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

    let re_blocks = vec![lme_rs::model_matrix::ReBlock { m: 18, k: 2, theta_len: 3 }];
    let model = LmmData::new(x_arr, zt_arr, y_arr, re_blocks);
    
    // Evaluate deviance using the newly structured array
    let deviance = model.log_reml_deviance(&data.outputs.theta);
    
    // Check against LME4 computed REML objective
    println!("lme4 reml_crit: {}, Rust deviance: {}", data.outputs.reml_crit, deviance);
    assert!((deviance - data.outputs.reml_crit).abs() < 1e-6);

    // Evaluate coefficients
    let coefs = model.evaluate(&data.outputs.theta);
    println!("lme4 beta0: {}, Rust beta0: {}", data.outputs.beta[0], coefs.beta[0]);
    println!("lme4 beta1: {}, Rust beta1: {}", data.outputs.beta[1], coefs.beta[1]);
    assert!((coefs.beta[0] - data.outputs.beta[0]).abs() < 1e-4);
    assert!((coefs.beta[1] - data.outputs.beta[1]).abs() < 1e-4);

    println!("lme4 SE beta0: 6.8246, Rust SE beta0: {}", coefs.beta_se[0]);
    println!("lme4 SE beta1: 1.5458, Rust SE beta1: {}", coefs.beta_se[1]);
    assert!((coefs.beta_se[0] - 6.8246).abs() < 1e-4);
    assert!((coefs.beta_se[1] - 1.5458).abs() < 1e-4);

    println!("lme4 t beta0: 36.838, Rust t beta0: {}", coefs.beta_t[0]);
    println!("lme4 t beta1: 6.771, Rust t beta1: {}", coefs.beta_t[1]);
    assert!((coefs.beta_t[0] - 36.838).abs() < 1e-3);
    assert!((coefs.beta_t[1] - 6.771).abs() < 1e-3);
}
