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
}

#[test]
fn test_mock_crossed_effects() {
    let file = File::open("tests/data/mock_crossed.json").expect("Failed to open JSON file.");
    let reader = BufReader::new(file);
    let data: TestData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    assert_eq!(data.model, "y ~ 1 + (1 | A) + (1 | B)");

    // Convert test inputs to arrays
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

    // Mock dataset has two crossed factors, A (k=1, m=2) and B (k=1, m=2). 
    // Both intercept-only, so theta_len = 1 each
    let re_blocks = vec![
        lme_rs::model_matrix::ReBlock { m: 2, k: 1, theta_len: 1, group_name: "A".to_string(), effect_names: vec!["(Intercept)".to_string()] },
        lme_rs::model_matrix::ReBlock { m: 2, k: 1, theta_len: 1, group_name: "B".to_string(), effect_names: vec!["(Intercept)".to_string()] },
    ];

    let model = LmmData::new(x_arr.clone(), zt_arr.clone(), y_arr.clone(), re_blocks.clone());
    
    // Evaluate deviance using static array mapping
    let deviance = model.log_reml_deviance(&data.outputs.theta, true);
    
    println!("Crossed effects deviance evaluating successfully: {}", deviance);
    assert!(!deviance.is_nan());

    // Try optimizing
    let initial_theta = ndarray::Array1::from_vec(vec![1.0, 1.0]);
    let best_th = lme_rs::optimizer::optimize_theta_nd(
        x_arr.clone(), zt_arr.clone(), y_arr.clone(), re_blocks.clone(), initial_theta, true
    ).unwrap();

    println!("Best theta array recovered: {:?}", best_th);
    assert!(best_th.len() == 2);
}
