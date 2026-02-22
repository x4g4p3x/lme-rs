use lme_rs::math::LmmData;
use ndarray::Array1;
use ndarray::Array2;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TestData {
    outputs: Outputs,
    inputs: Inputs,
}

#[derive(Debug, Deserialize)]
struct Inputs {
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    #[serde(rename = "Zt")]
    zt: Vec<Vec<f64>>,
    y: Vec<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Outputs {
    theta: Vec<f64>,
    beta: Vec<f64>,
    reml_crit: f64,
}

#[test]
fn test_ml_optimization() {
    let file = std::fs::File::open("tests/data/random_slopes.json").expect("dataset not found");
    let data: TestData = serde_json::from_reader(file).unwrap();

    let mut x_arr = Array2::<f64>::zeros((180, 2));
    for (i, row) in data.inputs.x.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            x_arr[[i, j]] = v;
        }
    }

    let mut zt_tri = sprs::TriMat::new((36, 180));
    for (i, row) in data.inputs.zt.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v.abs() > 1e-12 {
                zt_tri.add_triplet(i, j, v);
            }
        }
    }
    let zt_arr: sprs::CsMat<f64> = zt_tri.to_csr();
    let y_arr = Array1::from_vec(data.inputs.y);

    let re_blocks = vec![lme_rs::model_matrix::ReBlock { 
        m: 18, 
        k: 2, 
        theta_len: 3, 
        group_name: "Subject".to_string(), 
        effect_names: vec!["(Intercept)".to_string(), "Days".to_string()],
        group_map: std::collections::HashMap::new(),
    }];

    // Optimize ML directly via Rust
    let initial_theta = ndarray::Array1::from_vec(vec![1.0, 0.0, 1.0]);
    let opt_result = lme_rs::optimizer::optimize_theta_nd(
        x_arr.clone(), zt_arr.clone(), y_arr.clone(), re_blocks.clone(), initial_theta, false, None
    ).unwrap();
    let best_th = opt_result.theta;

    let model = LmmData::new(x_arr, zt_arr, y_arr, re_blocks);
    
    // Evaluate DEV against REML constraint to verify divergence
    let deviance = model.log_reml_deviance(best_th.as_slice().unwrap(), false);
    
    println!("ML deviance: {}", deviance);
    println!("Optimized ML theta: {:?}", best_th.to_vec());
    
    // In LME4, sleepstudy ML Deviance is approx 1751.939
    assert!((deviance - 1751.9).abs() < 1.0);

    // Verify optimized theta matches R's ML variance bound approximation
    // ML estimates are slightly smaller than REML (REML theta0 ~ 0.9667)
    assert!(best_th[0] > 0.9 && best_th[0] < 0.96);

    let coefs = model.evaluate(best_th.as_slice().unwrap(), false);
    println!("ML beta: {:?}", coefs.beta.to_vec());
    assert!((coefs.beta[0] - data.outputs.beta[0]).abs() < 0.1);
    assert!((coefs.beta[1] - data.outputs.beta[1]).abs() < 0.1);
}
