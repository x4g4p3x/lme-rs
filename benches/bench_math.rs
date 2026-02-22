use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lme_rs::math::LmmData;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use sprs::{CsMat, TriMat};

#[derive(Debug, Deserialize)]
struct TestData {
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

fn load_test_data() -> (LmmData, Vec<f64>) {
    let file = File::open("tests/data/random_slopes.json")
        .expect("Failed to open random_slopes.json for benchmarking.");
    let reader = BufReader::new(file);
    let data: TestData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    let x_arr = Array2::from_shape_vec(
        (data.inputs.x.len(), data.inputs.x[0].len()),
        data.inputs.x.into_iter().flatten().collect(),
    ).unwrap();

    let mut zt_tri = TriMat::new((data.inputs.zt.len(), data.inputs.zt[0].len()));
    for (i, row) in data.inputs.zt.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val != 0.0 {
                zt_tri.add_triplet(i, j, val);
            }
        }
    }
    let zt_arr: CsMat<f64> = zt_tri.to_csr();

    let y_arr = Array1::from_vec(data.inputs.y);

    let re_blocks = vec![lme_rs::model_matrix::ReBlock { m: 18, k: 2, theta_len: 3 }];
    (LmmData::new(x_arr, zt_arr, y_arr, re_blocks), data.outputs.theta)
}

fn bench_deviance_evaluation(c: &mut Criterion) {
    let (model, theta) = load_test_data();

    c.bench_function("log_reml_deviance_random_slopes", |b| {
        b.iter(|| {
            // Evaluate the deviance function multiple times using the same model 
            // as this is the exact closure the optimizer loops over.
            black_box(model.log_reml_deviance(black_box(&[0.8078]), true));
        })
    });
}

criterion_group!(benches, bench_deviance_evaluation);
criterion_main!(benches);
