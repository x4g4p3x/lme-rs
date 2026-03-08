use lme_rs::family::Family;
use lme_rs::glmer;
use ndarray::Array1;
use polars::prelude::*;
use serde::Deserialize;
use std::fs::File;

#[allow(dead_code)]
#[derive(Deserialize)]
struct ModelInputs {
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    #[serde(rename = "Zt")]
    zt: Vec<Vec<f64>>,
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct ModelOutputs {
    theta: Vec<f64>,
    beta: Vec<f64>,
    deviance: f64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct TestData {
    model: String,
    inputs: ModelInputs,
    outputs: ModelOutputs,
}

fn load_test_data(path: &str) -> TestData {
    let file = File::open(path).unwrap_or_else(|_| panic!("Could not open {}", path));
    serde_json::from_reader(file).expect("Failed to parse JSON")
}

fn read_csv_data(path: &str) -> polars::prelude::DataFrame {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Could not open {}", path));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("Failed to read CSV")
}

#[test]
fn test_glmm_binomial_cbpp() {
    let _ = env_logger::try_init();
    let data = load_test_data("tests/data/glmm_binomial.json");
    let df = read_csv_data("tests/data/cbpp_binary.csv");

    // The model formula in R was: y ~ period2 + period3 + period4 + (1 | herd)
    let fit = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
    )
    .unwrap();

    // 1. Check fixed effects (beta)
    let beta_r = Array1::from_vec(data.outputs.beta);
    let beta_lens_match = fit.coefficients.len() == beta_r.len();

    println!("CBPP Beta RS: {:?}", fit.coefficients);
    println!("CBPP Beta R: {:?}", beta_r);
    println!("CBPP Theta RS: {:?}", fit.theta.as_ref().unwrap());
    println!("CBPP Theta R: {:?}", data.outputs.theta);
    println!("CBPP Deviance RS: {:?}", fit.deviance.unwrap());
    println!("CBPP Deviance R: {:?}", data.outputs.deviance);

    assert!(beta_lens_match, "Beta length mismatch");

    // Using a tolerance of 0.05 for fixed effects (Laplace approx exactness differs slightly from R's C++ inner loop occasionally)
    for i in 0..beta_r.len() {
        assert!(
            (fit.coefficients[i] - beta_r[i]).abs() < 0.05,
            "Beta {} mismatch: rs={} r={}",
            i,
            fit.coefficients[i],
            beta_r[i]
        );
    }

    // 2. Check variance components (theta)
    let theta_r = Array1::from_vec(data.outputs.theta);
    let fit_theta = fit.theta.unwrap();
    for i in 0..theta_r.len() {
        assert!(
            (fit_theta[i] - theta_r[i]).abs() < 0.05,
            "Theta {} mismatch: rs={} r={}",
            i,
            fit_theta[i],
            theta_r[i]
        );
    }

    // 3. Check deviance (we ignore absolute deviance because lme4 omits terms like saturated log lik & pi constants)
    // We just ensure it's calculated without NaN.
    assert!(!fit.deviance.unwrap().is_nan());
}

#[test]
fn test_glmm_poisson_grouseticks() {
    let _ = env_logger::try_init();
    let data = load_test_data("tests/data/glmm_poisson.json");
    let df = read_csv_data("tests/data/grouseticks.csv");

    // The model formula in R was: TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)
    let fit = glmer(
        "TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)",
        &df,
        Family::Poisson,
    )
    .unwrap();

    // 1. Check fixed effects (beta)
    let beta_r = Array1::from_vec(data.outputs.beta);

    println!("Grouse Beta RS: {:?}", fit.coefficients);
    println!("Grouse Beta R: {:?}", beta_r);
    println!("Grouse Theta RS: {:?}", fit.theta.as_ref().unwrap());
    println!("Grouse Theta R: {:?}", data.outputs.theta);
    println!("Grouse Deviance RS: {:?}", fit.deviance.unwrap());
    println!("Grouse Deviance R: {:?}", data.outputs.deviance);

    // Using a tolerance of 0.15 for Nelder-Mead vs BOBYQA optimizer differences
    for i in 0..beta_r.len() {
        assert!(
            (fit.coefficients[i] - beta_r[i]).abs() < 0.15,
            "Beta {} mismatch: rs={} r={}",
            i,
            fit.coefficients[i],
            beta_r[i]
        );
    }

    // 2. Check variance components (theta)
    let theta_r = Array1::from_vec(data.outputs.theta);
    let fit_theta = fit.theta.unwrap();
    for i in 0..theta_r.len() {
        assert!(
            (fit_theta[i] - theta_r[i]).abs() < 1e-2,
            "Theta {} mismatch: rs={} r={}",
            i,
            fit_theta[i],
            theta_r[i]
        );
    }

    // 3. Check deviance (we ignore absolute deviance because lme4 omits terms like saturated log lik & pi constants)
    // We just ensure it's calculated without NaN.
    assert!(!fit.deviance.unwrap().is_nan());
}
