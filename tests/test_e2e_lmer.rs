use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

#[test]
fn test_lmer_intercept_only_e2e() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("tests/data/sleepstudy.csv").expect("Run generate_test_data.R first");
    let df = CsvReader::new(file).finish().expect("Failed to read CSV");

    // Call frontend
    let formula = "Reaction ~ 1 + (1 | Subject)";
    let fit = lmer(formula, &df)?;

    // Optimization check
    let optimized_theta = fit.theta.unwrap()[0];
    println!("LME4 theta: 0.80783103775588, Rust lmer() theta: {}", optimized_theta);
    assert!((optimized_theta - 0.80783103775588).abs() < 1e-4);

    // Beta check
    let beta = fit.coefficients[0];
    println!("LME4 beta: 298.508, Rust beta: {}", beta);
    assert!((beta - 298.508).abs() < 0.1); 

    Ok(())
}

#[test]
fn test_lmer_random_slopes_e2e() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("tests/data/sleepstudy.csv").expect("Run generate_test_data.R first");
    let df = CsvReader::new(file).finish().expect("Failed to read CSV");

    // Call frontend with random slopes
    let formula = "Reaction ~ Days + (Days | Subject)";
    let fit = lmer(formula, &df)?;

    let optimized_theta0 = fit.theta.unwrap()[0]; 
    println!("LME4 theta0: 0.9667, Rust lmer() theta0: {}", optimized_theta0);
    assert!((optimized_theta0 - 0.96673).abs() < 0.1);

    // Beta check
    let beta0 = fit.coefficients[0];
    let beta1 = fit.coefficients[1];
    println!("LME4 beta0: 251.405, Rust beta0: {}", beta0);
    println!("LME4 beta1: 10.467, Rust beta1: {}", beta1);
    assert!((beta0 - 251.405).abs() < 0.1); 
    assert!((beta1 - 10.467).abs() < 0.1); 

    Ok(())
}
