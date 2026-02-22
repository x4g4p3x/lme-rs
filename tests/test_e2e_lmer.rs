use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

#[test]
fn test_lmer_intercept_only_e2e() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read sleepstudy dataset
    let file = File::open("tests/data/sleepstudy.csv").expect("Run generate_test_data.R first");
    let df = CsvReader::new(file)
        .finish()
        .expect("Failed to read CSV");

    // 2. Call our frontend!
    let formula = "Reaction ~ 1 + (1 | Subject)";
    let fit = lmer(formula, &df)?;

    // 3. Verify exactly matching LME4's theta
    // LME4's theta for this model is ~0.80783
    let optimized_theta = fit.sigma2.unwrap(); // We temporarily stored it here
    println!("LME4 theta: 0.80783103775588, Rust lmer() theta: {}", optimized_theta);
    
    // We allow a low tolerance for optimization difference
    assert!((optimized_theta - 0.80783103775588).abs() < 1e-4);

    Ok(())
}
