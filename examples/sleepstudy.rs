use polars::prelude::*;
use lme_rs::lmer;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // 1. Locate the sleepstudy dataset (distributed in tests/data)
    // When running via `cargo run --example`, the working directory is the project root.
    let file_path = PathBuf::from("tests").join("data").join("sleepstudy.csv");
    
    if !file_path.exists() {
        eprintln!("Could not find the dataset at {}", file_path.display());
        eprintln!("Please run this example from the root of the lme-rs repository.");
        std::process::exit(1);
    }

    println!("Loading data from {}...", file_path.display());

    // 2. Load the Polars DataFrame
    let mut file = std::fs::File::open(&file_path)?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    println!("Successfully loaded {} rows.", df.height());

    // 3. Define the Wilkinson formula
    let formula = "Reaction ~ Days + (Days | Subject)";
    
    println!("\nFitting model: {}", formula);
    println!("Evaluating Restricted Maximum Likelihood (REML)...");
    
    // Evaluate standard REML model
    // The `true` parameter indicates we want to use REML instead of ML.
    let fit = lmer(formula, &df, true)?;

    // 4. Print the console summary which mirrors R's output format
    println!("\n=== Model Summary ===");
    println!("{}", fit);

    // 5. Generate Population-Level Predictions
    println!("\n=== Predictions ===");
    println!("Generating predictions for Subject 308 at Days 0, 1, 5, and 10...");
    
    let new_days = Series::new("Days".into(), &[0.0, 1.0, 5.0, 10.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308", "308"]);
    
    // Combine into a new DataFrame
    let newdata: DataFrame = DataFrame::new(vec![new_days.into(), new_subject.into()])?;
    
    // predict expects the new DataFrame
    let preds = fit.predict(&newdata)?;
    println!("Predictions:\n{:?}", preds);

    Ok(())
}
