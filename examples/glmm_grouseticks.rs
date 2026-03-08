use lme_rs::{family::Family, glmer};
use polars::prelude::*;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // 1. Locate the grouseticks dataset (distributed in tests/data)
    let file_path = PathBuf::from("tests").join("data").join("grouseticks.csv");

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

    // 3. Define the Wilkinson formula for the GLMM
    // TICKS is the count of ticks on the grouse chicks
    // YEAR is the year (treated as continuous or categorical)
    // HEIGHT is the altitude
    // BROOD is the family group identifier
    let formula = "TICKS ~ YEAR + HEIGHT + (1 | BROOD)";

    println!("\nFitting Poisson GLMM: {}", formula);
    println!("Evaluating Maximum Likelihood via Laplace Approximation...");

    // Evaluate GLMM with Poisson family and its canonical log link
    let fit = glmer(formula, &df, Family::Poisson)?;

    // 4. Print the summary
    println!("\n=== Model Summary ===");
    println!("{}", fit);

    // 5. Generate Predictions on the Response Scale (counts)
    println!("\n=== Predictions (Response Scale) ===");
    println!("Generating expected tick counts for 3 new broods...");

    let new_year = Series::new("YEAR".into(), &[96, 96, 97]);
    let new_height = Series::new("HEIGHT".into(), &[400, 500, 450]);
    let new_brood = Series::new("BROOD".into(), &["new1", "new2", "new3"]);

    let newdata = DataFrame::new(vec![new_year.into(), new_height.into(), new_brood.into()])?;

    // predict_response applies the inverse-link (exp for Poisson)
    // to give actual expected counts instead of log-counts
    let preds = fit.predict_response(&newdata)?;
    println!("Expected Tick Counts:\n{:?}", preds);

    Ok(())
}
