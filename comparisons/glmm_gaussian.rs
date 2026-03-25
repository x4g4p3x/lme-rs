use lme_rs::family::Family;
use lme_rs::glmer;
use polars::prelude::*;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let file_path = PathBuf::from("tests").join("data").join("sleepstudy.csv");

    if !file_path.exists() {
        eprintln!("Could not find the dataset at {}", file_path.display());
        eprintln!("Please run this example from the root of the lme-rs repository.");
        std::process::exit(1);
    }

    println!("Loading data from {}...", file_path.display());

    let mut file = std::fs::File::open(&file_path)?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    println!("Successfully loaded {} rows.", df.height());

    let formula = "Reaction ~ Days + (1 | Subject)";
    println!("\nFitting Gaussian GLMM (identity link): {}", formula);
    println!("Laplace / n_agq = 1 (PIRLS matches Gaussian LMM with ML-style objective in this family).");

    let fit = glmer(formula, &df, Family::Gaussian, 1)?;

    println!("\n=== Model Summary ===");
    println!("{}", fit);

    println!("\n=== Predictions (response scale) ===");
    println!("Population-level predictions for Subject 308 at Days 0 and 1...");

    let newdata = DataFrame::new(vec![
        Series::new("Days".into(), &[0.0, 1.0]).into(),
        Series::new("Subject".into(), &["308", "308"]).into(),
    ])?;

    let preds = fit.predict_response(&newdata)?;
    println!("Predictions:\n{:?}", preds);

    Ok(())
}
