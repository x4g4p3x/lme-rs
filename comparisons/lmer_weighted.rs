use lme_rs::lmer_weighted;
use ndarray::Array1;
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

    let formula = "Reaction ~ Days + (Days | Subject)";
    println!("\nFitting weighted model: {}", formula);
    println!(
        "Prior weights w_i = 0.5 + (row_index mod 5) * 0.1 (same pattern as `benches/bench_math`)"
    );
    println!("Evaluating Restricted Maximum Likelihood (REML)...");

    let n = df.height();
    let weights = Array1::from_vec((0..n).map(|i| 0.5_f64 + (i % 5) as f64 * 0.1).collect());

    let fit = lmer_weighted(formula, &df, true, Some(weights))?;

    println!("\n=== Model Summary ===");
    println!("{}", fit);

    println!("\n=== Predictions ===");
    println!("Generating predictions for Subject 308 at Days 0, 1, 5, and 10...");

    let new_days = Series::new("Days".into(), &[0.0, 1.0, 5.0, 10.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308", "308"]);
    let newdata: DataFrame = DataFrame::new(vec![new_days.into(), new_subject.into()])?;

    let preds = fit.predict(&newdata)?;
    println!("Predictions:\n{:?}", preds);

    Ok(())
}
