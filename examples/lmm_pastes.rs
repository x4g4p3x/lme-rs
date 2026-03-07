use polars::prelude::*;
use lme_rs::lmer;

fn main() -> anyhow::Result<()> {
    // 1. Load the dataset
    let mut file = std::fs::File::open("tests/data/pastes.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // Cast string columns
    let df = df
        .lazy()
        .with_columns(vec![
            col("batch").cast(DataType::String),
            col("cask").cast(DataType::String),
        ])
        .collect()?;

    println!("Fitting model: strength ~ 1 + (1 | batch/cask)");

    // 2. Fit the model
    let formula = "strength ~ 1 + (1 | batch/cask)";
    let fit = lmer(formula, &df, true)?;

    // 3. Print the summary
    println!("\n=== Model Summary ===");
    println!("{}", fit);

    println!("\n=== Predictions ===");
    println!("Generating predictions for population-level...");

    // 4. Generate Predictions
    let new_batch = Series::new("batch".into(), &["A", "B", "C"]);
    let new_cask = Series::new("cask".into(), &["a", "b", "c"]);

    let newdata = DataFrame::new(vec![new_batch.into(), new_cask.into()])?;

    let preds = fit.predict(&newdata)?;
    println!("Predictions:\n{:?}", preds);

    Ok(())
}
