use lme_rs::lmer;
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    // 1. Load the dataset
    let mut file = std::fs::File::open("tests/data/penicillin.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // Cast string columns
    let df = df
        .lazy()
        .with_columns(vec![
            col("plate").cast(DataType::String),
            col("sample").cast(DataType::String),
        ])
        .collect()?;

    println!("Fitting model: diameter ~ 1 + (1 | plate) + (1 | sample)");

    // 2. Fit the model
    let formula = "diameter ~ 1 + (1 | plate) + (1 | sample)";
    let fit = lmer(formula, &df, true)?;

    // 3. Print the summary
    println!("\n=== Model Summary ===");
    println!("{}", fit);

    println!("\n=== Predictions ===");
    println!("Generating predictions for new plates and samples...");

    // 4. Generate Predictions
    let new_plate = Series::new("plate".into(), &["a", "b", "c", "d"]);
    let new_sample = Series::new("sample".into(), &["A", "C", "E", "F"]);

    let newdata = DataFrame::new(vec![new_plate.into(), new_sample.into()])?;

    let preds = fit.predict(&newdata)?;
    println!("Predictions:\n{:?}", preds);

    Ok(())
}
