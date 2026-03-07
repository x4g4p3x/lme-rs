use polars::prelude::*;
use lme_rs::lmer;

fn main() -> anyhow::Result<()> {
    let mut file = std::fs::File::open("tests/data/dyestuff.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let df = df
        .lazy()
        .with_columns(vec![
            col("Batch").cast(DataType::String),
        ])
        .collect()?;

    println!("Fitting model: Yield ~ 1 + (1 | Batch)");

    let formula = "Yield ~ 1 + (1 | Batch)";
    let fit = lmer(formula, &df, true)?;

    println!("\n=== Model Summary ===");
    println!("{}", fit);

    Ok(())
}
