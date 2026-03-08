use lme_rs::glmer;
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    // 1. Load the dataset
    let mut file = std::fs::File::open("tests/data/cbpp_binary.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // Need to cast `herd` to String so lme-rs recognizes it as a group
    let df = df
        .lazy()
        .with_column(col("herd").cast(DataType::String))
        .collect()?;

    println!("Fitting Binomial GLMM: y ~ period2 + period3 + period4 + (1 | herd)");

    // 2. Fit the model
    // Note: glmer syntax takes the formula, the dataframe, Family::Binomial, and Link::Logit
    // Currently, lme-rs assumes Logit as the default link for Binomial if used via parsing wrapper,
    // or we can just pass the string "binomial" to match similar internal tests
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let fit = glmer(formula, &df, lme_rs::family::Family::Binomial)?;

    // 3. Print the summary
    println!("\n=== Model Summary ===");
    println!("{}", fit);

    println!("\n=== Predictions (Probabilities) ===");
    println!("Generating predictions for herd 1 across periods...");

    // 4. Generate Predictions
    let new_herd = Series::new("herd".into(), &["1", "1", "1", "1"]);
    let new_p2 = Series::new("period2".into(), &[0.0, 1.0, 0.0, 0.0]);
    let new_p3 = Series::new("period3".into(), &[0.0, 0.0, 1.0, 0.0]);
    let new_p4 = Series::new("period4".into(), &[0.0, 0.0, 0.0, 1.0]);

    let newdata = DataFrame::new(vec![
        new_herd.into(),
        new_p2.into(),
        new_p3.into(),
        new_p4.into(),
    ])?;

    // lme-rs `predict` currently evaluates the linear predictor (link scale)
    // We apply the inverse-logit function to get probabilities
    let eta = fit.predict(&newdata)?;

    let preds: Vec<f64> = eta.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();

    println!("Predictions:\n{:?}", preds);

    Ok(())
}
