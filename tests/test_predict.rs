use polars::prelude::*;
use lme_rs::lmer;

#[test]
fn test_predict_population_level() -> anyhow::Result<()> {
    // 1. Load the original dataset
    let mut file = std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // 2. Fit the model matching lme4's random slopes configuration
    let formula = "Reaction ~ Days + (Days | Subject)";
    let fit = lmer(formula, &df, true)?;
    
    // We establish our truth bounds against what we know lme4 generates for sleepstudy fixed effects:
    // (Intercept)    Days 
    //   251.4051   10.4673 

    // 3. Construct a specific new DataFrame to evaluate predictions
    let new_days = Series::new("Days".into(), &[0.0, 1.0, 5.0, 10.0]);
    // The subject column shouldn't technically matter for re.form=NA, but the dataframe might need it
    let new_subject = Series::new("Subject".into(), &["308", "308", "308", "308"]);
    
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()])?;

    // 4. Trigger predicting
    let preds = fit.predict(&newdata)?;
    println!("y_pred: {:?}", preds.to_vec());

    // 5. Verify mathematical precision
    // y_pred = beta0 + beta1 * Days
    let b0 = fit.coefficients[0];
    let b1 = fit.coefficients[1];

    assert!((preds[0] - (b0 + b1 * 0.0)).abs() < 1e-6); // Approx 251.4
    assert!((preds[1] - (b0 + b1 * 1.0)).abs() < 1e-6); // Approx 261.8
    assert!((preds[2] - (b0 + b1 * 5.0)).abs() < 1e-6); // Approx 303.7
    assert!((preds[3] - (b0 + b1 * 10.0)).abs() < 1e-6); // Approx 356.0

    Ok(())
}
