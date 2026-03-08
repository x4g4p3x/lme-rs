use lme_rs::lmer;
use polars::prelude::*;

#[test]
fn test_predict_population_level() -> anyhow::Result<()> {
    // 1. Load the original dataset
    let mut file =
        std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
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

#[test]
fn test_predict_conditional_unseen_levels() -> anyhow::Result<()> {
    // 1. Load the original dataset
    let mut file =
        std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // 2. Fit the model matching lme4's random slopes configuration
    let formula = "Reaction ~ Days + (Days | Subject)";
    let fit = lmer(formula, &df, true)?;

    // 3. Construct a dataframe with a Subject ID not present in the original dataset ("999")
    let new_days = Series::new("Days".into(), &[0.0]);
    let new_subject = Series::new("Subject".into(), &["999"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()])?;

    // 4. Test missing level trapping with `allow_new_levels = false`
    let cond_reject = fit.predict_conditional(&newdata, false);
    assert!(
        cond_reject.is_err(),
        "Expected error for unknown level when allow_new_levels is false"
    );
    let err_msg = cond_reject.unwrap_err().to_string();
    assert!(err_msg.contains("New level '999' found in grouping factor 'Subject'"));
    assert!(err_msg.contains("but allow_new_levels is false"));

    // 5. Test fallback execution returning Ok() handling missing levels matching `allow_new_levels = true`
    let cond_allow = fit.predict_conditional(&newdata, true);
    assert!(
        cond_allow.is_ok(),
        "Expected success for unknown level when allow_new_levels is true"
    );

    // The conditional output should exactly match population output since the RE contribution is 0
    let pop_allow = fit.predict(&newdata).unwrap();
    let cond_val = cond_allow.unwrap();

    // (Intercept) beta0
    assert!((cond_val[0] - pop_allow[0]).abs() < 1e-10);

    Ok(())
}
