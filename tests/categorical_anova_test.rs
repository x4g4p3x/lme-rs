use lme_rs::{lmer, DdfMethod};
use polars::prelude::*;
use std::fs::File;

#[test]
fn test_categorical_anova() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/data/pastes.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;
    fit.with_satterthwaite(&df)?;

    println!("{}", fit);
    let anova_res = fit.anova(DdfMethod::Satterthwaite)?;
    println!("{}", anova_res);

    // Test the specific Multi-DoF Wald Output
    assert_eq!(anova_res.terms.len(), 1); // 1 grouped term (cask)
    assert_eq!(anova_res.terms[0], "cask");
    assert_eq!(anova_res.num_df[0], 2.0); // 3 levels - 1 intercept = 2 DoF

    Ok(())
}
