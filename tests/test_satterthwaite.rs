use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

#[test]
fn test_satterthwaite_runs_and_formats() {
    let df = load_sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Fit should not initially have Satterthwaite results
    assert!(fit.satterthwaite.is_none());

    // Compute Satterthwaite df & p-values
    fit.with_satterthwaite(&df).unwrap();
    
    // Results should now be populated
    assert!(fit.satterthwaite.is_some());
    let satt = fit.satterthwaite.as_ref().unwrap();
    
    // We expect 2 fixed effects (Intercept and Days)
    assert_eq!(satt.dfs.len(), 2);
    assert_eq!(satt.p_values.len(), 2);
    
    // The df for sleepstudy (18 subjects, 180 obs) is typically:
    // Intercept: ~17.0
    // Days: ~17.0
    // (We allow some numeric variance here depending on optimizers)
    assert!(satt.dfs[0] > 10.0 && satt.dfs[0] < 30.0);
    assert!(satt.dfs[1] > 10.0 && satt.dfs[1] < 30.0);
    
    // Both effects are highly significant in sleepstudy
    assert!(satt.p_values[0] < 0.001);
    assert!(satt.p_values[1] < 0.001);

    // Verify Display formatting
    let output = format!("{}", fit);
    assert!(output.contains("df"));
    assert!(output.contains("Pr(>|t|)"));
}
