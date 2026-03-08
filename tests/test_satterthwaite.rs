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

#[test]
fn test_satterthwaite_anova_numerical_parity() {
    use lme_rs::anova::DdfMethod;
    use serde_json::Value;

    // 1. Load the generated reference JSON output
    let file = File::open("tests/data/random_slopes.json").unwrap();
    let r_data: Value = serde_json::from_reader(file).unwrap();

    let r_anova_f = r_data["outputs"]["sat_anova_f"]
        .as_array()
        .expect("No Satterthwaite anova F found in JSON");
    let r_anova_p = r_data["outputs"]["sat_anova_p"]
        .as_array()
        .expect("No Satterthwaite anova P found in JSON");
    let r_anova_ndf = r_data["outputs"]["sat_anova_ndf"]
        .as_array()
        .expect("No Satterthwaite anova NumDF found in JSON");
    let r_anova_ddf = r_data["outputs"]["sat_anova_ddf"]
        .as_array()
        .expect("No Satterthwaite anova DenDF found in JSON");

    // 2. Load the actual dataset and fit the model
    let df = load_sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Evaluate internal stats
    fit.with_satterthwaite(&df).unwrap();

    // Generate analytical F-tests mapped over Satterthwaite DoFs
    let anova_res = fit
        .anova(DdfMethod::Satterthwaite)
        .expect("Failed to build Satterthwaite ANOVA table");

    // 3. Compare values (Note: R's ANOVA excludes Intercept, so it only has 1 row for 'Days')
    assert_eq!(anova_res.terms.len(), 1);
    assert_eq!(anova_res.terms[0], "Days");

    let tol = 1e-2; // Allow reasonable precision
    // R `anova()` output lists the 'Days' effect as index 0 since Intercept is dropped.
    let rust_f = anova_res.f_value[0];
    let r_f = r_anova_f[0].as_f64().unwrap();
    assert!(
        (rust_f - r_f).abs() < tol,
        "Satt ANOVA F mismatch on effect Days: Rust={}, R={}",
        rust_f,
        r_f
    );

    let rust_p = anova_res.p_value[0];
    let r_p = r_anova_p[0].as_f64().unwrap();
    assert!(
        (rust_p - r_p).abs() < tol,
        "Satt ANOVA P mismatch on effect Days: Rust={}, R={}",
        rust_p,
        r_p
    );

    let rust_ndf = anova_res.num_df[0];
    let r_ndf = r_anova_ndf[0].as_f64().unwrap();
    assert!(
        (rust_ndf - r_ndf).abs() < tol,
        "Satt ANOVA NumDF mismatch on effect Days: Rust={}, R={}",
        rust_ndf,
        r_ndf
    );

    let rust_ddf = anova_res.den_df[0];
    let r_ddf = r_anova_ddf[0].as_f64().unwrap();
    assert!(
        (rust_ddf - r_ddf).abs() < tol,
        "Satt ANOVA DenDF mismatch on effect Days: Rust={}, R={}",
        rust_ddf,
        r_ddf
    );
}
