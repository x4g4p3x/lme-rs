use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;
use serde_json::Value;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

#[test]
fn test_kenward_roger_runs_and_formats() {
    let df = load_sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    assert!(fit.kenward_roger.is_none());

    // Compute Kenward-Roger df & p-values
    fit.with_kenward_roger(&df).unwrap();
    
    assert!(fit.kenward_roger.is_some());
    let kr = fit.kenward_roger.as_ref().unwrap();
    
    // We expect 2 fixed effects (Intercept and Days)
    assert_eq!(kr.dfs.len(), 2);
    assert_eq!(kr.p_values.len(), 2);
    
    // Verify Display formatting includes Kenward-Roger
    let output = format!("{}", fit);
    assert!(output.contains("df"));
    assert!(output.contains("Pr(>|t|)"));
    assert!(output.contains("[Kenward-Roger]"));
}

#[test]
fn test_kenward_roger_numerical_parity() {
    // 1. Load the generated reference JSON output
    let file = File::open("tests/data/random_slopes.json").unwrap();
    let r_data: Value = serde_json::from_reader(file).unwrap();

    let r_dfs = r_data["outputs"]["kr_dof"].as_array().expect("No KR dof found in JSON");
    let r_pvs = r_data["outputs"]["kr_p"].as_array().expect("No KR p-values found in JSON");

    // 2. Load the actual dataset and fit the model
    let df = load_sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Compute Kenward-Roger logic
    fit.with_satterthwaite(&df).unwrap();
    let satt = fit.satterthwaite.clone().unwrap();
    println!("Satterthwaite DoFs: {:?}", satt.dfs);
    
    fit.with_kenward_roger(&df).unwrap();
    let kr = fit.kenward_roger.as_ref().unwrap();
    println!("Kenward-Roger DoFs: {:?}", kr.dfs);

    // 3. Compare values
    let tol = 1e-2; // Expect reasonable numeric parity
    for i in 0..2 {
        let rust_df = kr.dfs[i];
        let r_df = r_dfs[i].as_f64().unwrap();
        assert!((rust_df - r_df).abs() < tol, "KR DoF mismatch on effect {}: Rust={}, R={}", i, rust_df, r_df);

        let rust_pv = kr.p_values[i];
        let r_pv = r_pvs[i].as_f64().unwrap();
        assert!((rust_pv - r_pv).abs() < tol, "KR p-value mismatch on effect {}: Rust={}, R={}", i, rust_pv, r_pv);
    }
}
