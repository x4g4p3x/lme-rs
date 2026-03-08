use lme_rs::lmer;
use polars::prelude::*;
use serde_json::Value;
use std::fs::File;

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

    let r_dfs = r_data["outputs"]["kr_dof"]
        .as_array()
        .expect("No KR dof found in JSON");
    let r_pvs = r_data["outputs"]["kr_p"]
        .as_array()
        .expect("No KR p-values found in JSON");

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
        assert!(
            (rust_df - r_df).abs() < tol,
            "KR DoF mismatch on effect {}: Rust={}, R={}",
            i,
            rust_df,
            r_df
        );

        let rust_pv = kr.p_values[i];
        let r_pv = r_pvs[i].as_f64().unwrap();
        assert!(
            (rust_pv - r_pv).abs() < tol,
            "KR p-value mismatch on effect {}: Rust={}, R={}",
            i,
            rust_pv,
            r_pv
        );
    }
}

#[test]
fn test_kenward_roger_anova_numerical_parity() {
    use lme_rs::anova::DdfMethod;

    // 1. Load the generated reference JSON output
    let file = File::open("tests/data/random_slopes.json").unwrap();
    let r_data: Value = serde_json::from_reader(file).unwrap();

    let r_anova_f = r_data["outputs"]["kr_anova_f"]
        .as_array()
        .expect("No KR anova F found in JSON");
    let r_anova_p = r_data["outputs"]["kr_anova_p"]
        .as_array()
        .expect("No KR anova P found in JSON");
    let r_anova_ndf = r_data["outputs"]["kr_anova_ndf"]
        .as_array()
        .expect("No KR anova NumDF found in JSON");
    let r_anova_ddf = r_data["outputs"]["kr_anova_ddf"]
        .as_array()
        .expect("No KR anova DenDF found in JSON");

    // 2. Load the actual dataset and fit the model
    let df = load_sleepstudy();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Evaluate internal stats
    fit.with_kenward_roger(&df).unwrap();

    // Generate analytical F-tests mapped over KR DoFs
    let anova_res = fit
        .anova(DdfMethod::KenwardRoger)
        .expect("Failed to build KR ANOVA table");

    // 3. Compare values (Note: R's ANOVA excludes Intercept, so it only has 1 row for 'Days')
    assert_eq!(anova_res.terms.len(), 1);
    assert_eq!(anova_res.terms[0], "Days");

    let tol = 1e-2;
    // R `anova()` output lists the 'Days' effect as index 0 since Intercept is dropped.
    let rust_f = anova_res.f_value[0];
    let r_f = r_anova_f[0].as_f64().unwrap();
    assert!(
        (rust_f - r_f).abs() < tol,
        "KR ANOVA F mismatch on effect Days: Rust={}, R={}",
        rust_f,
        r_f
    );

    let rust_p = anova_res.p_value[0];
    let r_p = r_anova_p[0].as_f64().unwrap();
    assert!(
        (rust_p - r_p).abs() < tol,
        "KR ANOVA P mismatch on effect Days: Rust={}, R={}",
        rust_p,
        r_p
    );

    let rust_ndf = anova_res.num_df[0];
    let r_ndf = r_anova_ndf[0].as_f64().unwrap();
    assert!(
        (rust_ndf - r_ndf).abs() < tol,
        "KR ANOVA NumDF mismatch on effect Days: Rust={}, R={}",
        rust_ndf,
        r_ndf
    );

    let rust_ddf = anova_res.den_df[0];
    let r_ddf = r_anova_ddf[0].as_f64().unwrap();
    assert!(
        (rust_ddf - r_ddf).abs() < tol,
        "KR ANOVA DenDF mismatch on effect Days: Rust={}, R={}",
        rust_ddf,
        r_ddf
    );
}
