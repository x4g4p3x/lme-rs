use lme_rs::{lmer, anova};
use polars::prelude::*;
use std::fs::File;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

#[test]
fn test_anova_nested_models() {
    let df = load_sleepstudy();

    // Simpler model: intercept-only random effect
    let fit0 = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    // More complex model: random slopes + intercept
    let fit1 = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let lrt = anova(&fit0, &fit1).unwrap();

    println!("{}", lrt);

    // Basic sanity checks:
    // More complex model should have more parameters
    assert!(lrt.n_params_1 > lrt.n_params_0, "Complex model should have more params");
    // Degrees of freedom should be the difference
    assert_eq!(lrt.df, lrt.n_params_1 - lrt.n_params_0);
    // Chi-sq should be non-negative
    assert!(lrt.chi_sq >= 0.0, "Chi-sq should be non-negative");
    // P-value should be in [0, 1]
    assert!(lrt.p_value >= 0.0 && lrt.p_value <= 1.0, "P-value should be in [0,1], got {}", lrt.p_value);
    // For sleepstudy, random slopes significantly improve fit
    assert!(lrt.p_value < 0.05, "Random slopes should significantly improve fit (p < 0.05), got {}", lrt.p_value);
}

#[test]
fn test_anova_reversed_argument_order() {
    let df = load_sleepstudy();

    let fit0 = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let fit1 = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // anova should auto-detect which model is simpler regardless of argument order
    let lrt_ab = anova(&fit0, &fit1).unwrap();
    let lrt_ba = anova(&fit1, &fit0).unwrap();

    assert!((lrt_ab.chi_sq - lrt_ba.chi_sq).abs() < 1e-10, "Chi-sq should be identical regardless of argument order");
    assert!((lrt_ab.p_value - lrt_ba.p_value).abs() < 1e-10, "P-value should be identical regardless of argument order");
    assert_eq!(lrt_ab.df, lrt_ba.df, "df should be identical regardless of argument order");
}

#[test]
fn test_anova_display_format() {
    let df = load_sleepstudy();
    let fit0 = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();
    let fit1 = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let lrt = anova(&fit0, &fit1).unwrap();
    let output = format!("{}", lrt);

    assert!(output.contains("Models:"), "Display should show model section");
    assert!(output.contains("Chisq"), "Display should show chi-sq header");
    assert!(output.contains("Pr(>Chisq)"), "Display should show p-value header");
    assert!(output.contains("npar"), "Display should show npar header");
}

#[test]
fn test_anova_different_data_sizes_error() {
    let df = load_sleepstudy();

    let fit0 = lmer("Reaction ~ Days + (1 | Subject)", &df, true).unwrap();

    // Create a modified fit with different num_obs
    let mut fit_fake = fit0.clone();
    fit_fake.num_obs = 999;

    let result = anova(&fit0, &fit_fake);
    assert!(result.is_err(), "anova should fail with different data sizes");
    assert!(result.unwrap_err().to_string().contains("same data"),
        "Error should mention same data");
}
