//! Tukey / Dunnett MCP (`glht`-style) vs lme4 Wald + `stats::ptukey` goldens.

use lme_rs::anova::DdfMethod;
use lme_rs::{lmer, McpAdjust, McpType};
use polars::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Deserialize)]
struct PastesGlhtFixture {
    comparisons: Vec<String>,
    estimate: Vec<f64>,
    std_error: Vec<f64>,
    z: Vec<f64>,
    p_raw: Vec<f64>,
    p_bonferroni: Vec<f64>,
    p_holm: Vec<f64>,
    p_tukey: Vec<f64>,
    dunnett_comparisons: Vec<String>,
    dunnett_p_bonferroni: Vec<f64>,
}

fn load_pastes() -> DataFrame {
    let file = File::open("tests/data/pastes.csv").expect("pastes.csv");
    CsvReader::new(file).finish().expect("read pastes")
}

fn load_fixture() -> PastesGlhtFixture {
    let mut buf = String::new();
    File::open("tests/data/pastes_glht_tukey.json")
        .expect("pastes_glht_tukey.json")
        .read_to_string(&mut buf)
        .unwrap();
    serde_json::from_str(&buf).expect("parse fixture")
}

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

fn assert_vec_close(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "{label} length");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close(&format!("{label}[{i}]"), *g, *e, tol);
    }
}

#[test]
fn test_pastes_cask_tukey_wald_matches_lme4_ptukey() -> Result<(), Box<dyn std::error::Error>> {
    let gold = load_fixture();
    let df = load_pastes();
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;

    let tukey = fit.glht("cask", McpType::Tukey, McpAdjust::Tukey, None)?;
    assert_eq!(tukey.statistic, "z");
    assert_eq!(tukey.comparisons, gold.comparisons);
    assert_vec_close(
        "estimate",
        tukey.estimate.as_slice().unwrap(),
        &gold.estimate,
        1e-8,
    );
    assert_vec_close(
        "se",
        tukey.std_error.as_slice().unwrap(),
        &gold.std_error,
        1e-8,
    );
    assert_vec_close(
        "z",
        tukey.statistic_values.as_slice().unwrap(),
        &gold.z,
        1e-8,
    );
    assert_vec_close(
        "p_raw",
        tukey.p_value.as_slice().unwrap(),
        &gold.p_raw,
        1e-10,
    );
    assert_vec_close(
        "p_tukey",
        tukey.p_adjust.as_slice().unwrap(),
        &gold.p_tukey,
        1e-8,
    );

    let bonf = fit.glht("cask", McpType::Tukey, McpAdjust::Bonferroni, None)?;
    assert_vec_close(
        "p_bonferroni",
        bonf.p_adjust.as_slice().unwrap(),
        &gold.p_bonferroni,
        1e-12,
    );

    let holm = fit.glht("cask", McpType::Tukey, McpAdjust::Holm, None)?;
    assert_vec_close(
        "p_holm",
        holm.p_adjust.as_slice().unwrap(),
        &gold.p_holm,
        1e-12,
    );

    let dunnett = fit.glht(
        "cask",
        McpType::Dunnett { control: None },
        McpAdjust::Bonferroni,
        None,
    )?;
    assert_eq!(dunnett.comparisons, gold.dunnett_comparisons);
    assert_vec_close(
        "dunnett_p_bonferroni",
        dunnett.p_adjust.as_slice().unwrap(),
        &gold.dunnett_p_bonferroni,
        1e-12,
    );

    Ok(())
}

#[test]
fn test_pastes_tukey_satterthwaite_matches_unit_contrast() -> Result<(), Box<dyn std::error::Error>>
{
    let df = load_pastes();
    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;
    fit.with_satterthwaite(&df)?;

    let glht = fit.glht(
        "cask",
        McpType::Tukey,
        McpAdjust::None,
        Some(DdfMethod::Satterthwaite),
    )?;
    assert_eq!(glht.statistic, "t");
    // `b - a` is the `caskb` dummy (unit contrast).
    let names = fit.fixed_names.as_ref().unwrap();
    let b_idx = names.iter().position(|n| n == "caskb").unwrap();
    let sat = fit.satterthwaite.as_ref().unwrap();
    assert_close(
        "b-a t",
        glht.statistic_values[0],
        fit.beta_t.as_ref().unwrap()[b_idx],
        1e-10,
    );
    assert_close("b-a df", glht.den_df[0], sat.dfs[b_idx], 1e-8);
    assert_close("b-a p", glht.p_value[0], sat.p_values[b_idx], 1e-10);

    Ok(())
}

#[test]
fn test_dunnett_rejects_tukey_adjust() {
    let df = load_pastes();
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true).unwrap();
    let err = fit
        .glht(
            "cask",
            McpType::Dunnett {
                control: Some("a".to_string()),
            },
            McpAdjust::Tukey,
            None,
        )
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Dunnett"), "{msg}");
}

#[test]
fn test_glht_unknown_term() {
    let df = load_pastes();
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true).unwrap();
    let err = fit
        .glht("batch", McpType::Tukey, McpAdjust::None, None)
        .unwrap_err();
    assert!(err.to_string().contains("batch"));
}
