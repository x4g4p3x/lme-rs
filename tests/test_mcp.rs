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
        2e-5,
    );
    assert_vec_close(
        "z",
        tukey.statistic_values.as_slice().unwrap(),
        &gold.z,
        2e-5,
    );
    assert_vec_close(
        "p_raw",
        tukey.p_value.as_slice().unwrap(),
        &gold.p_raw,
        1e-5,
    );
    assert_vec_close(
        "p_tukey",
        tukey.p_adjust.as_slice().unwrap(),
        &gold.p_tukey,
        1e-5,
    );

    let bonf = fit.glht("cask", McpType::Tukey, McpAdjust::Bonferroni, None)?;
    assert_vec_close(
        "p_bonferroni",
        bonf.p_adjust.as_slice().unwrap(),
        &gold.p_bonferroni,
        1e-5,
    );

    let holm = fit.glht("cask", McpType::Tukey, McpAdjust::Holm, None)?;
    assert_vec_close(
        "p_holm",
        holm.p_adjust.as_slice().unwrap(),
        &gold.p_holm,
        1e-5,
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
        1e-5,
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

fn additive_data() -> DataFrame {
    df!(
        "y" => [1.1, 0.9, 4.2, 3.8, 3.1, 2.9, 6.2, 5.8, 5.1, 4.9, 8.2, 7.8],
        "a" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        "b" => ["low", "low", "high", "high", "low", "low", "high", "high", "low", "low", "high", "high"]
    ).unwrap()
}

#[test]
fn mcp_preserves_additive_comparisons_without_intercept() {
    use lme_rs::{lm_df_with_factors, FactorCoding, FactorSpec};
    let data = additive_data();
    for coding in [FactorCoding::Treatment, FactorCoding::Sum] {
        let factors = std::collections::HashMap::from([
            (
                "a".into(),
                FactorSpec {
                    levels: vec!["C".into(), "A".into(), "B".into()],
                    coding: coding.clone(),
                },
            ),
            (
                "b".into(),
                FactorSpec {
                    levels: vec!["high".into(), "low".into()],
                    coding,
                },
            ),
        ]);
        let reference = lm_df_with_factors("y ~ a + b", &data, &factors).unwrap();
        for formula in ["y ~ 0 + a + b", "y ~ 0 + b + a"] {
            let fit = lm_df_with_factors(formula, &data, &factors).unwrap();
            for term in ["a", "b"] {
                for mcp in [McpType::Tukey, McpType::Dunnett { control: None }] {
                    let expected = reference
                        .glht(
                            term,
                            mcp.clone(),
                            McpAdjust::Holm,
                            Some(DdfMethod::Residual),
                        )
                        .unwrap();
                    let got = fit
                        .glht(term, mcp, McpAdjust::Holm, Some(DdfMethod::Residual))
                        .unwrap();
                    assert_eq!(got.comparisons, expected.comparisons);
                    assert_vec_close(
                        "estimate",
                        got.estimate.as_slice().unwrap(),
                        expected.estimate.as_slice().unwrap(),
                        1e-12,
                    );
                    assert_vec_close(
                        "se",
                        got.std_error.as_slice().unwrap(),
                        expected.std_error.as_slice().unwrap(),
                        1e-12,
                    );
                    assert_vec_close(
                        "adjusted p",
                        got.p_adjust.as_slice().unwrap(),
                        expected.p_adjust.as_slice().unwrap(),
                        1e-12,
                    );
                }
            }
        }
    }
}

#[test]
fn mcp_disambiguates_factor_columns_with_identical_names() {
    let mut data = additive_data();
    data.with_column(Column::new(
        "a".into(),
        [
            "a", "a", "a", "a", "bc", "bc", "bc", "bc", "d", "d", "d", "d",
        ],
    ))
    .unwrap();
    data.rename("b", "ab".into()).unwrap();
    data.with_column(Column::new(
        "ab".into(),
        ["a", "a", "c", "c", "a", "a", "c", "c", "a", "a", "c", "c"],
    ))
    .unwrap();
    let fit = lme_rs::lm_df("y ~ a + ab", &data).unwrap();
    // Both a[bc] and ab[c] are named "abc". Prediction uses their distinct columns.
    let grid = df!("a" => ["a", "bc", "a"], "ab" => ["a", "a", "c"]).unwrap();
    let predictions = fit.predict(&grid).unwrap();
    for (term, index) in [("a", 1), ("ab", 2)] {
        let got = fit
            .glht(
                term,
                McpType::Dunnett { control: None },
                McpAdjust::None,
                Some(DdfMethod::Residual),
            )
            .unwrap();
        assert_close(
            term,
            got.estimate[0],
            predictions[index] - predictions[0],
            1e-12,
        );
        let marginal = fit
            .emmeans_pairs(term, &data, McpAdjust::None, Some(DdfMethod::Residual))
            .unwrap();
        assert_close("se", got.std_error[0], marginal.std_error[0], 1e-12);
        assert_close("p", got.p_value[0], marginal.p_value[0], 1e-12);
    }
}

#[test]
fn mcp_rejects_interaction_only_factor_without_main_effect() {
    let mut data = additive_data();
    data.with_column(Column::new(
        "x".into(),
        [1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.],
    ))
    .unwrap();
    let fit = lme_rs::lm_df("y ~ a:x", &data).unwrap();
    let err = lme_rs::mcp::mcp_contrast_matrix(&fit, "a", &McpType::Tukey).unwrap_err();
    assert!(err.to_string().contains("main-effect coding"));
}
