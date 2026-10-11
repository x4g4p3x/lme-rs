//! Kenward–Roger multi-DoF ANOVA (`KRmodcomp`) on pastes / `cask`.
//!
//! Reference: R `lmerTest::anova(fit, ddf = "Kenward-Roger", type = 3)` and
//! `pbkrtest::KRmodcomp` with the Type III contrast for `cask`.

use lme_rs::anova::{AnovaType, DdfMethod};
use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

const R_CASK_F: f64 = 1.4069;
const R_CASK_P: f64 = 0.2548;
const R_CASK_NUM_DF: f64 = 2.0;
const R_CASK_DEN_DF: f64 = 48.0;

#[test]
fn balanced_block_by_harvest_kr_matches_paired_block_test() {
    let mut y = Vec::new();
    let mut block = Vec::new();
    let mut sow = Vec::new();
    let mut harvest = Vec::new();
    for b in 0..3 {
        for s in 0..5 {
            for h in 0..2 {
                let bd = b as f64 - 1.0;
                let noise = [0.0, 1.0, -1.0, 1.0, -1.0][s];
                y.push(
                    2.0 + 0.3 * b as f64 + 0.2 * s as f64 - 0.1 * h as f64
                        + 0.05 * bd * (2.0 * h as f64 - 1.0)
                        + 0.02 * bd * noise,
                );
                block.push(format!("B{b}"));
                sow.push(format!("S{s}"));
                harvest.push(format!("H{h}"));
            }
        }
    }
    let df = df!("y" => y, "block" => block, "sow" => sow, "harvest" => harvest).unwrap();
    let mut fit = lmer("y ~ block + sow + harvest + (1 | block:harvest)", &df, true).unwrap();
    fit.with_kenward_roger(&df).unwrap();
    let result = fit
        .linear_hypothesis("harvest", DdfMethod::KenwardRoger)
        .unwrap();
    // Blockwise harvest differences are -0.2, -0.1 and 0. Their mean is -0.1
    // and sample variance is 0.01, so the balanced paired-block t^2 is 3,
    // with G-1 = 2 degrees of freedom. This does not use another fit as oracle.
    assert_close("DenDF", result.den_df, 2.0, 1e-6);
    assert_close("F", result.f_value, 3.0, 1e-4);
}

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

#[test]
fn test_pastes_cask_kr_modcomp_anova() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/data/pastes.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;
    fit.with_kenward_roger(&df)?;

    let anova_res = fit.anova_typed(AnovaType::Type3, DdfMethod::KenwardRoger)?;

    assert_eq!(anova_res.terms.len(), 1);
    assert_eq!(anova_res.terms[0], "cask");
    assert_close("NumDF", anova_res.num_df[0], R_CASK_NUM_DF, 1e-6);
    assert_close("DenDF", anova_res.den_df[0], R_CASK_DEN_DF, 0.05);
    assert_close("F", anova_res.f_value[0], R_CASK_F, 0.02);
    assert_close("Pr(>F)", anova_res.p_value[0], R_CASK_P, 0.001);

    // KR DenDF should match marginal pooling when vcovAdj ≈ vcov (pastes batch RE).
    fit.with_satterthwaite(&df)?;
    let sat = fit.anova_typed(AnovaType::Type3, DdfMethod::Satterthwaite)?;
    assert!((sat.den_df[0] - R_CASK_DEN_DF).abs() < 0.1);
    assert!((anova_res.den_df[0] - sat.den_df[0]).abs() < 0.1);

    Ok(())
}
