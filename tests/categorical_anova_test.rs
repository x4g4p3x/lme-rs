use lme_rs::anova::DdfMethod;
use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

/// R / lmerTest reference for `strength ~ cask + (1 | batch)` on pastes
/// (see `comparisons/COMPARISONS.md` and `tests/data/pastes_cask_reml.json`).
const R_CASK_F: f64 = 1.4071;
const R_CASK_P: f64 = 0.2548;
const R_CASK_NUM_DF: f64 = 2.0;
const R_CASK_DEN_DF: f64 = 48.004;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

#[test]
fn test_categorical_anova() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/data/pastes.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;
    fit.with_satterthwaite(&df)?;

    let anova_res = fit.anova(DdfMethod::Satterthwaite)?;

    assert_eq!(anova_res.terms.len(), 1);
    assert_eq!(anova_res.terms[0], "cask");
    assert_close("NumDF", anova_res.num_df[0], R_CASK_NUM_DF, 1e-6);
    assert_close("DenDF", anova_res.den_df[0], R_CASK_DEN_DF, 0.01);
    assert_close("F", anova_res.f_value[0], R_CASK_F, 0.01);
    assert_close("Pr(>F)", anova_res.p_value[0], R_CASK_P, 0.0001);

    Ok(())
}
