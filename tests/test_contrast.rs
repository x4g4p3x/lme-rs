//! User-defined contrast matrices vs ANOVA / `lmerTest::contestMD` reference values.

use lme_rs::anova::{AnovaType, DdfMethod};
use lme_rs::contrast::{contrast_matrix, contrast_matrix_from_names, ContrastRowSpec};
use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

const R_CASK_F: f64 = 1.4071;
const R_CASK_P: f64 = 0.2548;
const R_CASK_DEN_DF: f64 = 48.004;

#[test]
fn named_contrasts_reject_ambiguous_coefficient_labels() {
    let data = df!(
        "y" => [1., 2.1, 3., 4.2, 5., 6.3, 7., 8.1],
        "a" => ["a", "a", "a", "a", "bc", "bc", "bc", "bc"],
        "ab" => ["a", "a", "c", "c", "a", "a", "c", "c"]
    )
    .unwrap();
    let fit = lme_rs::lm_df("y ~ a + ab", &data).unwrap();
    let names = fit.fixed_names.as_ref().unwrap();
    assert_eq!(names.iter().filter(|n| n.as_str() == "abc").count(), 2);
    let row = ContrastRowSpec {
        label: "ambiguous",
        weights: &[("abc", 1.)],
    };
    assert!(contrast_matrix_from_names(names, &[row]).is_err());
    // An unambiguous coefficient remains usable despite unrelated collisions.
    let row = ContrastRowSpec {
        label: "intercept",
        weights: &[("(Intercept)", 1.)],
    };
    let l = contrast_matrix_from_names(names, &[row]).unwrap();
    assert_eq!(l, ndarray::array![[1., 0., 0.]]);
    // Explicit indices and ANOVA term labels disambiguate the two effects.
    for (term, index) in [("a", 1), ("ab", 2)] {
        let named = fit.linear_hypothesis(term, DdfMethod::Residual).unwrap();
        let explicit = fit
            .test_contrast(
                &contrast_matrix(3, &[vec![(index, 1.)]]),
                DdfMethod::Residual,
            )
            .unwrap();
        assert!((named.f_value - explicit.f_value).abs() < 1e-12);
    }
}

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

#[test]
fn test_pastes_cask_contrast_matches_anova() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/data/pastes.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;
    fit.with_satterthwaite(&df)?;

    let names = fit.fixed_names.clone().unwrap();
    let l = contrast_matrix_from_names(
        &names,
        &[
            ContrastRowSpec {
                label: "caskb",
                weights: &[("caskb", 1.0)],
            },
            ContrastRowSpec {
                label: "caskc",
                weights: &[("caskc", 1.0)],
            },
        ],
    )?;

    let res = fit.test_contrast(&l, DdfMethod::Satterthwaite)?;
    assert_close("F", res.f_value, R_CASK_F, 0.02);
    assert_close("p", res.p_value, R_CASK_P, 0.001);
    assert_close("DenDF", res.den_df, R_CASK_DEN_DF, 0.05);
    assert!((res.num_df - 2.0).abs() < 1e-6);

    let anova = fit.anova_typed(AnovaType::Type3, DdfMethod::Satterthwaite)?;
    assert_eq!(anova.terms.len(), 1);
    assert_close("anova F", anova.f_value[0], res.f_value, 1e-10);
    assert_close("anova p", anova.p_value[0], res.p_value, 1e-10);
    assert_close("anova DenDF", anova.den_df[0], res.den_df, 1e-10);

    let lh = fit.linear_hypothesis("cask", DdfMethod::Satterthwaite)?;
    assert_close("linear_hypothesis F", lh.f_value, res.f_value, 1e-10);
    assert_close("linear_hypothesis p", lh.p_value, res.p_value, 1e-10);

    Ok(())
}

#[test]
fn test_sleepstudy_days_contrast_matches_marginal() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/data/sleepstudy.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
    fit.with_satterthwaite(&df)?;

    // Days coefficient is column 1 (Intercept, Days).
    let l = contrast_matrix(2, &[vec![(1, 1.0)]]);
    let res = fit.test_contrast(&l, DdfMethod::Satterthwaite)?;

    let t = fit.beta_t.as_ref().unwrap()[1];
    assert_close("F", res.f_value, t * t, 1e-8);
    assert!((res.num_df - 1.0).abs() < 1e-10);
    assert!((res.den_df - fit.satterthwaite.as_ref().unwrap().dfs[1]).abs() < 0.01);

    Ok(())
}
