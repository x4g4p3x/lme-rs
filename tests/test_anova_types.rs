//! Type II / III ANOVA integration tests.

use lme_rs::anova::{AnovaType, DdfMethod};
use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

#[test]
fn type3_factor_main_effect_averages_over_interaction_levels() {
    let df = df!(
        "y" => [0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 10.0, 12.0],
        "a" => ["A", "A", "A", "A", "B", "B", "B", "B"],
        "b" => ["C", "C", "D", "D", "C", "C", "D", "D"]
    )
    .unwrap();
    // Cell means are 1, 3, 5, 11, with SSE=8 and MSE=2. Uniform marginal
    // differences are 6 and 4, each with variance 1. The interaction has F=4.
    for coding in [lme_rs::FactorCoding::Treatment, lme_rs::FactorCoding::Sum] {
        for reverse in [false, true] {
            let levels = |a: &str, b: &str| {
                if reverse {
                    vec![b.into(), a.into()]
                } else {
                    vec![a.into(), b.into()]
                }
            };
            let factors = std::collections::HashMap::from([
                (
                    "a".into(),
                    lme_rs::FactorSpec {
                        levels: levels("A", "B"),
                        coding: coding.clone(),
                    },
                ),
                (
                    "b".into(),
                    lme_rs::FactorSpec {
                        levels: levels("C", "D"),
                        coding: coding.clone(),
                    },
                ),
            ]);
            let fit = lme_rs::lm_df_with_factors("y ~ a * b", &df, &factors).unwrap();
            let table = fit
                .anova_typed(AnovaType::Type3, DdfMethod::Residual)
                .unwrap();
            assert!((table.f_value[0] - 36.0).abs() < 1e-8, "{table:?}");
            assert!((table.f_value[1] - 16.0).abs() < 1e-8, "{table:?}");
            assert!((table.f_value[2] - 4.0).abs() < 1e-8, "{table:?}");
        }
    }
}

fn toy_interaction_df() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("y".into(), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).into(),
        Series::new("trt".into(), vec!["A", "A", "B", "B", "A", "A", "B", "B"]).into(),
        Series::new(
            "blk".into(),
            vec!["b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2"],
        )
        .into(),
    ])
    .unwrap()
}

#[test]
fn interaction_design_includes_main_and_interaction_terms() {
    let df = toy_interaction_df();
    let fit = lmer("y ~ trt * blk + (1 | blk)", &df, true).unwrap();
    let names = fit.fixed_names.unwrap();
    assert!(
        names.len() > 3,
        "expected intercept, mains, and interaction dummies"
    );
    assert!(fit
        .fixed_term_assign
        .as_ref()
        .unwrap()
        .contains(&"trt:blk".to_string()));
}

#[test]
fn type2_matches_type3_on_additive_sleepstudy() {
    let mut file = File::open("tests/data/sleepstudy.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    fit.with_satterthwaite(&df).unwrap();
    let t3 = fit
        .anova_typed(AnovaType::Type3, DdfMethod::Satterthwaite)
        .unwrap();
    let t2 = fit
        .anova_typed(AnovaType::Type2, DdfMethod::Satterthwaite)
        .unwrap();
    assert_eq!(t3.terms, t2.terms);
    assert!((t3.f_value[0] - t2.f_value[0]).abs() < 1e-6);
    assert!((t3.den_df[0] - t2.den_df[0]).abs() < 0.05);
}

#[test]
fn type1_anova_runs_on_sleepstudy() {
    let mut file = File::open("tests/data/sleepstudy.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    fit.with_satterthwaite(&df).unwrap();
    let t1 = fit
        .anova_typed(AnovaType::Type1, DdfMethod::Satterthwaite)
        .unwrap();
    assert_eq!(t1.terms, vec!["Days"]);
    assert!(t1.f_value[0].is_finite());
    assert!(t1.p_value[0].is_finite());
}
