//! Type II / III ANOVA integration tests.

use lme_rs::anova::{AnovaType, DdfMethod};
use lme_rs::lmer;
use polars::prelude::*;
use std::fs::File;

fn toy_interaction_df() -> DataFrame {
    DataFrame::new(vec![
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
