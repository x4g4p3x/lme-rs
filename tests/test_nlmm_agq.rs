//! Scalar and multivariate AGQ (`n_agq > 1`) smoke tests for nlmer.

use lme_rs::nlmer;
use lme_rs::nlmm::{builtin_mean, fit_nlmer, parse_nlmer_formula, NlmerOptions, NlmmStart};
use polars::prelude::*;
use std::fs::File;

#[test]
fn orange_agq_deviance_finite() {
    let mut file = File::open("tests/data/orange.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();
    let mut start = NlmmStart::new();
    start.insert("Asym".into(), 200.0);
    start.insert("xmid".into(), 725.0);
    start.insert("scal".into(), 350.0);
    let laplace = nlmer(
        "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
        &df,
        start.clone(),
        false,
    )
    .unwrap();
    let opts = NlmerOptions {
        start,
        n_agq: 5,
        ..NlmerOptions::default()
    };
    let formula = "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree";
    let (parsed, kind) = parse_nlmer_formula(formula).unwrap();
    let agq = fit_nlmer(&parsed, builtin_mean(kind), &df, formula, &opts).unwrap();
    assert!(laplace.deviance.unwrap().is_finite());
    assert!(agq.deviance.unwrap().is_finite());
}

#[test]
fn orange_multivariate_agq_deviance_finite() {
    let mut file = File::open("tests/data/orange.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();
    let mut start = NlmmStart::new();
    start.insert("Asym".into(), 200.0);
    start.insert("xmid".into(), 725.0);
    start.insert("scal".into(), 350.0);
    let formula = "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree";
    let laplace = nlmer(formula, &df, start.clone(), false).unwrap();
    let opts = NlmerOptions {
        start,
        n_agq: 3,
        ..NlmerOptions::default()
    };
    let (parsed, kind) = parse_nlmer_formula(formula).unwrap();
    let agq = fit_nlmer(&parsed, builtin_mean(kind), &df, formula, &opts).unwrap();
    assert!(laplace.deviance.unwrap().is_finite());
    assert!(agq.deviance.unwrap().is_finite());
    let tl = laplace.theta.as_ref().unwrap();
    let ta = agq.theta.as_ref().unwrap();
    assert_eq!(tl.len(), 3);
    assert_eq!(ta.len(), 3);
    for i in 0..tl.len() {
        assert!(ta[i].is_finite() && tl[i].is_finite());
        assert!(
            (ta[i] - tl[i]).abs() < (5.0_f64).max(10.0 * tl[i].abs()),
            "theta drifted too far: laplace={} agq={}",
            tl[i],
            ta[i]
        );
    }
}
