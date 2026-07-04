//! Scalar AGQ (`n_agq > 1`) smoke test for nlmer.

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
