//! Synthetic `SSpower` (`a * x^b + c`) nlmer tests for grouped calibration curves.

use lme_rs::nlmer;
use lme_rs::nlmm::{sspower_eval, NlmmStart};
use polars::prelude::*;
use std::fs::File;

fn power_df() -> DataFrame {
    let a = 2.0;
    let b = 0.5;
    let c = 1.0;
    let n = 50usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = 0.5 + (i as f64) * 0.2;
        let gi = if i < n / 2 { "A" } else { "B" };
        let offset = if gi == "A" { 0.15 } else { -0.1 };
        let (mu, _, _, _) = sspower_eval(a, b, c + offset, xi);
        x.push(xi);
        y.push(mu + 0.05 * (i as f64 - 25.0));
        g.push(gi.to_string());
    }
    DataFrame::new(vec![
        Column::new("y".into(), &y),
        Column::new("x".into(), &x),
        Column::new("g".into(), &g),
    ])
    .unwrap()
}

#[test]
fn sspower_nlmer_with_explicit_start() {
    let df = power_df();
    let mut start = NlmmStart::new();
    start.insert("a".into(), 1.5);
    start.insert("b".into(), 0.4);
    start.insert("c".into(), 0.5);
    let fit = nlmer("y ~ SSpower(x, a, b, c) ~ c|g", &df, start, false).unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 3);
}

#[test]
fn sspower_nlmer_self_start_runs() {
    let df = power_df();
    let fit = nlmer(
        "y ~ SSpower(x, a, b, c) ~ c|g",
        &df,
        NlmmStart::new(),
        false,
    )
    .unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 3);
    let pred = fit.predict(&df).unwrap();
    assert_eq!(pred.len(), df.height());
}

#[test]
fn sspower_nlmer_self_start_matches_lme4_fixture() {
    let mut file = File::open("tests/data/sspower_synthetic.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();
    let fit = nlmer(
        "y ~ SSpower(x, a, b, c) ~ c|id",
        &df,
        NlmmStart::new(),
        false,
    )
    .unwrap();

    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // Reference: tests/data/sspower_nlmer.json (lme4::nlmer + R custom selfStart SSpower)
    assert!((coef[idx("a")] - 2.2673043880442).abs() < 0.5, "a = {}", coef[idx("a")]);
    assert!((coef[idx("b")] - 0.455475338357398).abs() < 0.25, "b = {}", coef[idx("b")]);
    assert!((coef[idx("c")] - 0.754933725402219).abs() < 0.5, "c = {}", coef[idx("c")]);
    let tau = fit.theta.as_ref().unwrap()[0];
    assert!((tau - 0.945985754783105).abs() < 0.5, "theta = {tau}");
}
