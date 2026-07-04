//! `nlmer` with random effects on multiple nonlinear parameters (Orange / SSlogis).

use lme_rs::nlmm::{nlmer, NlmmStart};
use polars::prelude::*;
use std::fs::File;

const TOL_COEF: f64 = 12.0;
const TOL_THETA: f64 = 1.0;
const TOL_LL: f64 = 18.0;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

fn orange_multi_re_fit() -> lme_rs::LmeFit {
    let mut file = File::open("tests/data/orange.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let mut start = NlmmStart::new();
    start.insert("Asym".to_string(), 200.0);
    start.insert("xmid".to_string(), 725.0);
    start.insert("scal".to_string(), 350.0);

    nlmer(
        "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree",
        &df,
        start,
        false,
    )
    .unwrap()
}

#[test]
fn orange_multi_re_formula_and_structure() {
    let fit = orange_multi_re_fit();
    assert_eq!(fit.fixed_names.as_ref().unwrap().len(), 3);
    assert_eq!(fit.theta.as_ref().unwrap().len(), 3);
    assert!(fit.b.as_ref().unwrap().len() >= 5);
    assert!(fit.converged.unwrap_or(false));
}

#[test]
fn orange_multi_re_conditional_differs_from_population() {
    let fit = orange_multi_re_fit();
    let mut file = File::open("tests/data/orange.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let pop = fit.predict(&df).unwrap();
    let cond = fit.predict_conditional(&df, false).unwrap();
    let mut max_diff = 0.0_f64;
    for (p, c) in pop.iter().zip(cond.iter()) {
        max_diff = max_diff.max((p - c).abs());
    }
    assert!(
        max_diff > 0.5,
        "conditional predictions should differ from population (max diff={max_diff})"
    );
}

#[test]
fn orange_multi_re_matches_lme4_reference() {
    let fit = orange_multi_re_fit();
    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // lme4 1.1-38 on Orange
    assert_close("Asym", coef[idx("Asym")], 191.3665, TOL_COEF);
    assert_close("xmid", coef[idx("xmid")], 717.5343, TOL_COEF);
    assert_close("scal", coef[idx("scal")], 346.8667, TOL_COEF);

    let theta = fit.theta.as_ref().unwrap();
    assert_eq!(theta.len(), 3);
    assert_close("theta[0]", theta[0], 4.5954, TOL_THETA);
    assert_close("theta[1]", theta[1], 3.8097, TOL_THETA);
    assert_close("theta[2]", theta[2], 2.9886, TOL_THETA);

    let loglik = fit.log_likelihood.unwrap();
    assert_close("logLik", loglik, -130.869, TOL_LL);
}
