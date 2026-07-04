//! Explicit GLMM link selection (`glmer_with_link`).

use lme_rs::family::{Family, Link};
use lme_rs::{glmer, glmer_with_link};
use polars::prelude::*;
use std::fs::File;

fn load_cbpp() -> DataFrame {
    let mut file = File::open("tests/data/cbpp_binary.csv").expect("cbpp csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("read cbpp")
}

#[test]
fn test_invalid_link_for_family_rejected() {
    let df = load_cbpp();
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let err = glmer_with_link(formula, &df, Family::Binomial, Link::Log, 1).unwrap_err();
    assert!(
        err.to_string().contains("not valid"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_binomial_probit_differs_from_logit() {
    let df = load_cbpp();
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let logit = glmer(formula, &df, Family::Binomial, 1).unwrap();
    let probit = glmer_with_link(formula, &df, Family::Binomial, Link::Probit, 1).unwrap();

    assert_eq!(logit.link_name.as_deref(), Some("logit"));
    assert_eq!(probit.link_name.as_deref(), Some("probit"));

    let diff: f64 = logit
        .coefficients
        .iter()
        .zip(probit.coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-4,
        "probit and logit fits should differ (|Δ| sum = {diff})"
    );
}

#[test]
fn test_poisson_sqrt_link_runs() {
    let mut file = File::open("tests/data/grouseticks.csv").expect("grouseticks");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("read grouseticks");

    let formula = "TICKS ~ YEAR + HEIGHT + (1 | BROOD)";
    let fit = glmer_with_link(formula, &df, Family::Poisson, Link::Sqrt, 1).unwrap();
    assert_eq!(fit.link_name.as_deref(), Some("sqrt"));
    assert!(fit.converged.unwrap_or(false));
}

#[test]
fn test_predict_response_uses_stored_link() {
    let df = load_cbpp();
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let fit = glmer_with_link(formula, &df, Family::Binomial, Link::Probit, 1).unwrap();

    let preds = fit.predict_response(&df).unwrap();
    for p in preds.iter() {
        assert!(
            (0.0..=1.0).contains(p),
            "probit response-scale predictions should be probabilities, got {p}"
        );
    }
}
