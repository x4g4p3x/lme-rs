//! Emit JSON fixed-effect coefficients for cross-language parity checks.
//!
//! Usage: `cargo run --release --example parity_export -- <case>`
//! Cases: `glmm_probit`, `glmm_weighted`, `glmm_offset`, `sleepstudy_offset`

use lme_rs::family::{Family, Link};
use lme_rs::{glmer_weighted_with_link, lmer};
use ndarray::Array1;
use polars::prelude::*;
use serde_json::{json, Map};
use std::env;
use std::fs::File;

fn emit(case: &str, names: &[String], coef: &[f64]) {
    let mut coefficients = Map::new();
    for (name, value) in names.iter().zip(coef.iter()) {
        coefficients.insert(name.clone(), json!(value));
    }
    let payload = json!({
        "case": case,
        "implementation": "rust",
        "coefficients": coefficients,
    });
    println!("{}", payload);
}

fn read_csv(path: &str) -> DataFrame {
    let mut file = File::open(path).expect("csv path");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("read csv")
}

fn fit_glmm_probit() -> (Vec<String>, Vec<f64>) {
    let df = read_csv("tests/data/cbpp_binary.csv");
    let fit = glmer_weighted_with_link(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
        Link::Probit,
        1,
        None,
    )
    .expect("glmm probit");
    let names = fit.fixed_names.unwrap();
    let coef = fit.coefficients.as_slice().unwrap().to_vec();
    (names, coef)
}

fn fit_glmm_weighted() -> (Vec<String>, Vec<f64>) {
    let df = read_csv("tests/data/cbpp_binary_weighted.csv");
    let w = weights_column(&df, "prior_w");
    let fit = glmer_weighted_with_link(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
        Link::Logit,
        1,
        Some(w),
    )
    .expect("glmm weighted");
    let names = fit.fixed_names.unwrap();
    let coef = fit.coefficients.as_slice().unwrap().to_vec();
    (names, coef)
}

fn fit_glmm_offset() -> (Vec<String>, Vec<f64>) {
    let df = read_csv("tests/data/grouseticks.csv");
    let fit = glmer_weighted_with_link(
        "TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD)",
        &df,
        Family::Poisson,
        Link::Log,
        1,
        None,
    )
    .expect("glmm offset");
    let names = fit.fixed_names.unwrap();
    let coef = fit.coefficients.as_slice().unwrap().to_vec();
    (names, coef)
}

fn fit_sleepstudy_offset() -> (Vec<String>, Vec<f64>) {
    let df = read_csv("tests/data/sleepstudy.csv");
    let fit = lmer(
        "Reaction ~ Days + offset(OffsetDays10) + (Days | Subject)",
        &df,
        true,
    )
    .expect("lmer offset");
    let names = fit.fixed_names.unwrap();
    let coef = fit.coefficients.as_slice().unwrap().to_vec();
    (names, coef)
}

fn weights_column(df: &DataFrame, col: &str) -> Array1<f64> {
    let s = df.column(col).expect("weights column");
    Array1::from_iter(s.f64().expect("float weights").into_no_null_iter())
}

fn main() {
    let case = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: parity_export <case>");
        std::process::exit(2);
    });

    let (names, coef) = match case.as_str() {
        "glmm_probit" => fit_glmm_probit(),
        "glmm_weighted" => fit_glmm_weighted(),
        "glmm_offset" => fit_glmm_offset(),
        "sleepstudy_offset" => fit_sleepstudy_offset(),
        other => {
            eprintln!("unknown parity case: {other}");
            std::process::exit(2);
        }
    };
    emit(&case, &names, &coef);
}
