//! GLMM offset parity on grouseticks (log_height).

use lme_rs::family::Family;
use lme_rs::formula;
use lme_rs::glmer;
use lme_rs::model_matrix;
use polars::prelude::*;
use std::fs::File;

const FORMULA: &str = "TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD)";

#[test]
fn grouseticks_poisson_offset_matches_r_reference() {
    let mut file = File::open("tests/data/grouseticks.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let ast = formula::parse(FORMULA).unwrap();
    assert_eq!(ast.offset.as_deref(), Some("log_height"));
    let mats = model_matrix::build_design_matrices(&ast, &df).unwrap();
    let offset = mats.offset.as_ref().expect("offset vector");
    let mean_off = offset.mean().unwrap();
    assert!(
        mean_off > 5.0 && mean_off < 7.0,
        "mean log_height={mean_off}"
    );

    let fit = glmer(FORMULA, &df, Family::Poisson, 1).unwrap();

    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // R reference from tests/generate_test_data.R (lme4 1.1-38)
    let tol = 0.15;
    assert!((coef[idx("(Intercept)")] - (-5.81090428729957)).abs() <= tol);
    assert!((coef[idx("YEAR96")] - 1.37489000416232).abs() <= tol);
    assert!((coef[idx("YEAR97")] - (-0.90417933743634)).abs() <= tol);
    assert!(fit.converged.unwrap_or(false));
}
