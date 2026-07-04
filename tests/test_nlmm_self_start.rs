//! `nlmer` automatic starting values (`selfStart` / `getInitial` heuristics).

use lme_rs::nlmm::{nlmer, NlmmStart};
use polars::prelude::*;
use std::fs::File;

const TOL_COEF: f64 = 3.5;
const TOL_SD: f64 = 2.0;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

#[test]
fn orange_sslogis_self_start_matches_lme4() {
    let mut file = File::open("tests/data/orange.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let formula = "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree";
    let fit = nlmer(formula, &df, NlmmStart::new(), false).unwrap();

    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // Same lme4 reference as tests/test_nlmm_orange.rs (explicit start not required).
    assert_close("Asym", coef[idx("Asym")], 192.0528, TOL_COEF);
    assert_close("xmid", coef[idx("xmid")], 727.9045, TOL_COEF);
    assert_close("scal", coef[idx("scal")], 348.0721, TOL_COEF);

    let tau = fit.theta.as_ref().unwrap()[0];
    assert_close("theta (Asym)", tau, 4.035, 4.0);
    let sigma = fit.sigma2.unwrap().sqrt();
    assert_close("RE sd (Asym)", tau * sigma, 31.646, TOL_SD);
}
