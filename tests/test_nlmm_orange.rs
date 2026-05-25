//! `nlmer` parity on Orange / `SSlogis` (random effect on `Asym` only).

use lme_rs::nlmm::{nlmer, NlmmStart};
use polars::prelude::*;
use std::fs::File;

const TOL_COEF: f64 = 2.0;
const TOL_SD: f64 = 2.0;
const TOL_LL: f64 = 8.0;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

#[test]
fn test_orange_nlmer_sslogis() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("tests/data/orange.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    let mut start = NlmmStart::new();
    start.insert("Asym".to_string(), 200.0);
    start.insert("xmid".to_string(), 725.0);
    start.insert("scal".to_string(), 350.0);

    let formula = "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree";
    let fit = nlmer(formula, &df, start, false)?;

    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // R reference (lme4 1.1-38, Orange dataset)
    assert_close("Asym", coef[idx("Asym")], 192.0528, TOL_COEF);
    assert_close("xmid", coef[idx("xmid")], 727.9045, TOL_COEF);
    assert_close("scal", coef[idx("scal")], 348.0721, TOL_COEF);

    let tau = fit.theta.as_ref().unwrap()[0];
    assert_close("RE sd (Asym)", tau, 31.646, TOL_SD);

    let sigma2 = fit.sigma2.unwrap();
    assert_close("sigma2", sigma2, 61.5128, 10.0);

    let loglik = fit.log_likelihood.unwrap();
    assert_close("logLik", loglik, -131.5719, TOL_LL);

    Ok(())
}
