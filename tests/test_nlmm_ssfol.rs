//! `nlmer` with `SSfol` mean on synthetic grouped data (same mean as `SSasymp`).

use lme_rs::nlmm::{nlmer, ssfol_eval, NlmmStart};
use polars::prelude::*;
use std::fs::File;

const TOL_COEF: f64 = 2.0;
const TOL_SD: f64 = 1.5;
const TOL_PRED: f64 = 0.5;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

fn ssfol_fit(start: NlmmStart) -> (lme_rs::LmeFit, DataFrame) {
    let mut file = File::open("tests/data/ssasymp_synthetic.csv").unwrap();
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let fit = nlmer("y ~ SSfol(x, Asym, R0, lrc) ~ Asym|id", &df, start, false).unwrap();
    (fit, df)
}

#[test]
fn ssfol_synthetic_matches_r_reference() {
    let mut start = NlmmStart::new();
    start.insert("Asym".to_string(), 90.0);
    start.insert("R0".to_string(), 20.0);
    start.insert("lrc".to_string(), 0.4_f64.ln());

    let (fit, _) = ssfol_fit(start);
    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // lme4 1.1-38 on tests/data/ssasymp_synthetic.csv with SSfol (same as SSasymp)
    assert_close("Asym", coef[idx("Asym")], 99.9553874, TOL_COEF);
    assert_close("R0", coef[idx("R0")], 16.6873691, TOL_COEF);
    assert_close("lrc", coef[idx("lrc")], -0.6380392, TOL_COEF);

    let tau = fit.theta.as_ref().unwrap()[0];
    assert_close("RE sd (Asym)", tau, 4.571058, TOL_SD);
}

#[test]
fn ssfol_self_start_converges_to_same_fit() {
    let (fit_explicit, _) = {
        let mut start = NlmmStart::new();
        start.insert("Asym".to_string(), 90.0);
        start.insert("R0".to_string(), 20.0);
        start.insert("lrc".to_string(), 0.4_f64.ln());
        ssfol_fit(start)
    };
    let (fit_auto, _) = ssfol_fit(NlmmStart::new());

    let names = fit_explicit.fixed_names.as_ref().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();
    for name in ["Asym", "R0", "lrc"] {
        let i = idx(name);
        assert_close(
            name,
            fit_auto.coefficients.as_slice().unwrap()[i],
            fit_explicit.coefficients.as_slice().unwrap()[i],
            0.05,
        );
    }
}

#[test]
fn ssfol_predict_population_uses_fixed_parameters() {
    let mut start = NlmmStart::new();
    start.insert("Asym".to_string(), 90.0);
    start.insert("R0".to_string(), 20.0);
    start.insert("lrc".to_string(), 0.4_f64.ln());
    let (fit, df) = ssfol_fit(start);

    let pop = fit.predict(&df).unwrap();
    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();
    let asym = coef[idx("Asym")];
    let r0 = coef[idx("R0")];
    let lrc = coef[idx("lrc")];

    let x_col = df.column("x").unwrap().f64().unwrap();
    for i in 0..df.height() {
        let x = x_col.get(i).unwrap();
        let expected = ssfol_eval(asym, r0, lrc, x).0;
        assert_close(&format!("row {i}"), pop[i], expected, TOL_PRED);
    }
}
