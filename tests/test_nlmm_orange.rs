//! `nlmer` parity on Orange / `SSlogis` (random effect on `Asym` only).

use lme_rs::nlmm::sslogis_eval;
use lme_rs::nlmm::{nlmer, NlmmStart};
use polars::prelude::*;
use std::fs::File;

const TOL_COEF: f64 = 3.5;
const TOL_SD: f64 = 2.0;
const TOL_LL: f64 = 12.0;
const TOL_PRED: f64 = 1e-3;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "{name}: got {got}, expected ~{expected} (|Δ|={diff} > tol {tol})"
    );
}

fn orange_fit() -> Result<(lme_rs::LmeFit, DataFrame), Box<dyn std::error::Error>> {
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
    Ok((fit, df))
}

#[test]
fn test_orange_nlmer_sslogis() -> Result<(), Box<dyn std::error::Error>> {
    let (fit, _df) = orange_fit()?;

    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();

    // R reference (lme4 1.1-38, Orange dataset)
    assert_close("Asym", coef[idx("Asym")], 192.0528, TOL_COEF);
    assert_close("xmid", coef[idx("xmid")], 727.9045, TOL_COEF);
    assert_close("scal", coef[idx("scal")], 348.0721, TOL_COEF);

    let tau = fit.theta.as_ref().unwrap()[0];
    assert_close("theta (Asym)", tau, 4.035, 4.0);
    let sigma = fit.sigma2.unwrap().sqrt();
    assert_close("RE sd (Asym)", tau * sigma, 31.646, TOL_SD);

    let sigma2 = fit.sigma2.unwrap();
    assert_close("sigma2", sigma2, 61.5128, 10.0);

    let loglik = fit.log_likelihood.unwrap();
    assert_close("logLik", loglik, -131.5719, TOL_LL);

    Ok(())
}

#[test]
fn test_orange_nlmer_predict_population() -> Result<(), Box<dyn std::error::Error>> {
    let (fit, df) = orange_fit()?;

    let pop = fit.predict(&df)?;
    let names = fit.fixed_names.as_ref().unwrap();
    let coef = fit.coefficients.as_slice().unwrap();
    let idx = |n: &str| names.iter().position(|x| x == n).unwrap();
    let asym = coef[idx("Asym")];
    let xmid = coef[idx("xmid")];
    let scal = coef[idx("scal")];

    let age_col = df.column("age")?;
    for i in 0..df.height() {
        let age = match age_col.dtype() {
            DataType::Float64 => age_col.f64()?.get(i).unwrap(),
            DataType::Int64 => age_col.i64()?.get(i).unwrap() as f64,
            other => panic!("unexpected age dtype: {other:?}"),
        };
        let expected = sslogis_eval(asym, xmid, scal, age).0;
        assert_close(&format!("population row {i}"), pop[i], expected, TOL_PRED);
    }

    Ok(())
}

#[test]
fn test_orange_nlmer_predict_conditional_matches_fitted() -> Result<(), Box<dyn std::error::Error>>
{
    let (fit, df) = orange_fit()?;

    let cond = fit.predict_conditional(&df, false)?;
    let fitted = fit.fitted.as_slice().unwrap();

    for (i, (c, f)) in cond.iter().zip(fitted.iter()).enumerate() {
        assert_close(&format!("conditional row {i}"), *c, *f, TOL_PRED);
    }

    let pop = fit.predict(&df)?;
    let diff: f64 = pop
        .iter()
        .zip(cond.iter())
        .map(|(p, c)| (p - c).abs())
        .sum();
    assert!(
        diff > 1.0,
        "population and conditional predictions should differ on training data (|Δ| sum = {diff})"
    );

    Ok(())
}

#[test]
fn test_orange_nlmer_predict_new_level_errors() -> Result<(), Box<dyn std::error::Error>> {
    let (fit, df) = orange_fit()?;

    let df_tree_str = df
        .clone()
        .lazy()
        .with_column(col("Tree").cast(DataType::String))
        .collect()?;
    let trees: Vec<String> = df_tree_str
        .column("Tree")?
        .str()?
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            if i == 0 {
                "999".to_string()
            } else {
                v.unwrap().to_string()
            }
        })
        .collect();
    let mut df_clone = df_tree_str;
    let new_df = df_clone.with_column(Column::new("Tree".into(), &trees))?;

    let err = fit
        .predict_conditional(new_df, false)
        .expect_err("expected error for unseen Tree level");
    assert!(
        err.to_string().contains("New level"),
        "unexpected error: {err}"
    );

    let ok = fit.predict_conditional(new_df, true)?;
    assert_eq!(ok.len(), df.height());

    Ok(())
}
