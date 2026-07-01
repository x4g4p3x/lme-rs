//! Prior observation weights for `glmer_weighted`.

use lme_rs::{family::Family, glmer, glmer_weighted};
use ndarray::Array1;
use polars::prelude::*;

fn cbpp_df() -> DataFrame {
    let mut file = std::fs::File::open("tests/data/cbpp_binary.csv").unwrap();
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap()
}

#[test]
fn glmer_weighted_changes_fit_vs_unweighted() {
    let df = cbpp_df();
    let n = df.height();
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        w.push(0.5 + (i % 7) as f64 * 0.1);
    }
    let formula = "y ~ period2 + period3 + period4 + (1 | herd)";
    let uw = glmer(formula, &df, Family::Binomial, 1).unwrap();
    let ww = glmer_weighted(formula, &df, Family::Binomial, 1, Some(Array1::from_vec(w))).unwrap();
    assert!(uw.converged.unwrap_or(false));
    assert!(ww.converged.unwrap_or(false));
    let udev = uw.deviance.unwrap();
    let wdev = ww.deviance.unwrap();
    assert!(
        (udev - wdev).abs() > 1e-6,
        "weighted and unweighted deviances should differ (u={udev}, w={wdev})"
    );
}

#[test]
fn glmer_weights_validation_matches_lmer() {
    let df = df!(
        "y" => &[0.0_f64, 1.0],
        "x" => &[0.0_f64, 1.0],
        "g" => &["a", "b"],
    )
    .unwrap();
    let bad_len = glmer_weighted(
        "y ~ x + (1 | g)",
        &df,
        Family::Binomial,
        1,
        Some(Array1::from_vec(vec![1.0])),
    )
    .unwrap_err();
    assert!(format!("{bad_len}").contains("weights"));

    let bad_zero = glmer_weighted(
        "y ~ x + (1 | g)",
        &df,
        Family::Binomial,
        1,
        Some(Array1::from_vec(vec![1.0, 0.0])),
    )
    .unwrap_err();
    assert!(format!("{bad_zero}").contains("strictly positive"));
}
