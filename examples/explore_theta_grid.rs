//! Profile the REML criterion on a 1-D θ grid and compare it to the MLE.
//!
//! Uses the public `prepare` path: parse → design matrices → [`lme_rs::math::LmmData::evaluate`].

use lme_rs::formula;
use lme_rs::lmer;
use lme_rs::math::LmmData;
use lme_rs::model_matrix::build_design_matrices;
use polars::prelude::*;
use std::fs::File;
use std::path::PathBuf;

fn load_sleepstudy() -> DataFrame {
    let path = PathBuf::from("tests/data/sleepstudy.csv");
    if !path.exists() {
        eprintln!("Could not find {}", path.display());
        eprintln!("Run this example from the repository root.");
        std::process::exit(1);
    }
    let file = File::open(&path).expect("open sleepstudy.csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .expect("read sleepstudy.csv")
}

fn main() {
    let df = load_sleepstudy();
    let formula = "Reaction ~ 1 + (1 | Subject)";
    let ast = formula::parse(formula).expect("parse");
    let matrices = build_design_matrices(&ast, &df).expect("design matrices");
    let lmm = LmmData::new_weighted(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        None,
    );

    let fit = lmer(formula, &df, true).expect("lmer");
    let mle = fit.theta.as_ref().expect("theta")[0];
    let mle_crit = fit.deviance.expect("REML criterion");

    println!("formula: {formula}");
    println!("MLE θ = {mle:.6}   REML criterion = {mle_crit:.6}");
    println!("theta_grid  reml_criterion");

    let mut best_theta = f64::NAN;
    let mut best_crit = f64::INFINITY;
    for step in 1..=80 {
        let theta = 0.05 * f64::from(step);
        let crit = lmm.evaluate(&[theta], true).reml_crit;
        if (step % 4 == 0) || (theta - mle).abs() < 0.03 {
            println!("{theta:10.3}  {crit:.6}");
        }
        if crit < best_crit {
            best_crit = crit;
            best_theta = theta;
        }
    }

    println!(
        "\ngrid min θ = {best_theta:.3} (crit {best_crit:.6}); |θ_grid - θ_MLE| = {:.4}",
        (best_theta - mle).abs()
    );
    assert!(
        (best_theta - mle).abs() < 0.08,
        "grid minimum {best_theta} should sit next to MLE {mle}"
    );
    assert!(
        (best_crit - mle_crit).abs() < 0.5,
        "grid criterion {best_crit} should match MLE criterion {mle_crit}"
    );
    println!("OK");
}
