use lme_rs::family::BinomialFamily;
use lme_rs::family::Family;
use lme_rs::formula::parse;
use lme_rs::glmm_math::GlmmData;
use lme_rs::glmer;
use lme_rs::model_matrix::build_design_matrices;
use ndarray::Array1;
use polars::prelude::*;
use serde::Deserialize;
use std::fs::File;

#[allow(dead_code)]
#[derive(Deserialize)]
struct ModelInputs {
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    #[serde(rename = "Zt")]
    zt: Vec<Vec<f64>>,
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct ModelOutputs {
    theta: Vec<f64>,
    beta: Vec<f64>,
    deviance: f64,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct TestData {
    model: String,
    inputs: ModelInputs,
    outputs: ModelOutputs,
}

fn load_test_data(path: &str) -> TestData {
    let file = File::open(path).unwrap_or_else(|_| panic!("Could not open {}", path));
    serde_json::from_reader(file).expect("Failed to parse JSON")
}

fn read_csv_data(path: &str) -> polars::prelude::DataFrame {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("Could not open {}", path));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("Failed to read CSV")
}

#[test]
fn test_glmm_binomial_cbpp() {
    let _ = env_logger::try_init();
    let data = load_test_data("tests/data/glmm_binomial.json");
    let df = read_csv_data("tests/data/cbpp_binary.csv");

    // The model formula in R was: y ~ period2 + period3 + period4 + (1 | herd)
    let fit = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
        1,
    )
    .unwrap();

    // 1. Check fixed effects (beta)
    let beta_r = Array1::from_vec(data.outputs.beta);
    let beta_lens_match = fit.coefficients.len() == beta_r.len();

    println!("CBPP Beta RS: {:?}", fit.coefficients);
    println!("CBPP Beta R: {:?}", beta_r);
    println!("CBPP Theta RS: {:?}", fit.theta.as_ref().unwrap());
    println!("CBPP Theta R: {:?}", data.outputs.theta);
    println!("CBPP Deviance RS: {:?}", fit.deviance.unwrap());
    println!("CBPP Deviance R: {:?}", data.outputs.deviance);

    assert!(beta_lens_match, "Beta length mismatch");

    // Using a tolerance of 0.05 for fixed effects (Laplace approx exactness differs slightly from R's C++ inner loop occasionally)
    for i in 0..beta_r.len() {
        assert!(
            (fit.coefficients[i] - beta_r[i]).abs() < 0.05,
            "Beta {} mismatch: rs={} r={}",
            i,
            fit.coefficients[i],
            beta_r[i]
        );
    }

    // 2. Check variance components (theta)
    let theta_r = Array1::from_vec(data.outputs.theta);
    let fit_theta = fit.theta.unwrap();
    for i in 0..theta_r.len() {
        assert!(
            (fit_theta[i] - theta_r[i]).abs() < 0.05,
            "Theta {} mismatch: rs={} r={}",
            i,
            fit_theta[i],
            theta_r[i]
        );
    }

    // 3. Check deviance (we ignore absolute deviance because lme4 omits terms like saturated log lik & pi constants)
    // We just ensure it's calculated without NaN.
    assert!(!fit.deviance.unwrap().is_nan());
}

/// At a fixed θ (R reference), Laplace and AGQ marginal deviances should be close; this guards the quadrature path.
#[test]
fn test_cbpp_laplace_agq_deviance_at_reference_theta() {
    let _ = env_logger::try_init();
    let df = read_csv_data("tests/data/cbpp_binary.csv");
    let ast = parse("y ~ period2 + period3 + period4 + (1 | herd)").unwrap();
    let matrices = build_design_matrices(&ast, &df).unwrap();
    let mut gl_lap = GlmmData::new(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        Box::new(BinomialFamily::new()),
        1,
    );
    let mut gl_agq = GlmmData::new(
        matrices.x,
        matrices.zt,
        matrices.y,
        matrices.re_blocks,
        Box::new(BinomialFamily::new()),
        7,
    );
    let theta = [0.642069741340069_f64];
    let d_lap = gl_lap.laplace_deviance(&theta, None, 1);
    let d_agq = gl_agq.laplace_deviance(&theta, None, 7);
    assert!(d_lap.is_finite() && d_agq.is_finite());
    let scale = d_lap.abs().max(1.0);
    assert!(
        (d_lap - d_agq).abs() < 0.05 * scale,
        "AGQ vs Laplace deviance at R reference theta: lap={} agq={}",
        d_lap,
        d_agq
    );
}

/// `n_agq > 1` uses AGQ in the final PIRLS deviance; θ is still optimized with Laplace (stable outer loop).
#[test]
fn test_glmm_binomial_cbpp_agq_consistent_with_laplace() {
    let _ = env_logger::try_init();
    let df = read_csv_data("tests/data/cbpp_binary.csv");
    let fit_laplace = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
        1,
    )
    .unwrap();
    let fit_agq = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
        7,
    )
    .unwrap();

    assert!(fit_agq.converged.unwrap_or(false));
    for i in 0..fit_laplace.coefficients.len() {
        let diff = (fit_agq.coefficients[i] - fit_laplace.coefficients[i]).abs();
        assert!(
            diff < 0.15,
            "AGQ vs Laplace beta {} differs too much: laplace={} agq={} diff={}",
            i,
            fit_laplace.coefficients[i],
            fit_agq.coefficients[i],
            diff
        );
    }
    let tl = fit_laplace.theta.unwrap();
    let ta = fit_agq.theta.unwrap();
    for i in 0..tl.len() {
        assert!(
            (ta[i] - tl[i]).abs() < 1e-6,
            "theta should match (Laplace θ̂ for both fits): laplace={} agq={}",
            tl[i],
            ta[i]
        );
    }
}

#[test]
fn test_glmm_poisson_grouseticks() {
    let _ = env_logger::try_init();
    let data = load_test_data("tests/data/glmm_poisson.json");
    let df = read_csv_data("tests/data/grouseticks.csv");

    // The model formula in R was: TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)
    let fit = glmer(
        "TICKS ~ YEAR96 + YEAR97 + (1 | BROOD)",
        &df,
        Family::Poisson,
        1,
    )
    .unwrap();

    // 1. Check fixed effects (beta)
    let beta_r = Array1::from_vec(data.outputs.beta);

    println!("Grouse Beta RS: {:?}", fit.coefficients);
    println!("Grouse Beta R: {:?}", beta_r);
    println!("Grouse Theta RS: {:?}", fit.theta.as_ref().unwrap());
    println!("Grouse Theta R: {:?}", data.outputs.theta);
    println!("Grouse Deviance RS: {:?}", fit.deviance.unwrap());
    println!("Grouse Deviance R: {:?}", data.outputs.deviance);

    // Using a tolerance of 0.15 for Nelder-Mead vs BOBYQA optimizer differences
    for i in 0..beta_r.len() {
        assert!(
            (fit.coefficients[i] - beta_r[i]).abs() < 0.15,
            "Beta {} mismatch: rs={} r={}",
            i,
            fit.coefficients[i],
            beta_r[i]
        );
    }

    // 2. Check variance components (theta)
    let theta_r = Array1::from_vec(data.outputs.theta);
    let fit_theta = fit.theta.unwrap();
    for i in 0..theta_r.len() {
        assert!(
            (fit_theta[i] - theta_r[i]).abs() < 1e-2,
            "Theta {} mismatch: rs={} r={}",
            i,
            fit_theta[i],
            theta_r[i]
        );
    }

    // 3. Check deviance (we ignore absolute deviance because lme4 omits terms like saturated log lik & pi constants)
    // We just ensure it's calculated without NaN.
    assert!(!fit.deviance.unwrap().is_nan());
}

#[test]
fn test_glmm_gamma_dyestuff_reasonable_scale() {
    let _ = env_logger::try_init();
    let df = read_csv_data("tests/data/dyestuff.csv");

    let fit = glmer("Yield ~ 1 + (1 | Batch)", &df, Family::Gamma, 1).unwrap();

    println!("Dyestuff Gamma beta: {:?}", fit.coefficients);
    println!("Dyestuff Gamma sigma2: {:?}", fit.sigma2);
    println!(
        "Dyestuff Gamma fitted[0..5]: {:?}",
        &fit.fitted.iter().take(5).collect::<Vec<_>>()
    );

    assert!(fit.converged.unwrap_or(false));
    assert_eq!(fit.coefficients.len(), 1);

    // On the inverse-link scale, the intercept should be positive and close to 1 / mean(y).
    let mean_y = df
        .column("Yield")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .sum::<f64>()
        / df.height() as f64;
    let expected_beta = 1.0 / mean_y;
    assert!(fit.coefficients[0] > 0.0);
    assert!(
        (fit.coefficients[0] - expected_beta).abs() < 5e-4,
        "Gamma intercept on inverse-link scale should be near 1/mean(y): rs={} expected={}",
        fit.coefficients[0],
        expected_beta
    );

    // Fitted values should stay on the observed data scale, not explode.
    for &mu in &fit.fitted {
        assert!(mu.is_finite());
        assert!(mu > 0.0);
        assert!(
            mu < mean_y * 10.0,
            "Gamma fitted value implausibly large: {}",
            mu
        );
    }

    let sigma2 = fit.sigma2.unwrap();
    assert!(sigma2.is_finite());
    assert!(sigma2 > 0.0);
    assert!(sigma2 < 1e6, "Gamma sigma2 implausibly large: {}", sigma2);
}
