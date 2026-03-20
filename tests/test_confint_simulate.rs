use lme_rs::{family::Family, glmer, lmer};
use polars::prelude::*;
use std::fs::File;

fn load_sleepstudy() -> DataFrame {
    let file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

fn load_grouseticks() -> DataFrame {
    let file = File::open("tests/data/grouseticks.csv").expect("grouseticks.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

fn load_cbpp_binary() -> DataFrame {
    let file = File::open("tests/data/cbpp_binary.csv").expect("cbpp_binary.csv not found");
    CsvReader::new(file).finish().expect("Failed to read CSV")
}

#[test]
fn test_confint_95() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let ci = fit.confint(0.95).unwrap();
    println!("{}", ci);

    // Basic checks
    assert_eq!(ci.names.len(), 2, "Should have 2 fixed effects");
    assert_eq!(ci.lower.len(), 2);
    assert_eq!(ci.upper.len(), 2);
    assert!((ci.level - 0.95).abs() < 1e-10);

    // Lower must be less than upper
    for i in 0..ci.lower.len() {
        assert!(
            ci.lower[i] < ci.upper[i],
            "Lower bound {} should be < upper bound {} for {}",
            ci.lower[i],
            ci.upper[i],
            ci.names[i]
        );
    }

    // The point estimate (beta) should be inside the CI
    for i in 0..fit.coefficients.len() {
        assert!(
            fit.coefficients[i] >= ci.lower[i] && fit.coefficients[i] <= ci.upper[i],
            "Beta {} = {} should be within [{}, {}]",
            ci.names[i],
            fit.coefficients[i],
            ci.lower[i],
            ci.upper[i]
        );
    }
}

#[test]
fn test_confint_99_wider_than_95() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let ci95 = fit.confint(0.95).unwrap();
    let ci99 = fit.confint(0.99).unwrap();

    // 99% CI should be strictly wider than 95%
    for i in 0..ci95.lower.len() {
        assert!(
            ci99.lower[i] < ci95.lower[i],
            "99% lower should be smaller than 95% lower"
        );
        assert!(
            ci99.upper[i] > ci95.upper[i],
            "99% upper should be larger than 95% upper"
        );
    }
}

#[test]
fn test_confint_invalid_level() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    assert!(fit.confint(0.0).is_err(), "level=0 should fail");
    assert!(fit.confint(1.0).is_err(), "level=1 should fail");
    assert!(fit.confint(-0.5).is_err(), "level=-0.5 should fail");
}

#[test]
fn test_confint_display_format() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let ci = fit.confint(0.95).unwrap();
    let output = format!("{}", ci);

    assert!(output.contains("2.5 %"), "Display should show 2.5% column");
    assert!(
        output.contains("97.5 %"),
        "Display should show 97.5% column"
    );
    assert!(
        output.contains("(Intercept)"),
        "Display should show coefficient names"
    );
}

#[test]
fn test_simulate_produces_correct_dimensions() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let sim = fit.simulate(100).unwrap();
    assert_eq!(sim.simulations.len(), 100, "Should have 100 simulations");

    for (i, s) in sim.simulations.iter().enumerate() {
        assert_eq!(
            s.len(),
            fit.num_obs,
            "Simulation {} has wrong length: {} vs {}",
            i,
            s.len(),
            fit.num_obs
        );
    }
}

#[test]
fn test_simulate_variability() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let sim = fit.simulate(10).unwrap();

    // Simulations should not all be identical (i.e., there is randomness)
    let first = &sim.simulations[0];
    let mut any_different = false;
    for s in &sim.simulations[1..] {
        if (first[0] - s[0]).abs() > 1e-10 {
            any_different = true;
            break;
        }
    }
    assert!(
        any_different,
        "Simulations should have stochastic variability"
    );
}

#[test]
fn test_simulate_mean_close_to_fitted() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let nsim = 1000;
    let sim = fit.simulate(nsim).unwrap();

    // Mean of simulations should be close to fitted values (law of large numbers)
    let n = fit.num_obs;
    let mut means = ndarray::Array1::<f64>::zeros(n);
    for s in &sim.simulations {
        means += s;
    }
    means /= nsim as f64;

    let max_diff: f64 = (&means - &fit.fitted)
        .mapv(f64::abs)
        .iter()
        .cloned()
        .fold(0.0, f64::max);
    let sigma = fit.sigma2.unwrap().sqrt();
    // With 1000 sims, mean should be within ~3*sigma/sqrt(1000) ≈ 0.1*sigma of fitted
    assert!(
        max_diff < sigma,
        "Mean of simulations should be close to fitted values, max diff = {}, sigma = {}",
        max_diff,
        sigma
    );
}

#[test]
fn test_simulate_poisson_stays_nonnegative_counts() {
    let df = load_grouseticks();
    let fit = glmer("TICKS ~ YEAR + HEIGHT + (1 | BROOD)", &df, Family::Poisson).unwrap();

    let sim = fit.simulate(5).unwrap();
    for s in &sim.simulations {
        for &y in s {
            assert!(y >= 0.0, "Poisson simulations must be nonnegative, got {}", y);
            assert!(
                (y - y.round()).abs() < 1e-10,
                "Poisson simulations must be integer-valued, got {}",
                y
            );
        }
    }
}

#[test]
fn test_simulate_binomial_stays_binary() {
    let df = load_cbpp_binary();
    let fit = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &df,
        Family::Binomial,
    )
    .unwrap();

    let sim = fit.simulate(5).unwrap();
    for s in &sim.simulations {
        for &y in s {
            assert!(
                (y - 0.0).abs() < 1e-10 || (y - 1.0).abs() < 1e-10,
                "Binomial simulations must be binary, got {}",
                y
            );
        }
    }
}
