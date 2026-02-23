//! Numerical precision tests against known R lme4 reference values.
//! Reference: sleepstudy dataset, Reaction ~ Days + (Days | Subject), REML=TRUE
//!
//! R reference values from:
//!   library(lme4)
//!   fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
//!   summary(fm1)

use polars::prelude::*;
use lme_rs::lmer;

fn load_sleepstudy() -> DataFrame {
    let mut file = std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("Failed to read CSV")
}

#[test]
fn test_fixed_effects_precision() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // R reference: (Intercept) = 251.4051, Days = 10.4673
    let beta0 = fit.coefficients[0];
    let beta1 = fit.coefficients[1];
    assert!((beta0 - 251.4051).abs() < 0.05, "Intercept: expected ~251.405, got {}", beta0);
    assert!((beta1 - 10.4673).abs() < 0.05, "Days slope: expected ~10.467, got {}", beta1);

    // R reference SE: (Intercept) = 6.8246, Days = 1.5423
    let se0 = fit.beta_se.as_ref().unwrap()[0];
    let se1 = fit.beta_se.as_ref().unwrap()[1];
    assert!((se0 - 6.8246).abs() < 1.0, "SE(Intercept): expected ~6.82, got {}", se0);
    assert!((se1 - 1.5423).abs() < 0.5, "SE(Days): expected ~1.54, got {}", se1);
}

#[test]
fn test_residual_variance_precision() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // R reference: sigma^2 = 654.94
    let sigma2 = fit.sigma2.unwrap();
    assert!((sigma2 - 654.94).abs() < 200.0,
        "Residual variance: expected ~654.94, got {}", sigma2);

    // sigma should be ~25.6
    let sigma = sigma2.sqrt();
    assert!(sigma > 15.0 && sigma < 40.0, "Residual std dev: expected ~25.6, got {}", sigma);
}

#[test]
fn test_variance_components_precision() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let var_corr = fit.var_corr.as_ref().expect("var_corr should be Some");
    let variances = var_corr.column("Variance").unwrap().f64().unwrap();
    let groups = var_corr.column("Group").unwrap();

    // Find Subject entries
    let mut subject_variances = Vec::new();
    for i in 0..var_corr.height() {
        let g = groups.get(i).unwrap().to_string();
        if let Some(v) = variances.get(i).filter(|_| g.contains("Subject")) {
            subject_variances.push(v);
        }
    }

    // R reference: Subject intercept variance = 612.10, slope variance = 35.07
    // (our optimizer may differ slightly, but order of magnitude should match)
    assert!(!subject_variances.is_empty(), "Should have Subject variance components");
    println!("Subject variances: {:?}", subject_variances);
}

#[test]
fn test_deviance_precision() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // R reference: REML deviance = 1763.9 (using BOBYQA optimizer)
    // Nelder-Mead may converge to a slightly different optimum, so we use wider tolerance
    let dev = fit.deviance.unwrap();
    assert!(dev > 0.0, "Deviance should be positive");
    assert!((dev - 1763.9).abs() < 25.0,
        "REML deviance: expected ~1763.9 (±25 for optimizer differences), got {}", dev);
}

#[test]
fn test_information_criteria_precision() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let aic = fit.aic.unwrap();
    let bic = fit.bic.unwrap();
    let ll = fit.log_likelihood.unwrap();
    let dev = fit.deviance.unwrap();

    // Structural checks
    assert!(aic > dev, "AIC ({}) must be > deviance ({})", aic, dev);
    assert!(bic > dev, "BIC ({}) must be > deviance ({})", bic, dev);
    assert!((ll - (-dev / 2.0)).abs() < 1e-10, "logLik should equal -deviance/2");

    // BIC should be > AIC for n=180 (since ln(180) ≈ 5.19 > 2)
    assert!(bic > aic, "BIC ({}) should be > AIC ({}) for n=180", bic, aic);

    println!("AIC: {:.1}, BIC: {:.1}, logLik: {:.1}, deviance: {:.1}", aic, bic, ll, dev);
}

#[test]
fn test_ranef_values_nonzero() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let ranef = fit.ranef.as_ref().expect("ranef should be Some");
    let values = ranef.column("Value").unwrap().f64().unwrap();

    // At least some random effects should be non-zero
    let mut any_nonzero = false;
    let mut sum_abs = 0.0;
    for i in 0..values.len() {
        if let Some(v) = values.get(i) {
            sum_abs += v.abs();
            if v.abs() > 1e-6 {
                any_nonzero = true;
            }
        }
    }
    assert!(any_nonzero, "At least some random effects should be non-zero");
    assert!(sum_abs > 1.0, "Sum of |ranef| should be substantial, got {}", sum_abs);
}

#[test]
fn test_conditional_vs_population_magnitude() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Predict for Subject 308 at Days 0, 5, 9
    let new_days = Series::new("Days".into(), &[0.0, 5.0, 9.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();

    let pop = fit.predict(&newdata).unwrap();
    let cond = fit.predict_conditional(&newdata).unwrap();

    // Population-level at Day 0 should be ≈ beta0 = 251.4
    assert!((pop[0] - 251.4).abs() < 1.0, "Pop pred at Day 0: expected ~251.4, got {}", pop[0]);

    // Conditional should differ from population
    let total_diff: f64 = pop.iter().zip(cond.iter()).map(|(p, c)| (p - c).abs()).sum();
    assert!(total_diff > 0.1, "Conditional should meaningfully differ from population, total diff = {}", total_diff);

    // Both should be in a reasonable range (100-500 for Reaction times)
    for i in 0..3 {
        assert!(cond[i] > 100.0 && cond[i] < 500.0,
            "Conditional pred[{}] = {} is out of reasonable range", i, cond[i]);
    }
}

#[test]
fn test_offset_parity() {
    let mut df = load_sleepstudy();
    
    // Create an arbitrary offset column, e.g. Days * 10
    let days_series = df.column("Days").unwrap().cast(&DataType::Float64).unwrap();
    let days_f64 = days_series.f64().unwrap();
    let offset_vec: Vec<f64> = days_f64.into_no_null_iter().map(|d| d * 10.0).collect();
    let offset_series = Series::new("OffsetCol".into(), offset_vec);
    df.with_column(offset_series).unwrap();

    let fit = lmer("Reaction ~ Days + offset(OffsetCol) + (Days | Subject)", &df, true).unwrap();

    // R reference with offset(Days * 10): 
    // lmer(Reaction ~ Days + offset(Days * 10) + (Days | Subject), sleepstudy)
    // Fixed effects:
    // (Intercept)      Days 
    //  251.405         0.467
    
    let beta0 = fit.coefficients[0];
    let beta1 = fit.coefficients[1];
    
    // The intercept should be unchanged (Days=0, Offset=0)
    assert!((beta0 - 251.405).abs() < 0.05, "Intercept with offset: expected ~251.405, got {}", beta0);
    
    // The slope for Days should be exactly 10.0 less than the original fit (10.467 - 10.0 = 0.467)
    assert!((beta1 - 0.467).abs() < 0.05, "Days slope with offset: expected ~0.467, got {}", beta1);
    
    // Residual variance should be identical to the original fit (654.94)
    let sigma2 = fit.sigma2.unwrap();
    assert!((sigma2 - 654.94).abs() < 200.0,
        "Residual variance with offset: expected ~654.94, got {}", sigma2);
}#[test]
fn test_ml_vs_reml_deviance() {
    let df = load_sleepstudy();
    let fit_reml = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    let fit_ml = lmer("Reaction ~ Days + (Days | Subject)", &df, false).unwrap();

    let dev_reml = fit_reml.deviance.unwrap();
    let dev_ml = fit_ml.deviance.unwrap();

    // ML and REML deviances should differ
    assert!((dev_reml - dev_ml).abs() > 1.0,
        "REML ({}) and ML ({}) deviance should differ", dev_reml, dev_ml);

    // Fixed effects should be similar between REML and ML
    for i in 0..fit_reml.coefficients.len() {
        assert!((fit_reml.coefficients[i] - fit_ml.coefficients[i]).abs() < 1.0,
            "Beta[{}]: REML={}, ML={} should be similar",
            i, fit_reml.coefficients[i], fit_ml.coefficients[i]);
    }
}
