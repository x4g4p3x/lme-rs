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
fn test_residuals_fitted_computed() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 1: residuals and fitted values should be non-zero
    let _residuals_sum: f64 = fit.residuals.iter().sum();
    let fitted_sum: f64 = fit.fitted.iter().sum();
    
    // Fitted values must be non-trivial (not all zeros)
    assert!(fitted_sum.abs() > 1.0, "Fitted values should not all be zero");
    
    // y ≈ fitted + residuals (should hold exactly)
    let y_series = df.column("Reaction").unwrap().cast(&DataType::Float64).unwrap();
    let y_f64 = y_series.f64().unwrap();
    let y_vec: Vec<f64> = y_f64.into_no_null_iter().collect();
    
    for (i, &y_val) in y_vec.iter().enumerate() {
        let reconstructed = fit.fitted[i] + fit.residuals[i];
        assert!(
            (reconstructed - y_val).abs() < 1e-6,
            "y[{}] = {} but fitted + residuals = {}", i, y_val, reconstructed
        );
    }
}

#[test]
fn test_ranef_dataframe_structure() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 2: ranef should be Some and have correct structure
    let ranef = fit.ranef.as_ref().expect("ranef should be Some");
    assert!(ranef.height() > 0, "ranef DataFrame should have rows");
    
    // Should have columns: Grouping, Group, Effect, Value
    assert!(ranef.column("Grouping").is_ok(), "ranef should have 'Grouping' column");
    assert!(ranef.column("Group").is_ok(), "ranef should have 'Group' column");
    assert!(ranef.column("Effect").is_ok(), "ranef should have 'Effect' column");
    assert!(ranef.column("Value").is_ok(), "ranef should have 'Value' column");
    
    // With (Days | Subject), 18 subjects × 2 effects (intercept + slope) = 36 rows
    assert_eq!(ranef.height(), 36, "Expected 18 subjects × 2 effects = 36 rows");
    
    println!("ranef DataFrame:\n{}", ranef);
}

#[test]
fn test_var_corr_positive_definite() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 3: var_corr should be Some with positive variances
    let var_corr = fit.var_corr.as_ref().expect("var_corr should be Some");
    assert!(var_corr.height() > 0, "var_corr DataFrame should have rows");
    
    let variances = var_corr.column("Variance").unwrap().f64().unwrap();
    // Diagonal entries (variances) must be positive
    let stddev = var_corr.column("StdDev").unwrap().f64().unwrap();
    
    for i in 0..variances.len() {
        if let Some(v) = variances.get(i) {
            // Diagonal variances should be positive
            if stddev.get(i).is_some_and(|sd| !sd.is_nan()) {
                assert!(v >= 0.0, "Variance at row {} should be non-negative, got {}", i, v);
            }
        }
    }
    
    println!("var_corr DataFrame:\n{}", var_corr);
}

#[test]
fn test_log_likelihood_sign() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 4: log_likelihood should be computed and negative
    let ll = fit.log_likelihood.expect("log_likelihood should be Some");
    assert!(ll < 0.0, "log-likelihood should be negative for real data, got {}", ll);
    
    // It should be -deviance/2
    let dev = fit.deviance.expect("deviance should be Some");
    assert!((ll - (-dev / 2.0)).abs() < 1e-10, "log_lik should equal -deviance/2");
}

#[test]
fn test_aic_bic_reasonable() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 9: AIC and BIC should be computed
    let aic = fit.aic.expect("AIC should be Some");
    let bic = fit.bic.expect("BIC should be Some");
    let dev = fit.deviance.expect("deviance should be Some");
    
    // AIC = deviance + 2*n_params, so AIC > deviance
    assert!(aic > dev, "AIC ({}) should be > deviance ({})", aic, dev);
    // BIC = deviance + n_params * ln(n), so BIC > deviance  
    assert!(bic > dev, "BIC ({}) should be > deviance ({})", bic, dev);
    
    println!("AIC: {:.1}, BIC: {:.1}, deviance: {:.1}", aic, bic, dev);
}

#[test]
fn test_convergence_reported() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 10: convergence should be reported
    let converged = fit.converged.expect("converged should be Some");
    assert!(converged, "Optimizer should converge on sleepstudy data");
    
    let iterations = fit.iterations.expect("iterations should be Some");
    assert!(iterations > 0, "Should have done at least 1 iteration");
    assert!(iterations < 1000, "Should converge in fewer than max iterations");
    
    println!("Converged: {}, iterations: {}", converged, iterations);
}

#[test]
fn test_display_summary_format() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 6 + overall: Display should include correlations, AIC/BIC, convergence
    let summary = format!("{}", fit);
    
    assert!(summary.contains("AIC"), "Summary should contain AIC");
    assert!(summary.contains("BIC"), "Summary should contain BIC");
    assert!(summary.contains("logLik"), "Summary should contain logLik");
    assert!(summary.contains("Corr:"), "Summary should contain correlation for random slopes");
    assert!(summary.contains("converged"), "Summary should contain convergence info");
    assert!(summary.contains("Scaled residuals:"), "Summary should contain scaled residuals");
    
    println!("Full summary:\n{}", summary);
}

#[test]
fn test_conditional_predictions() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    
    // Gap 7: conditional predictions should exist
    let new_days = Series::new("Days".into(), &[0.0, 1.0, 5.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();
    
    let pop_preds = fit.predict(&newdata).unwrap();
    let cond_preds = fit.predict_conditional(&newdata).unwrap();
    
    // Both should succeed and have the same length
    assert_eq!(pop_preds.len(), 3);
    assert_eq!(cond_preds.len(), 3);
    
    println!("Population predictions: {:?}", pop_preds.to_vec());
    println!("Conditional predictions: {:?}", cond_preds.to_vec());
}

#[test]
fn test_lib_rs_edge_cases() {
    use lme_rs::family::Family;
    use lme_rs::{glmer, lmer_weighted, anova, AnovaResult};

    let df = load_sleepstudy();

    // 1. Empty formula errors
    assert!(glmer(" ", &df, Family::Binomial).is_err());
    assert!(lmer_weighted(" ", &df, true, None).is_err());

    // 2. Anova same parameters error
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    let res = anova(&fit, &fit);
    assert!(res.is_err());

    // 3. GLMM Gaussian family and predict_response limits
    let fit_gauss = glmer("Reaction ~ Days + (1 | Subject)", &df, Family::Gaussian).unwrap();
    assert!(fit_gauss.sigma2.is_some()); // uses dispersion

    let nd = DataFrame::new(vec![
        Series::new("Days".into(), &[0.0, 1.0]).into(),
        Series::new("Subject".into(), &["308", "308"]).into()
    ]).unwrap();

    let p_res = fit_gauss.predict_response(&nd).unwrap();
    let pc_res = fit_gauss.predict_conditional_response(&nd).unwrap();
    assert_eq!(p_res.len(), 2);
    assert_eq!(pc_res.len(), 2);

    // 4. confint without fixed names
    use lme_rs::lm;
    use ndarray::{array, Array2};
    let y = array![1.0, 2.0];
    let x = Array2::<f64>::ones((2, 1));
    let mut lm_fit = lm(&y, &x).unwrap();
    lm_fit.beta_se = Some(array![0.5]); // mock SE
    let ci = lm_fit.confint(0.95).unwrap();
    assert_eq!(ci.names[0], "beta_0");

    // 5. Display GLMM and AnovaResult
    let glmm_str = format!("{}", fit_gauss);
    assert!(glmm_str.contains("Generalized linear mixed model fit"));
    assert!(glmm_str.contains("Family: gaussian"));

    let a_res = AnovaResult {
        n_params_0: 1, n_params_1: 2,
        deviance_0: 10.0, deviance_1: 5.0,
        chi_sq: 5.0, df: 1, p_value: 0.05,
        formula_0: "y ~ 1".into(), formula_1: "y ~ x".into(),
    };
    let a_str = format!("{}", a_res);
    assert!(a_str.contains("  1     2      5.00"));

    // 6. Simulate from GLMM
    let sim = fit_gauss.simulate(2).unwrap();
    assert_eq!(sim.simulations.len(), 2);
}

#[test]
fn test_lib_coverage_remaining() {
    let df = load_sleepstudy();

    // 1. LMM conditional prediction (identity link)
    let fit = lme_rs::lmer("Reaction ~ Days + (1|Subject)", &df, false).unwrap();
    let nd = fit.predict_conditional_response(&df);
    assert!(nd.is_ok());

    // 2. ANOVA precise formatting and PR string (using unweighted LMM)
    let fit_b = lme_rs::lmer("Reaction ~ 1 + (1|Subject)", &df, false).unwrap();
    let res = lme_rs::anova(&fit_b, &fit).unwrap(); // Compare nested models
    let s = format!("{}", res);
    assert!(s.contains("Pr(>Chisq)"));

    // 3. Nested RE with missing column
    let bad_nested_ast = lme_rs::formula::parse("Reaction ~ Days + (1 | Subject:missing)").unwrap();
    let res = lme_rs::model_matrix::build_design_matrices(&bad_nested_ast, &df);
    assert!(res.is_err());
    if let Err(e) = res {
        assert!(e.to_string().contains("not found"));
    }

}
