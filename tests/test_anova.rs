use lme_rs::anova::DdfMethod;
use lme_rs::LmeFit;
use ndarray::{Array1, Array2};

// Simple mock test ignoring actual data
#[test]
fn test_anova_methods_produce_valid_f_tests() {
    let mut fit = LmeFit {
        coefficients: Array1::zeros(2),
        residuals: Array1::zeros(1),
        fitted: Array1::zeros(1),
        ranef: None,
        var_corr: None,
        theta: None,
        sigma2: None,
        reml: Some(0.0),
        log_likelihood: None,
        aic: None,
        bic: None,
        deviance: None,
        b: None,
        u: None,
        beta_se: None,
        beta_t: None,
        formula: None,
        fixed_names: Some(vec!["(Intercept)".to_string(), "Days".to_string()]),
        re_blocks: None,
        num_obs: 180,
        converged: Some(true),
        iterations: Some(10),
        family_name: None,
        link_name: None,
        family: None,
        satterthwaite: None,
        kenward_roger: None,
        v_beta_unscaled: Some(Array2::eye(2)),
        robust: None,
        categorical_levels: None,
    };

    // Test rejection without having evaluated tracking matrices
    let err = fit.anova(DdfMethod::Satterthwaite).unwrap_err();
    assert!(err.to_string().contains("Satterthwaite values missing"));

    // Inject mock evaluations
    let s_df = Array1::from_vec(vec![17.0, 17.0]);
    let s_p = Array1::from_vec(vec![0.001, 0.002]);
    fit.satterthwaite = Some(lme_rs::SatterthwaiteResult {
        dfs: s_df,
        p_values: s_p,
    });

    // Inject corresponding beta_t for F mapping
    let t_vals = Array1::from_vec(vec![10.0, 5.0]);
    fit.beta_t = Some(t_vals);

    // Run anova mapping
    let s_anova = fit.anova(DdfMethod::Satterthwaite).unwrap();

    assert_eq!(s_anova.terms, vec!["Days".to_string()]); // Intercept should be removed
    assert_eq!(s_anova.num_df[0], 1.0); // 1-DoF mapping
    assert_eq!(s_anova.den_df[0], 17.0); // Propagated natively
    assert_eq!(s_anova.f_value[0], 25.0); // t^2 where t=5.0
    assert_eq!(s_anova.p_value[0], 0.002); // Propagated natively
}
