//! Single-binary integration-test harness for hosted non-Linux CI.
//!
//! The canonical `cargo test` flow still runs every file as its own crate on
//! Ubuntu and locally. Windows and macOS use this harness to avoid linking one
//! executable per integration-test file while executing the same test bodies.

#[path = "categorical_anova_test.rs"]
mod categorical_anova_test;
#[path = "test_anova.rs"]
mod test_anova;
#[path = "test_anova_types.rs"]
mod test_anova_types;
#[path = "test_bootstrap.rs"]
mod test_bootstrap;
#[path = "test_conditional_real.rs"]
mod test_conditional_real;
#[path = "test_confint_profile.rs"]
mod test_confint_profile;
#[path = "test_confint_simulate.rs"]
mod test_confint_simulate;
#[path = "test_contrast.rs"]
mod test_contrast;
#[path = "test_coverage_edge_cases.rs"]
mod test_coverage_edge_cases;
#[path = "test_coverage_gaps.rs"]
mod test_coverage_gaps;
#[path = "test_crossed_mock.rs"]
mod test_crossed_mock;
#[path = "test_cv.rs"]
mod test_cv;
#[path = "test_e2e_lmer.rs"]
mod test_e2e_lmer;
#[path = "test_edge_cases.rs"]
mod test_edge_cases;
#[path = "test_emmeans.rs"]
mod test_emmeans;
#[path = "test_errors.rs"]
mod test_errors;
#[path = "test_explorations.rs"]
mod test_explorations;
#[path = "test_external_timings_artifact.rs"]
mod test_external_timings_artifact;
#[path = "test_failure_modes.rs"]
mod test_failure_modes;
#[path = "test_formula.rs"]
mod test_formula;
#[path = "test_formula_stress.rs"]
mod test_formula_stress;
#[path = "test_gaps.rs"]
mod test_gaps;
#[path = "test_glmm.rs"]
mod test_glmm;
#[path = "test_glmm_links.rs"]
mod test_glmm_links;
#[path = "test_glmm_offset_grouseticks.rs"]
mod test_glmm_offset_grouseticks;
#[path = "test_glmm_weighted.rs"]
mod test_glmm_weighted;
#[path = "test_golden_parity.rs"]
mod test_golden_parity;
#[path = "test_intercept_only.rs"]
mod test_intercept_only;
#[path = "test_kenward_roger.rs"]
mod test_kenward_roger;
#[path = "test_kr_modcomp_pastes.rs"]
mod test_kr_modcomp_pastes;
#[path = "test_mcp.rs"]
mod test_mcp;
#[path = "test_ml_optimization.rs"]
mod test_ml_optimization;
#[path = "test_nlmm_agq.rs"]
mod test_nlmm_agq;
#[path = "test_nlmm_bounds.rs"]
mod test_nlmm_bounds;
#[path = "test_nlmm_custom_mean.rs"]
mod test_nlmm_custom_mean;
#[path = "test_nlmm_orange.rs"]
mod test_nlmm_orange;
#[path = "test_nlmm_orange_multi_re.rs"]
mod test_nlmm_orange_multi_re;
#[path = "test_nlmm_self_start.rs"]
mod test_nlmm_self_start;
#[path = "test_nlmm_ss_new_means.rs"]
mod test_nlmm_ss_new_means;
#[path = "test_nlmm_ssasymp.rs"]
mod test_nlmm_ssasymp;
#[path = "test_nlmm_ssasymp_off_orig.rs"]
mod test_nlmm_ssasymp_off_orig;
#[path = "test_nlmm_ssfol.rs"]
mod test_nlmm_ssfol;
#[path = "test_nlmm_ssmicmen.rs"]
mod test_nlmm_ssmicmen;
#[path = "test_nlmm_sspower.rs"]
mod test_nlmm_sspower;
#[path = "test_no_intercept.rs"]
mod test_no_intercept;
#[path = "test_numerical_parity.rs"]
mod test_numerical_parity;
#[path = "test_predict.rs"]
mod test_predict;
#[path = "test_production_load.rs"]
mod test_production_load;
#[path = "test_r_edge_case_matrix.rs"]
mod test_r_edge_case_matrix;
#[path = "test_random_slopes.rs"]
mod test_random_slopes;
#[path = "test_robust.rs"]
mod test_robust;
#[path = "test_satterthwaite.rs"]
mod test_satterthwaite;
#[path = "test_statistical_identities.rs"]
mod test_statistical_identities;
