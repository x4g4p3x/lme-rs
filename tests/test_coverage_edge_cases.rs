use lme_rs::family::Family;
use lme_rs::optimizer::{optimize_theta_nd, optimize_theta_glmm};
use ndarray::{array, Array2};
use polars::prelude::*;

#[test]
fn test_lib_log_coverage() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1. Trigger lmer_weighted and glmer to run log::debug! macros
    let df = df!(
        "y" => &[1.0, 2.0, 3.0],
        "x" => &[1.0, 2.0, 3.0],
        "g" => &["A", "A", "B"]
    ).unwrap();

    let _ = lme_rs::lmer("y ~ x + offset(x) + (1|g)", &df, false);
    let _ = lme_rs::glmer("y ~ x + offset(x) + (1|g)", &df, Family::Gaussian);
}



#[test]
fn test_offset_glmer_errors() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    // Test dataset with a broken offset (different type) returning an error
    let df_err = df!(
        "y" => &[1.0, 2.0, 3.0],
        "x" => &[1.0, 2.0, 3.0],
        "g" => &["A", "A", "B"],
        "off" => &["bad", "type", "string"]
    ).unwrap();

    let err1 = lme_rs::glmer("y ~ x + offset(off) + (1|g)", &df_err, lme_rs::family::Family::Gaussian);
    assert!(err1.is_err(), "GLMM should fail when offset is string");

    let err2 = lme_rs::lmer("y ~ x + offset(off) + (1|g)", &df_err, false);
    assert!(err2.is_err(), "LMM should fail when offset is string");
}

