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

    let _ = lme_rs::lmer("y ~ x + (1|g)", &df, false);
    let _ = lme_rs::glmer("y ~ x + (1|g)", &df, Family::Gaussian);
}


