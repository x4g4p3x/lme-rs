use lme_rs::optimizer::optimize_theta_nd;
use lme_rs::{lm, lmer, LmeError};
use ndarray::array;
use polars::prelude::*;

#[test]
fn test_lm_dimension_mismatch() {
    let y = array![1.0, 2.0];
    let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0],];
    let result = lm(&y, &x);
    assert!(matches!(result, Err(LmeError::DimensionMismatch { .. })));
}

#[test]
fn test_lm_underdetermined() {
    let y = array![1.0, 2.0];
    let x = array![[1.0, 1.0, 1.0], [1.0, 2.0, 3.0],];
    let result = lm(&y, &x);
    assert!(matches!(result, Err(LmeError::Underdetermined { .. })));
}

#[test]
fn test_lm_sigma2_none() {
    let y = array![1.0, 2.0];
    let x = array![[1.0, 1.0], [1.0, 2.0],];
    let fit = lm(&y, &x).unwrap();
    assert!(fit.sigma2.is_none());
}

#[test]
fn test_lmer_empty_formula() {
    let df = DataFrame::default();
    let result = lmer("   ", &df, true);
    assert!(matches!(result, Err(LmeError::EmptyFormula)));
}

#[test]
fn test_predict_mismatched_columns() {
    let mut file =
        std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Create new data missing the 'Days' column
    let new_subject = Series::new("Subject".into(), &["308"]);
    let newdata = DataFrame::new(vec![new_subject.into()]).unwrap();

    // predict should fail building X matrix
    let result = fit.predict(&newdata);
    assert!(result.is_err());

    // Create new data with an extra unexpected column that alters layout
    // Actually, `predict` explicitly checks `x_names != self.fixed_names` if build_x_matrix succeeds.
    // If we just supply a valid dataframe that parses out different X cols.
    // E.g., if predict tries to parse `Reaction ~ Days...` but Days is missing, `build_x_matrix` errors.
    // We already assert it's an error.
}

#[test]
fn test_optimizer_nan_handling() {
    let mut file =
        std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap();

    let ast = lme_rs::formula::parse("Reaction ~ Days + (Days | Subject)").unwrap();
    let matrices = lme_rs::model_matrix::build_design_matrices(&ast, &df).unwrap();

    // Pass NAN y to propagate f64::NAN through deviance bounds
    let mut nan_y = matrices.y.clone();
    nan_y[0] = f64::NAN;

    let init_theta = ndarray::Array1::from_vec(vec![1.0, 0.0, 1.0]);
    let _res = optimize_theta_nd(
        matrices.x,
        matrices.zt,
        nan_y,
        matrices.re_blocks,
        init_theta,
        true,
        None,
    );
}
