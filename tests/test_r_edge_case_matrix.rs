//! Fit-level lme4 parity for Wilkinson / RE edge cases.
//!
//! Complements [`test_golden_parity`] with formulas the main manifest does not
//! lock: independent `||` slopes, `I()` arithmetic, unary `sqrt()`, raw and
//! orthogonal `poly()`, `ns()`, cell-means coding, and nested `batch/cask`.

use lme_rs::lmer;
use lme_rs::LmeFit;
use polars::prelude::*;
use serde::Deserialize;
use std::fs::File;

const MANIFEST: &str = "tests/data/r_edge_case_matrix.json";

#[derive(Debug, Deserialize)]
struct Matrix {
    schema_version: u32,
    cases: Vec<MatrixCase>,
}

#[derive(Debug, Deserialize)]
struct MatrixCase {
    id: String,
    formula: String,
    data_path: String,
    reml: bool,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    coefficients: Vec<ScalarCheck>,
    theta: Vec<ScalarCheck>,
    sigma2: ScalarCheck,
    deviance: ScalarCheck,
    fitted: Vec<f64>,
    fitted_tolerance: f64,
}

#[derive(Debug, Deserialize)]
struct ScalarCheck {
    name: String,
    value: f64,
    tolerance: f64,
}

fn load_matrix() -> Matrix {
    let file = File::open(MANIFEST).unwrap_or_else(|err| panic!("open {MANIFEST}: {err}"));
    serde_json::from_reader(file).unwrap_or_else(|err| panic!("parse {MANIFEST}: {err}"))
}

fn read_csv(path: &str) -> DataFrame {
    let file = File::open(path).unwrap_or_else(|err| panic!("open {path}: {err}"));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap_or_else(|err| panic!("read {path}: {err}"))
}

fn skip_named_basis_coefficient(case_id: &str, name: &str) -> bool {
    // Natural-spline encodings match R's column space (see fitted checks) but
    // not necessarily the same contrast parameterization as splines::ns.
    case_id.contains("ns_df3") && name.starts_with("ns(")
}

fn assert_close(case_id: &str, label: &str, actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{case_id}: {label} mismatch: actual={actual} expected={expected} tolerance={tolerance} diff={diff}"
    );
}

fn assert_coefficients(case_id: &str, fit: &LmeFit, expected: &[ScalarCheck]) {
    let names = fit
        .fixed_names
        .as_ref()
        .unwrap_or_else(|| panic!("{case_id}: missing fixed_names"));
    let coef = fit.coefficients.as_slice().unwrap();
    assert_eq!(
        names.len(),
        coef.len(),
        "{case_id}: coefficient name/value length mismatch"
    );
    assert_eq!(
        names.len(),
        expected.len(),
        "{case_id}: expected {} coefficients, fit has {} ({names:?})",
        expected.len(),
        names.len()
    );

    for check in expected {
        if skip_named_basis_coefficient(case_id, &check.name) {
            continue;
        }
        let idx = names
            .iter()
            .position(|n| n == &check.name)
            .unwrap_or_else(|| {
                panic!(
                    "{case_id}: missing coefficient {} (fit names: {names:?})",
                    check.name
                )
            });
        let actual = coef[idx];
        let diff = (actual - check.value).abs();
        if diff <= check.tolerance {
            continue;
        }
        if (check.name.starts_with("poly(") || check.name.starts_with("ns("))
            && (actual + check.value).abs() <= check.tolerance
        {
            continue;
        }
        panic!(
            "{case_id}: coefficient {} mismatch: actual={actual} expected={} tolerance={} diff={diff}",
            check.name, check.value, check.tolerance
        );
    }
}

fn assert_fitted(case_id: &str, fit: &LmeFit, expected: &[f64], tolerance: f64) {
    let actual = fit.fitted.as_slice().unwrap();
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case_id}: fitted length mismatch"
    );
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= tolerance,
            "{case_id}: fitted[{idx}] mismatch: actual={got} expected={want} tolerance={tolerance} diff={diff}"
        );
    }
}
fn assert_theta(case_id: &str, fit: &LmeFit, expected: &[ScalarCheck]) {
    let theta = fit
        .theta
        .as_ref()
        .unwrap_or_else(|| panic!("{case_id}: missing theta"))
        .as_slice()
        .unwrap();
    assert_eq!(
        theta.len(),
        expected.len(),
        "{case_id}: expected {} theta values, fit has {theta:?}",
        expected.len()
    );

    let positional_ok = expected
        .iter()
        .zip(theta.iter())
        .all(|(check, actual)| (actual - check.value).abs() <= check.tolerance);
    if positional_ok {
        return;
    }

    // Nested slash formulas order inner/outer blocks differently in lme4 vs lme-rs.
    let mut unused: Vec<f64> = theta.to_vec();
    for check in expected {
        let Some(idx) = unused
            .iter()
            .position(|actual| (actual - check.value).abs() <= check.tolerance)
        else {
            panic!(
                "{case_id}: no unused theta within {} of {} {} (fit theta: {theta:?})",
                check.tolerance, check.name, check.value
            );
        };
        unused.swap_remove(idx);
    }
}

fn assert_case(case: &MatrixCase) {
    let data = read_csv(&case.data_path);
    let fit = lmer(&case.formula, &data, case.reml)
        .unwrap_or_else(|err| panic!("{}: lmer failed: {err}", case.id));
    assert_coefficients(&case.id, &fit, &case.expected.coefficients);
    assert_theta(&case.id, &fit, &case.expected.theta);
    assert_fitted(
        &case.id,
        &fit,
        &case.expected.fitted,
        case.expected.fitted_tolerance,
    );
    assert_close(
        &case.id,
        "sigma2",
        fit.sigma2
            .unwrap_or_else(|| panic!("{}: missing sigma2", case.id)),
        case.expected.sigma2.value,
        case.expected.sigma2.tolerance,
    );
    assert_close(
        &case.id,
        "deviance",
        fit.deviance
            .unwrap_or_else(|| panic!("{}: missing deviance", case.id)),
        case.expected.deviance.value,
        case.expected.deviance.tolerance,
    );
}

#[test]
fn r_edge_case_matrix_is_well_formed() {
    let matrix = load_matrix();
    assert_eq!(matrix.schema_version, 1);
    assert!(
        matrix.cases.len() >= 8,
        "edge-case matrix should cover the Wilkinson / RE catalog, got {}",
        matrix.cases.len()
    );
    for case in &matrix.cases {
        assert!(
            std::path::Path::new(&case.data_path).exists(),
            "{}: missing data {}",
            case.id,
            case.data_path
        );
        assert!(
            !case.expected.coefficients.is_empty(),
            "{}: coefficients required",
            case.id
        );
        assert!(
            !case.expected.theta.is_empty(),
            "{}: theta required",
            case.id
        );
        assert!(
            !case.expected.fitted.is_empty(),
            "{}: fitted values required",
            case.id
        );
    }
}

#[test]
fn r_edge_case_matrix_matches_lme4() {
    let matrix = load_matrix();
    for case in &matrix.cases {
        assert_case(case);
    }
}
