use lme_rs::lmer;
use polars::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct ModelOutput {
    outputs: Outputs,
}

#[derive(Deserialize)]
struct Outputs {
    robust_v_beta: Vec<f64>,
    robust_se: Vec<f64>,
    robust_t: Vec<f64>,
}

fn load_sleepstudy_data() -> DataFrame {
    let file = std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReader::new(file).finish().unwrap()
}

#[test]
fn robust_covariance_matches_orthogonal_design_identity() {
    // Four orthogonal +/-1 columns give (X'X)^-1 = I/16. The residual
    // s8*(1 + .2*s1 - .3*s2) is orthogonal to every fitted column.
    let sign = |i: usize, bit: usize| if i & bit == 0 { 1.0 } else { -1.0 };
    let data = df!(
        "x1" => (0..16).map(|i| sign(i, 1)).collect::<Vec<_>>(),
        "x2" => (0..16).map(|i| sign(i, 2)).collect::<Vec<_>>(),
        "x4" => (0..16).map(|i| sign(i, 4)).collect::<Vec<_>>(),
        "y" => (0..16).map(|i| {
            1.0 + 2.0 * sign(i, 1) - sign(i, 2) + 0.5 * sign(i, 4)
                + sign(i, 8) * (1.0 + 0.2 * sign(i, 1) - 0.3 * sign(i, 2))
        }).collect::<Vec<_>>(),
        "cluster" => (0..16).map(|i| i / 4).collect::<Vec<_>>()
    )
    .unwrap();
    let fit = lme_rs::lm_df("y ~ x1 + x2 + x4", &data).unwrap();
    let hc0 = ndarray::array![
        [1.13, 0.4, -0.6, 0.0],
        [0.4, 1.13, -0.12, 0.0],
        [-0.6, -0.12, 1.13, 0.0],
        [0.0, 0.0, 0.0, 1.13],
    ] / 16.0;
    // Each four-row cluster has score (s8/4)*[1, .2, -.3, s4].
    let clustered = ndarray::array![
        [1.0, 0.2, -0.3, 0.0],
        [0.2, 0.04, -0.06, 0.0],
        [-0.3, -0.06, 0.09, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ] / 4.0;
    for (cluster, expected) in [(None, hc0), (Some("cluster"), clustered)] {
        let actual = lme_rs::robust::compute_robust_se(&fit, &data, cluster).unwrap();
        assert_eq!(actual.v_beta_robust, actual.v_beta_robust.t());
        for (actual, expected) in actual.v_beta_robust.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-14, "{actual} vs {expected}");
        }
        for i in 0..4 {
            assert!((actual.robust_se[i] - expected[[i, i]].sqrt()).abs() < 1e-14);
        }
    }
}

#[test]
fn robust_design_validation_preserves_column_units_and_layouts() {
    // Walsh columns are orthogonal even though two predictors have very
    // different units. A global tolerance would hide changes to the small one.
    let sign = |row: usize, mask: usize| {
        if (row & mask).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    };
    let mut columns = Vec::new();
    let mut terms = Vec::new();
    for mask in 1..16 {
        let units = match mask {
            1 => 1e-100,
            2 => 1e100,
            _ => 1.0,
        };
        let name = format!("x{mask}");
        columns.push(Column::new(
            name.clone().into(),
            (0..128)
                .map(|row| units * sign(row, mask))
                .collect::<Vec<_>>(),
        ));
        terms.push(name);
    }
    columns.push(Column::new(
        "y".into(),
        (0..128)
            .map(|row| 1.0 + sign(row, 1) + sign(row, 64) * (1.0 + 0.2 * sign(row, 2)))
            .collect::<Vec<_>>(),
    ));
    let data = DataFrame::new(columns).unwrap();
    let base = lme_rs::lm_df(&format!("y ~ {}", terms.join(" + ")), &data).unwrap();
    for column_major in [false, true] {
        let mut fit = base.clone();
        if column_major {
            fit.fixed_design_x = fit.fixed_design_x.map(|x| {
                x.reversed_axes()
                    .as_standard_layout()
                    .to_owned()
                    .reversed_axes()
            });
        }
        assert_eq!(
            fit.fixed_design_x.as_ref().unwrap().is_standard_layout(),
            !column_major
        );
        assert!(lme_rs::robust::compute_robust_se(&fit, &data, None).is_ok());
        for column in [1, 2, 3] {
            let mut changed = fit.clone();
            changed.fixed_design_x.as_mut().unwrap()[[0, column]] *= 1.001;
            assert!(lme_rs::robust::compute_robust_se(&changed, &data, None).is_err());
        }
        let mut rounded = fit.clone();
        rounded.fixed_design_x.as_mut().unwrap()[[0, 3]] += 32.0 * f64::EPSILON;
        assert!(lme_rs::robust::compute_robust_se(&rounded, &data, None).is_ok());
        rounded.fixed_design_x.as_mut().unwrap()[[0, 3]] += 128.0 * f64::EPSILON;
        assert!(lme_rs::robust::compute_robust_se(&rounded, &data, None).is_err());
    }
}

#[test]
fn test_robust_se_numerical_parity() {
    let json_str = std::fs::read_to_string("tests/data/random_slopes.json")
        .expect("Failed to read random_slopes.json");
    let test_data: ModelOutput = serde_json::from_str(&json_str).expect("Failed to parse JSON");
    let r_outputs = test_data.outputs;

    let df = load_sleepstudy_data();
    let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Compute CR0 with Subject clustering
    fit.with_robust_se(&df, Some("Subject")).unwrap();
    let robust = fit.robust.as_ref().unwrap();

    let rust_robust_se = &robust.robust_se;
    let rust_robust_t = &robust.robust_t;
    let rust_v_beta = &robust.v_beta_robust;

    // Convert flat R variance matrix to Array2
    let p = 2; // Intercept, Days
    let r_v_beta = ndarray::Array2::from_shape_vec((p, p), r_outputs.robust_v_beta).unwrap();

    let tol = 1e-4;

    for i in 0..p {
        for j in 0..p {
            assert!(
                (rust_v_beta[[i, j]] - r_v_beta[[i, j]]).abs() < tol,
                "V_beta ({}, {}) mismatch. Rust: {}, R: {}",
                i,
                j,
                rust_v_beta[[i, j]],
                r_v_beta[[i, j]]
            );
        }

        assert!(
            (rust_robust_se[i] - r_outputs.robust_se[i]).abs() < tol,
            "Robust SE [{}] mismatch. Rust: {}, R: {}",
            i,
            rust_robust_se[i],
            r_outputs.robust_se[i]
        );

        assert!(
            (rust_robust_t[i] - r_outputs.robust_t[i]).abs() < tol,
            "Robust t-value [{}] mismatch. Rust: {}, R: {}",
            i,
            rust_robust_t[i],
            r_outputs.robust_t[i]
        );
    }
}
