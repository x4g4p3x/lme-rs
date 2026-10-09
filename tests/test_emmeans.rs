//! Estimated marginal means and pairwise comparisons vs R `emmeans`.

use lme_rs::{lmer, McpAdjust};
use polars::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[test]
fn proportional_joint_weights_match_observed_prediction_averages() {
    use lme_rs::{lm_df_with_factors, FactorCoding, FactorSpec, GridWeights, ReferenceGrid};
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut y = Vec::new();
    for ai in 0..2 {
        for bi in 0..2 {
            for ci in 0..2 {
                for repeat in 0..4 {
                    a.push(if ai == 0 { "A" } else { "B" });
                    b.push(if bi == 0 { 10_i32 } else { 20 });
                    c.push(ci != 0);
                    y.push((ai + 2 * bi + 3 * ci + 2 * bi * ci) as f64 + repeat as f64 / 10.0);
                }
            }
        }
    }
    let data = df!("y" => y, "a" => a, "b" => b, "c" => c).unwrap();
    let factors = [
        (
            "a".into(),
            FactorSpec {
                levels: vec!["B".into(), "A".into()],
                coding: FactorCoding::Sum,
            },
        ),
        (
            "b".into(),
            FactorSpec {
                levels: vec!["20".into(), "10".into()],
                coding: FactorCoding::Treatment,
            },
        ),
        (
            "c".into(),
            FactorSpec {
                levels: vec!["true".into(), "false".into()],
                coding: FactorCoding::Sum,
            },
        ),
    ]
    .into();
    let fit = lm_df_with_factors("y ~ a + b*c", &data, &factors).unwrap();
    let grid = ReferenceGrid {
        terms: vec!["a".into()],
        weights: GridWeights::Proportional,
        ..Default::default()
    };
    // Joint frequencies differ from products of the marginal frequencies, and
    // one nuisance combination is absent. Changing data must change weights.
    for reference in [
        df!("b" => [20_i32, 20, 10, 10, 10], "c" => [false, true, true, true, true]).unwrap(),
        df!("b" => [10_i32, 20, 20, 20, 10], "c" => [false, false, false, true, false]).unwrap(),
    ] {
        let means = fit
            .emmeans_with_grid(&reference, &grid, 0.95, None)
            .unwrap();
        for (i, cell) in means.cells.iter().enumerate() {
            let mut prediction_data = reference.clone();
            prediction_data
                .with_column(Column::new(
                    "a".into(),
                    vec![cell["a"].clone(); reference.height()],
                ))
                .unwrap();
            let predicted = fit.predict(&prediction_data).unwrap();
            let expected = predicted.mean().unwrap();
            assert!((means.estimate[i] - expected).abs() < 1e-12);
            let x = fit
                .model_spec()
                .prediction_matrix(&prediction_data)
                .unwrap()
                .0;
            let row = x.mean_axis(ndarray::Axis(0)).unwrap();
            assert!((&means.linfct.row(i) - &row)
                .iter()
                .all(|v| v.abs() < 1e-12));
            let variance =
                row.dot(&(fit.v_beta_unscaled.as_ref().unwrap() * fit.sigma2.unwrap()).dot(&row));
            assert!((means.std_error[i] - variance.sqrt()).abs() < 1e-12);
        }
        let pairs = fit
            .emmeans_pairs_with_grid(&reference, &grid, McpAdjust::None, None)
            .unwrap();
        assert!((pairs.estimate[0] - (means.estimate[1] - means.estimate[0])).abs() < 1e-12);
        let reversed = fit
            .emmeans_with_grid(&reference.reverse(), &grid, 0.95, None)
            .unwrap();
        assert_eq!(reversed.linfct, means.linfct);
    }
    for invalid in [
        df!("b" => [Some(20_i32), None], "c" => [true, false]).unwrap(),
        df!("b" => [20_i32, 99], "c" => [true, false]).unwrap(),
    ] {
        assert!(fit.emmeans_with_grid(&invalid, &grid, 0.95, None).is_err());
        assert!(fit
            .emmeans_pairs_with_grid(&invalid, &grid, McpAdjust::None, None)
            .is_err());
    }
}

#[derive(Deserialize)]
struct PastesEmmeansFixture {
    levels: Vec<String>,
    estimate: Vec<f64>,
    std_error: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    comparisons: Vec<String>,
    pair_estimate: Vec<f64>,
    pair_std_error: Vec<f64>,
    pair_z: Vec<f64>,
    pair_p_tukey: Vec<f64>,
}

fn load_pastes() -> DataFrame {
    let file = File::open("tests/data/pastes.csv").expect("pastes.csv");
    CsvReader::new(file).finish().expect("read pastes")
}

fn load_fixture() -> PastesEmmeansFixture {
    let mut buf = String::new();
    File::open("tests/data/pastes_emmeans.json")
        .expect("pastes_emmeans.json")
        .read_to_string(&mut buf)
        .unwrap();
    serde_json::from_str(&buf).expect("parse fixture")
}

fn assert_vec_close(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "{label} length");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: got {g}, expected {e}, |delta|={diff} > {tol}"
        );
    }
}

#[test]
fn pastes_cask_emmeans_matches_r_asymptotic_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let gold = load_fixture();
    let df = load_pastes();
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;

    let means = fit.emmeans("cask", &df, 0.95, None)?;
    assert_eq!(means.levels, gold.levels);
    assert_eq!(means.statistic, "z");
    assert!(means.den_df.iter().all(|df| df.is_infinite()));
    assert_vec_close(
        "estimate",
        means.estimate.as_slice().unwrap(),
        &gold.estimate,
        1e-8,
    );
    assert_vec_close(
        "std_error",
        means.std_error.as_slice().unwrap(),
        &gold.std_error,
        2e-5,
    );
    assert_vec_close("lower", means.lower.as_slice().unwrap(), &gold.lower, 4e-5);
    assert_vec_close("upper", means.upper.as_slice().unwrap(), &gold.upper, 4e-5);

    let pairs = fit.emmeans_pairs("cask", &df, McpAdjust::Tukey, None)?;
    assert_eq!(pairs.comparisons, gold.comparisons);
    assert_vec_close(
        "pair estimate",
        pairs.estimate.as_slice().unwrap(),
        &gold.pair_estimate,
        1e-8,
    );
    assert_vec_close(
        "pair se",
        pairs.std_error.as_slice().unwrap(),
        &gold.pair_std_error,
        2e-5,
    );
    assert_vec_close(
        "pair z",
        pairs.statistic_values.as_slice().unwrap(),
        &gold.pair_z,
        2e-5,
    );
    assert_vec_close(
        "pair tukey p",
        pairs.p_adjust.as_slice().unwrap(),
        &gold.pair_p_tukey,
        1e-5,
    );
    Ok(())
}

#[test]
fn emmeans_reference_grid_equal_weights_nuisance_factor() -> Result<(), Box<dyn std::error::Error>>
{
    let mut y = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for g in 0..8 {
        for (ai, av) in ["a1", "a2"].iter().enumerate() {
            for (bi, bv) in ["b1", "b2"].iter().enumerate() {
                let xv = (g as f64) - 2.5 + bi as f64;
                y.push(
                    10.0 + 2.0 * ai as f64
                        + 4.0 * bi as f64
                        + 3.0 * (ai * bi) as f64
                        + 0.5 * xv
                        + 0.2 * g as f64,
                );
                a.push(*av);
                b.push(*bv);
                x.push(xv);
                group.push(format!("g{g}"));
            }
        }
    }
    let df = DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("a".into(), a),
        Column::new("b".into(), b),
        Column::new("x".into(), x.clone()),
        Column::new("g".into(), group),
    ])?;
    let fit = lmer("y ~ a * b + x + (1 | g)", &df, false)?;
    let means = fit.emmeans("a", &df, 0.95, None)?;

    let x_mean = x.iter().sum::<f64>() / x.len() as f64;
    let reference = DataFrame::new(vec![
        Column::new("a".into(), ["a1", "a1", "a2", "a2"]),
        Column::new("b".into(), ["b1", "b2", "b1", "b2"]),
        Column::new("x".into(), vec![x_mean; 4]),
    ])?;
    let predicted = fit.predict(&reference)?;
    let expected = [
        (predicted[0] + predicted[1]) / 2.0,
        (predicted[2] + predicted[3]) / 2.0,
    ];
    assert_vec_close(
        "equal-weight EMM",
        means.estimate.as_slice().unwrap(),
        &expected,
        1e-10,
    );
    Ok(())
}

#[test]
fn emmeans_rejects_unknown_or_non_categorical_terms() {
    let df = load_pastes();
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true).unwrap();
    let err = fit.emmeans("batch", &df, 0.95, None).unwrap_err();
    assert!(err.to_string().contains("batch"));
    let err = fit.emmeans("cask", &df, 1.0, None).unwrap_err();
    assert!(err.to_string().contains("between 0 and 1"));
}
