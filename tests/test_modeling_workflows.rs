//! Numerical references from project-owned data and black-box R calls.
use lme_rs::{
    AnovaType, DdfMethod, FactorCoding, FactorSpec, FitControl, GridWeights, McpAdjust,
    ReferenceGrid,
};
use ndarray::{array, Array1};
use polars::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

fn reference() -> Value {
    serde_json::from_str(include_str!("data/modeling_reference.json")).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{a} != {b} (tol {tol})"
    );
}
fn values(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn frame(rows: &Value, lmm: bool) -> DataFrame {
    let rows = rows.as_array().unwrap();
    let nums = |key: &str| {
        rows.iter()
            .map(|r| r[key].as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let strs = |key: &str| {
        rows.iter()
            .map(|r| r[key].as_str().unwrap())
            .collect::<Vec<_>>()
    };
    let mut df = df!("y" => nums("y"), "x" => nums("x"), "a" => strs("a")).unwrap();
    let key = if lmm { "id" } else { "b" };
    df.with_column(Column::new(key.into(), strs(key))).unwrap();
    df
}
fn factors(lmm: bool) -> HashMap<String, FactorSpec> {
    let mut f = HashMap::from([(
        "a".into(),
        FactorSpec {
            levels: vec!["C".into(), "A".into(), "B".into()],
            coding: FactorCoding::Sum,
        },
    )]);
    if !lmm {
        f.insert(
            "b".into(),
            FactorSpec {
                levels: vec!["late".into(), "early".into()],
                coding: FactorCoding::Sum,
            },
        );
    }
    f
}

#[test]
fn ordered_sum_coding_ols_inference_matches_r() {
    let r = reference();
    let r = &r["ols"];
    let data = frame(&r["data"], false);
    let fit = lme_rs::lm_df_with_factors("y ~ a*b + x", &data, &factors(false)).unwrap();
    for (a, b) in fit.coefficients.iter().zip(values(&r["coefficients"])) {
        close(*a, b, 1e-10);
    }
    for (a, b) in fit.predict(&data).unwrap().iter().zip(values(&r["fitted"])) {
        close(*a, b, 1e-10);
    }
    let ci = fit.confint(0.95).unwrap();
    for (i, row) in r["ci"].as_array().unwrap().iter().enumerate() {
        close(ci.lower[i], row[0].as_f64().unwrap(), 1e-9);
        close(ci.upper[i], row[1].as_f64().unwrap(), 1e-9);
    }
    for (index, kind) in [AnovaType::Type1, AnovaType::Type2, AnovaType::Type3]
        .into_iter()
        .enumerate()
    {
        let a = fit.anova_typed(kind, DdfMethod::Residual).unwrap();
        close(
            a.residual_df.unwrap(),
            r["residual_df"].as_f64().unwrap(),
            1e-12,
        );
        close(
            a.residual_sum_sq.unwrap(),
            r["residual_ss"].as_f64().unwrap(),
            1e-12,
        );
        for row in r["anova"][index]["rows"].as_array().unwrap() {
            let i = a
                .terms
                .iter()
                .position(|n| n == row["term"].as_str().unwrap())
                .unwrap();
            close(a.f_value[i], row["f"].as_f64().unwrap(), 1e-9);
            close(
                a.sum_sq.as_ref().unwrap()[i],
                row["ss"].as_f64().unwrap(),
                1e-9,
            );
            close(a.p_value[i], row["p"].as_f64().unwrap(), 1e-10);
            close(a.num_df[i], row["df"].as_f64().unwrap(), 1e-12);
        }
    }
    let x = fit.model_spec().prediction_matrix(&data).unwrap().0;
    let unit = array![[0., 1., 0., 0., 0., 0., 0.]];
    let redundant = array![[0., 1., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0., 0.]];
    close(
        fit.test_contrast(&unit, DdfMethod::Residual)
            .unwrap()
            .f_value,
        fit.test_contrast(&redundant, DdfMethod::Residual)
            .unwrap()
            .f_value,
        1e-12,
    );
    assert_eq!(x.ncols(), 7);
    let mut bad = factors(false);
    bad.get_mut("a").unwrap().levels.push("unknown".into());
    assert!(lme_rs::lm_df_with_factors("y ~ a*b + x", &data, &bad).is_err());
    let unseen = df!("a" => ["unknown"], "b" => ["late"], "x" => [1.]).unwrap();
    assert!(fit.predict(&unseen).is_err());
}

#[test]
fn adjusted_means_simple_comparisons_and_weights_match_r() {
    let r = reference();
    let r = &r["ols"];
    let data = frame(&r["data"], false);
    let fit = lme_rs::lm_df_with_factors("y ~ a*b + x", &data, &factors(false)).unwrap();
    let mut grid = ReferenceGrid {
        terms: vec!["a".into()],
        by: vec!["b".into()],
        at: [("x".into(), 0.75)].into(),
        ..Default::default()
    };
    let means = fit
        .emmeans_with_grid(&data, &grid, 0.95, Some(DdfMethod::Residual))
        .unwrap();
    for (i, cell) in means.cells.iter().enumerate() {
        let row = r["means"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["a"] == cell["a"] && row["b"] == cell["b"])
            .unwrap();
        close(means.estimate[i], row["emmean"].as_f64().unwrap(), 1e-10);
        close(means.std_error[i], row["SE"].as_f64().unwrap(), 1e-10);
        close(means.lower[i], row["lower.CL"].as_f64().unwrap(), 1e-9);
    }
    let pairs = fit
        .emmeans_pairs_with_grid(&data, &grid, McpAdjust::Holm, Some(DdfMethod::Residual))
        .unwrap();
    let expected = r["pairs"].as_array().unwrap();
    assert_eq!(pairs.comparisons.len(), 6);
    for i in 0..6 {
        // Grid and R both preserve declared by/target order; pairs enumerate i then j.
        close(
            pairs.estimate[i],
            expected[i]["estimate"].as_f64().unwrap(),
            1e-10,
        );
        close(
            pairs.p_adjust[i],
            expected[i]["p.value"].as_f64().unwrap(),
            1e-9,
        );
        assert_eq!(pairs.groups[i]["b"], expected[i]["b"].as_str().unwrap());
    }
    grid.by.clear();
    for (policy, key) in [
        (GridWeights::Equal, "equal"),
        (GridWeights::Proportional, "proportional"),
    ] {
        grid.weights = policy;
        let m = fit
            .emmeans_with_grid(&data, &grid, 0.95, Some(DdfMethod::Residual))
            .unwrap();
        for (i, row) in r[key].as_array().unwrap().iter().enumerate() {
            close(m.estimate[i], row["emmean"].as_f64().unwrap(), 1e-10);
        }
    }
    grid.terms.push("b".into());
    assert_eq!(
        fit.emmeans_with_grid(&data, &grid, 0.95, Some(DdfMethod::Residual))
            .unwrap()
            .cells
            .len(),
        6
    );
    grid.at.insert("missing".into(), 0.0);
    assert!(fit.emmeans_with_grid(&data, &grid, 0.95, None).is_err());
}

#[test]
fn incomplete_subject_lmm_and_null_refits_match_r() {
    let r = reference();
    let r = &r["lmm"];
    let data = frame(&r["data"], true);
    let full =
        lme_rs::prepare_lmer_with_factors("y ~ a*x + (1|id)", &data, None, &factors(true)).unwrap();
    let null = lme_rs::prepare_lmer_with_factors("y ~ a + x + (1|id)", &data, None, &factors(true))
        .unwrap();
    let control = FitControl {
        require_convergence: true,
        ..Default::default()
    };
    let fit = full.fit(None, false, &control).unwrap();
    close(
        fit.deviance.unwrap(),
        r["full_deviance"].as_f64().unwrap(),
        1e-6,
    );
    close(
        null.fit(None, false, &control).unwrap().deviance.unwrap(),
        r["null_deviance"].as_f64().unwrap(),
        1e-6,
    );
    for (a, b) in fit.predict(&data).unwrap().iter().zip(values(&r["fitted"])) {
        close(*a, b, 1e-5);
    }
    let mut reml = full.fit(None, true, &control).unwrap();
    reml.with_satterthwaite(&data).unwrap();
    let a = reml.anova(DdfMethod::Satterthwaite).unwrap();
    for row in r["anova"].as_array().unwrap() {
        let i = a
            .terms
            .iter()
            .position(|n| n == row["term"].as_str().unwrap())
            .unwrap();
        close(a.f_value[i], row["f"].as_f64().unwrap(), 1e-4);
        close(a.den_df[i], row["den_df"].as_f64().unwrap(), 2e-3);
    }
    for row in r["refits"].as_array().unwrap() {
        let y = Array1::from(values(&row["y"]));
        close(
            full.fit(Some(y.clone()), false, &control)
                .unwrap()
                .deviance
                .unwrap(),
            row["full_deviance"].as_f64().unwrap(),
            1e-6,
        );
        close(
            null.fit(Some(y), false, &control)
                .unwrap()
                .deviance
                .unwrap(),
            row["null_deviance"].as_f64().unwrap(),
            1e-6,
        );
    }
    let one = lme_rs::bootstrap_lrt(&full, &null, 7, 127, 1, &control).unwrap();
    let two = lme_rs::bootstrap_lrt(&full, &null, 7, 127, 2, &control).unwrap();
    assert_eq!(one.statistics, two.statistics);
    assert_eq!(one.valid, 7, "{:?}", one.errors);
    assert_eq!(one.p_value, Some((1 + one.exceedances) as f64 / 8.0));
    assert!(lme_rs::bootstrap_lrt(&null, &full, 2, 0, 1, &control).is_err());
    assert!(fit
        .test_contrast(&array![[1., 0., 0., 0., 0., 0.]], DdfMethod::Residual)
        .is_err());
}

#[test]
fn classical_contrast_rank_is_invariant_to_row_units() {
    let r = reference();
    let data = frame(&r["ols"]["data"], false);
    let fit = lme_rs::lm_df_with_factors("y ~ a*b + x", &data, &factors(false)).unwrap();
    let base = array![[0., 1., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0.]];
    let expected = fit.test_contrast(&base, DdfMethod::Residual).unwrap();
    for scale in [1e-100, -1e-12, 1e100] {
        let mut l = base.clone();
        l.row_mut(1).mapv_inplace(|v| v * scale);
        let result = fit.test_contrast(&l, DdfMethod::Residual).unwrap();
        assert_eq!(result.num_df, 2.0);
        close(result.f_value, expected.f_value, 1e-12);
        close(result.p_value, expected.p_value, 1e-12);
    }
}
