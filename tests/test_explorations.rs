//! Tests for the standalone native-parser explorations.
//!
//! These lock the scientific checks the `explore_*` examples print: formula AST
//! shape, a 1-D θ-grid minimum next to the MLE, and MCP p-value ordering.

use lme_rs::formula::{parse, ColumnRole};
use lme_rs::lmer;
use lme_rs::math::LmmData;
use lme_rs::mcp::{McpAdjust, McpType};
use lme_rs::model_matrix::build_design_matrices;
use polars::prelude::*;
use std::fs::File;

fn read_csv(path: &str) -> DataFrame {
    let file = File::open(path).unwrap_or_else(|err| panic!("open {path}: {err}"));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap_or_else(|err| panic!("read {path}: {err}"))
}

#[test]
fn formula_ast_catalog_parses_native_edge_cases() {
    let independent = parse("Reaction ~ Days + (Days || Subject)").unwrap();
    assert!(independent.metadata.has_intercept);
    assert!(independent.metadata.is_random_effects_model);
    let subject = independent
        .columns
        .get("Subject")
        .expect("Subject grouping column");
    assert!(
        subject.random_effects.iter().any(|effect| {
            !effect.correlated
                && effect.has_intercept
                && effect.variables.iter().map(String::as_str).eq(["Days"])
        }),
        "|| should keep intercept+slope on one uncorrelated declaration: {:?}",
        subject.random_effects
    );
    let sleep = read_csv("tests/data/sleepstudy.csv");
    let matrices = build_design_matrices(&independent, &sleep).unwrap();
    assert_eq!(
        matrices.re_blocks.len(),
        2,
        "|| should expand to two variance blocks"
    );
    assert!(matrices
        .re_blocks
        .iter()
        .all(|block| block.k == 1 && block.theta_len == 1));

    let cell_means = parse("strength ~ 0 + cask + (1 | batch)").unwrap();
    assert!(!cell_means.metadata.has_intercept);

    let nested = parse("strength ~ 1 + (1 | batch/cask)").unwrap();
    assert!(nested.columns.contains_key("batch"));
    assert!(nested.columns.contains_key("batch:cask"));
    assert!(nested
        .columns
        .get("batch:cask")
        .unwrap()
        .has_role(ColumnRole::GroupingVariable));

    let poly = parse("Reaction ~ poly(Days, 2) + (1 | Subject)").unwrap();
    assert!(poly.columns.contains_key("poly(Days, 2)"));
    assert!(poly.columns["poly(Days, 2)"].basis.is_some());

    let ns = parse("Reaction ~ ns(Days, 3) + (1 | Subject)").unwrap();
    assert!(ns.columns.contains_key("ns(Days, 3)"));

    let identity = parse("Reaction ~ Days + I(Days^2) + (1 | Subject)").unwrap();
    assert!(identity.columns.contains_key("I(Days^2)"));

    let offset = parse("y ~ log(x) + offset(log(w)) + (1 | g)").unwrap();
    assert_eq!(
        offset.offset.as_ref().map(|expr| expr.label()),
        Some("log(w)".to_string())
    );
}

#[test]
fn theta_grid_minimum_matches_intercept_only_mle() {
    let df = read_csv("tests/data/sleepstudy.csv");
    let formula = "Reaction ~ 1 + (1 | Subject)";
    let ast = parse(formula).unwrap();
    let matrices = build_design_matrices(&ast, &df).unwrap();
    let lmm = LmmData::new_weighted(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        None,
    );
    let fit = lmer(formula, &df, true).unwrap();
    let mle = fit.theta.as_ref().unwrap()[0];
    let mle_crit = fit.deviance.unwrap();

    let mut best_theta = f64::NAN;
    let mut best_crit = f64::INFINITY;
    for step in 1..=80 {
        let theta = 0.05 * f64::from(step);
        let crit = lmm.evaluate(&[theta], true).reml_crit;
        if crit < best_crit {
            best_crit = crit;
            best_theta = theta;
        }
    }
    assert!(
        (best_theta - mle).abs() < 0.08,
        "grid min θ={best_theta} vs MLE {mle}"
    );
    assert!(
        (best_crit - mle_crit).abs() < 0.5,
        "grid crit={best_crit} vs MLE {mle_crit}"
    );
}

#[test]
fn mcp_adjustment_ordering_on_pastes_cask() {
    let df = read_csv("tests/data/pastes.csv");
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true).unwrap();
    let none = fit
        .glht("cask", McpType::Tukey, McpAdjust::None, None)
        .unwrap();
    let bonf = fit
        .glht("cask", McpType::Tukey, McpAdjust::Bonferroni, None)
        .unwrap();
    let holm = fit
        .glht("cask", McpType::Tukey, McpAdjust::Holm, None)
        .unwrap();
    let tukey = fit
        .glht("cask", McpType::Tukey, McpAdjust::Tukey, None)
        .unwrap();

    assert_eq!(none.comparisons.len(), 3);
    for i in 0..none.comparisons.len() {
        assert!(bonf.p_adjust[i] + 1e-12 >= none.p_value[i]);
        assert!(holm.p_adjust[i] + 1e-12 >= none.p_value[i]);
        assert!(holm.p_adjust[i] <= bonf.p_adjust[i] + 1e-12);
        assert!(tukey.p_adjust[i].is_finite());
        assert!((0.0..=1.0).contains(&tukey.p_adjust[i]));
    }
}

#[test]
fn exploration_example_sources_exist() {
    for path in [
        "examples/explore_formula_ast.rs",
        "examples/explore_theta_grid.rs",
        "examples/explore_mcp_adjust.rs",
        "scripts/explorations/README.md",
    ] {
        assert!(
            std::path::Path::new(path).exists(),
            "missing standalone exploration {path}"
        );
    }
}
