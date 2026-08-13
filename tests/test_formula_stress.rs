//! Systematic and adversarial coverage for Wilkinson formula parsing (`lme_rs::formula::parse`)
//! and the hand-off to design-matrix construction (`build_design_matrices`).
//!
//! These tests document current behavior: some inputs are rejected at parse time, others parse
//! but fail later with a missing-column style error when data do not match.

use lme_rs::formula::{parse, ColumnRole};
use lme_rs::model_matrix::build_design_matrices;
use lme_rs::LmeError;
use polars::prelude::*;

fn assert_parse_ok(formula: &str) -> lme_rs::formula::FormulaModel {
    parse(formula).unwrap_or_else(|e| panic!("expected parse ok for {:?}, got {:?}", formula, e))
}

fn assert_parse_err(formula: &str) {
    let err = parse(formula).expect_err(&format!("expected parse error for {:?}", formula));
    match err {
        LmeError::NotImplemented { ref feature } => {
            assert!(
                feature.starts_with("Formula parsing error:"),
                "unexpected NotImplemented message: {}",
                feature
            );
        }
        other => panic!("expected NotImplemented from parse, got {:?}", other),
    }
}

fn assert_build_fails_missing_column(formula: &str, df: &DataFrame) {
    let ast = assert_parse_ok(formula);
    let res = build_design_matrices(&ast, df);
    assert!(
        res.is_err(),
        "expected build_design_matrices to fail for formula {:?}",
        formula
    );
    let err = res.err().expect("checked is_err");
    match err {
        LmeError::NotImplemented { feature } => {
            let lower = feature.to_lowercase();
            assert!(
                lower.contains("not found")
                    || lower.contains("missing")
                    || lower.contains("column")
                    || lower.contains("grouping"),
                "expected missing-column style error, got: {}",
                feature
            );
        }
        e => panic!("unexpected error type: {:?}", e),
    }
}

#[test]
fn malformed_and_incomplete_formulas_rejected() {
    assert_parse_err("y"); // no tilde
    assert_parse_err("~ y"); // no response
    assert_parse_err("y ~ ~ x"); // double tilde
    assert_parse_err("y ~ x + * z");
    assert_parse_err("y ~ (1 | )"); // empty grouping
    assert_parse_err("y ~ ( | g)");
    assert_parse_err("y ~ x + (1 | g"); // unclosed paren
    assert_parse_err("y ~ )("); // junk parens (must not panic in expand_nested_re)
    assert_parse_err("Rey ~ 1 (+ (1 |"); // Regression: the former dependency panicked here.
}

#[test]
fn bare_rhs_tilde_parses_as_intercept_only_model() {
    // An empty RHS is an implicit intercept-only model.
    let ast = assert_parse_ok("y ~");
    assert!(ast.metadata.has_intercept);
    assert!(!ast.metadata.is_random_effects_model);
}

#[test]
fn empty_or_whitespace_only_strings_rejected_or_fail_parse() {
    // `lmer` treats all-whitespace as EmptyFormula before the formula parser runs.
    assert_parse_err("");
    assert_parse_err("   ");
    assert_parse_err("\t\n  \t");
}

#[test]
fn whitespace_variants_parse_like_canonical() {
    let base = assert_parse_ok("y ~ x + (1 | g)");
    let spaced = assert_parse_ok("  y   ~   x   +   (  1  |  g  )  ");
    assert_eq!(
        base.metadata.has_intercept, spaced.metadata.has_intercept,
        "intercept flag should match for whitespace variant"
    );
    assert_eq!(
        base.metadata.is_random_effects_model,
        spaced.metadata.is_random_effects_model
    );

    let tabbed = assert_parse_ok("y~\tx+\n(1|\tg)");
    assert!(tabbed.metadata.is_random_effects_model);
}

#[test]
fn repeated_fixed_terms_still_parse() {
    let ast = assert_parse_ok("y ~ x + x + x");
    assert!(ast.columns.contains_key("y"));
    assert!(ast.columns.contains_key("x"));
}

#[test]
fn nested_random_effect_expansion_end_to_end() {
    let ast = assert_parse_ok("y ~ x + (1 | school/student)");
    // Original string preserved on the model for downstream RE suppression heuristics.
    assert_eq!(ast.formula, "y ~ x + (1 | school/student)");
    assert!(ast.metadata.is_random_effects_model);
    assert!(
        ast.all_generated_columns
            .iter()
            .any(|c| c.contains("school")),
        "columns: {:?}",
        ast.all_generated_columns
    );
}

#[test]
fn three_level_nested_re_grouping_expands_end_to_end() {
    let ast = assert_parse_ok("y ~ (1 | district/school/student)");
    assert!(ast.columns.contains_key("district"));
    assert!(ast.columns.contains_key("district:school"));
    assert!(ast.columns.contains_key("district:school:student"));
}

#[test]
fn crossed_random_intercepts_parse() {
    let ast = assert_parse_ok("y ~ 1 + (1 | A) + (1 | B)");
    assert!(ast.metadata.is_random_effects_model);
    assert!(ast.columns.contains_key("A"));
    assert!(ast.columns.contains_key("B"));
}

#[test]
fn crossed_without_spaces_parse() {
    let ast = assert_parse_ok("y~x+(1|A)+(1|B)");
    assert!(ast.columns.contains_key("A"));
    assert!(ast.columns.contains_key("B"));
}

#[test]
fn independent_random_slopes_double_bar_expansion() {
    let ast = assert_parse_ok("y ~ x + (x || school)");
    assert!(ast.metadata.is_random_effects_model);
    assert!(ast.columns.contains_key("school"));
}

#[test]
fn covariance_syntax_is_preserved_in_design_blocks() {
    let data = df!(
        "y" => [1.0, 2.0, 3.0, 4.0],
        "x" => [0.5, 1.0, 1.5, 2.0],
        "g" => ["a", "a", "b", "b"],
    )
    .unwrap();

    let correlated = build_design_matrices(&assert_parse_ok("y ~ x + (x | g)"), &data).unwrap();
    assert_eq!(correlated.re_blocks.len(), 1);
    assert_eq!(correlated.re_blocks[0].k, 2);
    assert_eq!(correlated.re_blocks[0].theta_len, 3);
    assert_eq!(correlated.re_blocks[0].effect_names, ["(Intercept)", "x"]);

    let independent = build_design_matrices(&assert_parse_ok("y ~ x + (x || g)"), &data).unwrap();
    assert_eq!(independent.re_blocks.len(), 2);
    assert!(independent
        .re_blocks
        .iter()
        .all(|block| block.k == 1 && block.theta_len == 1));
    assert_eq!(independent.re_blocks[0].effect_names, ["(Intercept)"]);
    assert_eq!(independent.re_blocks[1].effect_names, ["x"]);

    let split =
        build_design_matrices(&assert_parse_ok("y ~ x + (1 | g) + (0 + x | g)"), &data).unwrap();
    assert_eq!(split.re_blocks.len(), 2);
    assert_eq!(split.re_blocks[0].effect_names, ["(Intercept)"]);
    assert_eq!(split.re_blocks[1].effect_names, ["x"]);
    assert_eq!(independent.zt, split.zt);
}

#[test]
fn fixed_effect_zero_plus_x_removes_intercept() {
    let ast = assert_parse_ok("y ~ 0 + x + (1 | g)");
    assert!(!ast.metadata.has_intercept);
}

#[test]
fn default_fixed_formula_includes_intercept() {
    let ast = assert_parse_ok("y ~ x + (1 | g)");
    assert!(ast.metadata.has_intercept);
}

#[test]
fn random_intercept_suppression_zero_plus_slopes() {
    let ast = assert_parse_ok("y ~ x + (0 + x | g)");
    assert!(ast.metadata.is_random_effects_model);
    assert!(ast.columns.contains_key("g"));
}

#[test]
fn wilkinson_star_expansion_parses() {
    // Main effects + interaction
    let ast = assert_parse_ok("y ~ a * b + (1 | g)");
    assert!(ast.columns.contains_key("a"));
    assert!(ast.columns.contains_key("b"));
    assert!(ast.metadata.is_random_effects_model);
}

#[test]
fn bare_colon_interaction_parses_without_mains() {
    let ast = assert_parse_ok("y ~ a:b + (1 | g)");
    assert!(ast.columns["a:b"].has_role(ColumnRole::Interaction));
    assert!(!ast.columns["a"].has_role(ColumnRole::FixedEffect));
    assert!(!ast.columns["b"].has_role(ColumnRole::FixedEffect));
}

#[test]
fn numeric_literal_in_formula_parses() {
    let ast = assert_parse_ok("y ~ 1 + x + col2 + (1 | g)");
    assert!(ast.columns.contains_key("col2"));
}

#[test]
fn ascii_identifiers_with_underscore_parse() {
    let ast = assert_parse_ok("resp_2 ~ pred_x + weird_col + (1 | group_id)");
    assert!(ast.columns.contains_key("resp_2"));
    assert!(ast.columns.contains_key("pred_x"));
    assert!(ast.columns.contains_key("weird_col"));
}

#[test]
fn offset_extracted_and_rhs_still_parseable() {
    let ast = assert_parse_ok("y ~ x + offset(off) + (1 | g)");
    assert_eq!(ast.offset_column(), Some("off"));
}

#[test]
fn transformed_offset_parses() {
    let ast = assert_parse_ok("y ~ offset(log(w)) + x + (1 | g)");
    assert!(ast.offset_column().is_none());
    assert!(ast.offset.is_some());
}

#[test]
fn offset_does_not_clean_double_plus_before_tokenize() {
    assert_parse_err("y ~ x + + offset(off) + (1 | g)");
}

#[test]
fn unsupported_or_non_wilkinson_syntax_rejected() {
    assert_parse_err("y ~ ."); // dot expansion
    assert_parse_err("cbind(y1, y2) ~ x"); // multivariate
    assert_parse_err("y ~ ns(x, df = 3)"); // splines
    assert_parse_err("y ~ x^2"); // Wilkinson power; use I(x^2)
}

#[test]
fn poly_is_still_rejected() {
    assert_parse_err("y ~ poly(x, 2) + (1 | g)");
}

#[test]
fn formula_data_mismatch_fails_at_matrix_build() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), vec![1.0_f64, 2.0, 3.0]).into(),
        Series::new("x".into(), vec![1.0_f64, 2.0, 3.0]).into(),
        Series::new("g".into(), vec!["a", "b", "a"]).into(),
    ])
    .unwrap();

    assert_build_fails_missing_column("y ~ z + (1 | g)", &df);
    assert_build_fails_missing_column("missing ~ x + (1 | g)", &df);
    assert_build_fails_missing_column("y ~ x + (1 | h)", &df);
}

#[test]
fn weird_column_names_round_trip_when_present_in_dataframe() {
    let df = DataFrame::new(vec![
        Series::new("resp_2".into(), vec![1.0_f64, 2.0]).into(),
        Series::new("pred_x".into(), vec![0.0_f64, 1.0]).into(),
        Series::new("weird_col".into(), vec![10.0_f64, 20.0]).into(),
        Series::new("group_id".into(), vec!["g1", "g2"]).into(),
    ])
    .unwrap();

    let ast = assert_parse_ok("resp_2 ~ pred_x + weird_col + (1 | group_id)");
    let matrices = build_design_matrices(&ast, &df).expect("design matrices should build");
    assert_eq!(matrices.y.len(), 2);
}

#[test]
fn categorical_style_factor_columns_parse_and_build() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), vec![1.0_f64, 2.0, 3.0, 4.0]).into(),
        Series::new("trt".into(), vec!["A", "A", "B", "B"]).into(),
        Series::new("blk".into(), vec!["b1", "b2", "b1", "b2"]).into(),
    ])
    .unwrap();

    let ast = assert_parse_ok("y ~ trt * blk + (1 | blk)");
    let res = build_design_matrices(&ast, &df);
    assert!(
        res.is_ok(),
        "categorical interaction formula should build, got {:?}",
        res.err()
    );
}

#[test]
fn long_formula_many_terms_still_parses() {
    let mut rhs = String::new();
    for i in 0..40 {
        if i > 0 {
            rhs.push_str(" + ");
        }
        rhs.push_str(&format!("x{}", i));
    }
    let formula = format!("y ~ {}", rhs);
    let ast = assert_parse_ok(&formula);
    assert!(ast.columns.contains_key("y"));
    assert!(ast.columns.contains_key("x0"));
    assert!(ast.columns.contains_key("x39"));
}

#[test]
fn intercept_forms_match_wilkinson_semantics() {
    for formula in ["y ~ 0 + x", "y ~ -1 + x", "y ~ x - 1", "y ~ 1 - 1 + x"] {
        let ast = assert_parse_ok(formula);
        assert!(!ast.metadata.has_intercept, "{formula}");
        assert!(!ast
            .all_generated_columns
            .iter()
            .any(|name| name == "intercept"));
    }
    for formula in ["y ~ x", "y ~ 1 + x", "y ~ 0 + 1 + x"] {
        let ast = assert_parse_ok(formula);
        assert!(ast.metadata.has_intercept, "{formula}");
        assert_eq!(ast.all_generated_columns[1], "intercept");
    }
}

#[test]
fn typed_roles_and_formula_order_are_exact_and_deduplicated() {
    let ast = assert_parse_ok("y ~ x + x + a * b + (x | g)");
    assert_eq!(
        ast.all_generated_columns,
        ["y", "intercept", "x", "a", "b", "a_b", "g"]
    );
    assert!(ast.columns["y"].has_role(ColumnRole::Response));
    assert!(ast.columns["x"].has_role(ColumnRole::FixedEffect));
    assert!(ast.columns["x"].has_role(ColumnRole::RandomEffect));
    assert!(ast.columns["a_b"].has_role(ColumnRole::Interaction));
    assert!(ast.columns["g"].has_role(ColumnRole::GroupingVariable));
}

#[test]
fn random_effect_metadata_distinguishes_correlation_and_intercepts() {
    let correlated = assert_parse_ok("y ~ x + (x | g)");
    let effect = &correlated.columns["g"].random_effects[0];
    assert!(effect.correlated);
    assert!(effect.has_intercept);
    assert_eq!(effect.variables.as_slice(), &["x".to_string()]);

    let independent = assert_parse_ok("y ~ x + (0 + x || g)");
    let effect = &independent.columns["g"].random_effects[0];
    assert!(!effect.correlated);
    assert!(!effect.has_intercept);

    let split = assert_parse_ok("y ~ x + (1 | g) + (0 + x | g)");
    let effects = &split.columns["g"].random_effects;
    assert_eq!(effects.len(), 2);
    assert!(effects[0].has_intercept);
    assert!(effects[0].variables.is_empty());
    assert!(!effects[1].has_intercept);
    assert_eq!(effects[1].variables.as_slice(), &["x".to_string()]);
}

#[test]
fn deep_nesting_preserves_group_order() {
    let ast = assert_parse_ok("y ~ x + (1 | district/school/class/student)");
    assert_eq!(
        ast.all_generated_columns,
        [
            "y",
            "intercept",
            "x",
            "district",
            "district:school",
            "district:school:class",
            "district:school:class:student",
        ]
    );
}

#[test]
fn parser_is_deterministic() {
    let formula = "y ~ 0 + a * b + offset(exposure) + (0 + x || g) + (1 | batch/cask)";
    let expected = assert_parse_ok(formula);
    for _ in 0..100 {
        assert_eq!(assert_parse_ok(formula), expected);
    }
}

#[test]
fn malformed_corpus_is_rejected_without_panics() {
    for formula in [
        "y ~ + x",
        "y ~ x +",
        "y ~ x -",
        "y ~ x + + z",
        "y ~ x - -1",
        "y ~ 2",
        "y ~ a * b * c",
        "y ~ (1)",
        "y ~ ((1 | g))",
        "y ~ (0 | g)",
        "y ~ (1 ||)",
        "y ~ (1 | g | h)",
        "y ~ (1 | g/)",
        "y ~ (1 | g/:h)",
        "y ~ offset()",
        "y ~ offset(x, z)",
        "y ~ offset(x) + offset(z)",
        "y ~ unknown(x)",
        "bind(y, z) ~ x",
    ] {
        assert_parse_err(formula);
    }
}

#[test]
fn thousand_term_formula_scales_and_deduplicates() {
    let rhs = (0..1_000)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>()
        .join(" + ");
    let ast = assert_parse_ok(&format!("y ~ {rhs} + x999"));
    assert_eq!(ast.columns.len(), 1_001);
    assert_eq!(ast.all_generated_columns.len(), 1_002);
    assert_eq!(
        ast.all_generated_columns.last().map(String::as_str),
        Some("x999")
    );
}

#[test]
fn numeric_star_interaction_materializes_expected_product() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0]).into(),
        Series::new("a".into(), [2.0, 3.0]).into(),
        Series::new("b".into(), [5.0, 7.0]).into(),
    ])
    .unwrap();
    let matrices = build_design_matrices(&assert_parse_ok("y ~ a * b"), &df).unwrap();
    assert_eq!(matrices.x.shape(), &[2, 4]);
    assert_eq!(matrices.x.row(0).to_vec(), [1.0, 2.0, 5.0, 10.0]);
    assert_eq!(matrices.x.row(1).to_vec(), [1.0, 3.0, 7.0, 21.0]);
}

#[test]
fn numeric_colon_interaction_materializes_product_only() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0]).into(),
        Series::new("a".into(), [2.0, 3.0]).into(),
        Series::new("b".into(), [5.0, 7.0]).into(),
    ])
    .unwrap();
    let matrices = build_design_matrices(&assert_parse_ok("y ~ a:b"), &df).unwrap();
    assert_eq!(matrices.x.shape(), &[2, 2]);
    assert_eq!(matrices.fixed_names, ["(Intercept)", "a:b"]);
    assert_eq!(matrices.x.row(0).to_vec(), [1.0, 10.0]);
    assert_eq!(matrices.x.row(1).to_vec(), [1.0, 21.0]);
}

#[test]
fn log_sqrt_exp_and_identity_materialize() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0]).into(),
        Series::new("x".into(), [4.0, 9.0]).into(),
        Series::new("z".into(), [1.0, 2.0]).into(),
    ])
    .unwrap();
    let matrices = build_design_matrices(
        &assert_parse_ok("y ~ log(x) + sqrt(x) + exp(z) + I(x^2 + z)"),
        &df,
    )
    .unwrap();
    assert_eq!(
        matrices.fixed_names,
        ["(Intercept)", "log(x)", "sqrt(x)", "exp(z)", "I(x^2 + z)"]
    );
    let row0 = matrices.x.row(0);
    assert!((row0[1] - 4.0_f64.ln()).abs() < 1e-12);
    assert!((row0[2] - 2.0).abs() < 1e-12);
    assert!((row0[3] - 1.0_f64.exp()).abs() < 1e-12);
    assert!((row0[4] - 17.0).abs() < 1e-12);
}

#[test]
fn offset_log_matches_precomputed_column() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0]).into(),
        Series::new("x".into(), [1.0, 1.0]).into(),
        Series::new(
            "w".into(),
            [std::f64::consts::E, std::f64::consts::E.powi(2)],
        )
        .into(),
        Series::new("g".into(), ["a", "b"]).into(),
    ])
    .unwrap();
    let matrices =
        build_design_matrices(&assert_parse_ok("y ~ x + offset(log(w)) + (1 | g)"), &df).unwrap();
    let offset = matrices.offset.expect("offset");
    assert!((offset[0] - 1.0).abs() < 1e-12);
    assert!((offset[1] - 2.0).abs() < 1e-12);
}

#[test]
fn string_offset_column_errors() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0, 3.0]).into(),
        Series::new("x".into(), [1.0, 2.0, 3.0]).into(),
        Series::new("g".into(), ["A", "A", "B"]).into(),
        Series::new("off".into(), ["bad", "type", "string"]).into(),
    ])
    .unwrap();
    let res = build_design_matrices(&assert_parse_ok("y ~ x + offset(off) + (1 | g)"), &df);
    assert!(res.is_err(), "string offset must be rejected");
    let err = res.err().expect("checked is_err");
    let msg = err.to_string();
    assert!(
        msg.contains("off") && msg.contains("invalid"),
        "unexpected error: {msg}"
    );
}

#[test]
fn integer_offset_column_is_accepted() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0]).into(),
        Series::new("x".into(), [1.0, 1.0]).into(),
        Series::new("off".into(), [2i32, 3]).into(),
        Series::new("g".into(), ["a", "b"]).into(),
    ])
    .unwrap();
    let matrices =
        build_design_matrices(&assert_parse_ok("y ~ x + offset(off) + (1 | g)"), &df).unwrap();
    let offset = matrices.offset.expect("offset");
    assert!((offset[0] - 2.0).abs() < 1e-12);
    assert!((offset[1] - 3.0).abs() < 1e-12);
}

#[test]
fn categorical_colon_drops_reference_cell_with_intercept() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0, 3.0, 4.0]).into(),
        Series::new("a".into(), ["A", "A", "B", "B"]).into(),
        Series::new("b".into(), ["X", "Y", "X", "Y"]).into(),
    ])
    .unwrap();
    let with_intercept = build_design_matrices(&assert_parse_ok("y ~ a:b"), &df).unwrap();
    assert_eq!(with_intercept.x.nrows(), 4);
    assert_eq!(with_intercept.x.ncols(), 4); // intercept + 3 cells
    assert_eq!(with_intercept.x.row(0).to_vec(), [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(with_intercept.x.row(1).to_vec(), [1.0, 1.0, 0.0, 0.0]);
    assert_eq!(with_intercept.x.row(2).to_vec(), [1.0, 0.0, 1.0, 0.0]);
    assert_eq!(with_intercept.x.row(3).to_vec(), [1.0, 0.0, 0.0, 1.0]);

    let no_intercept = build_design_matrices(&assert_parse_ok("y ~ 0 + a:b"), &df).unwrap();
    assert_eq!(no_intercept.x.ncols(), 4); // all cells
    assert_eq!(no_intercept.x.row(0).to_vec(), [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(no_intercept.x.row(1).to_vec(), [0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn numeric_by_factor_colon_uses_all_levels() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0, 3.0, 4.0]).into(),
        Series::new("x".into(), [1.0, 2.0, 3.0, 4.0]).into(),
        Series::new("f".into(), ["U", "U", "V", "V"]).into(),
        Series::new("g".into(), ["a", "a", "b", "b"]).into(),
    ])
    .unwrap();
    let matrices = build_design_matrices(&assert_parse_ok("y ~ x:f"), &df).unwrap();
    assert_eq!(matrices.fixed_names, ["(Intercept)", "x:fU", "x:fV"]);
    assert_eq!(matrices.x.row(0).to_vec(), [1.0, 1.0, 0.0]);
    assert_eq!(matrices.x.row(2).to_vec(), [1.0, 0.0, 3.0]);
}

#[test]
fn lmer_fits_log_and_colon_formulas() {
    let df = DataFrame::new(vec![
        Series::new("y".into(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).into(),
        Series::new("x".into(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).into(),
        Series::new("a".into(), ["A", "A", "B", "B", "A", "A", "B", "B"]).into(),
        Series::new("b".into(), ["X", "Y", "X", "Y", "X", "Y", "X", "Y"]).into(),
        Series::new("g".into(), ["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"]).into(),
    ])
    .unwrap();
    let log_fit = lme_rs::lmer("y ~ log(x) + (1 | g)", &df, true).unwrap();
    assert_eq!(log_fit.coefficients.len(), 2);
    assert_eq!(log_fit.predict(&df).unwrap().len(), 8);
    let colon_fit = lme_rs::lmer("y ~ a:b + (1 | g)", &df, true).unwrap();
    assert_eq!(colon_fit.coefficients.len(), 4);
    assert_eq!(colon_fit.predict(&df).unwrap().len(), 8);
}

#[test]
fn unicode_identifiers_rejected_at_lex() {
    assert_parse_err("réponse ~ prédicteur + (1 | groupe)");
}

#[test]
fn backtick_quoting_not_supported_in_lexer() {
    assert_parse_err("y ~ `weird-col` + (1 | g)");
}

#[test]
fn formula_with_leading_zero_width_space_stripped_for_success() {
    let zwsp = "\u{200B}";
    let formula = format!("{}y ~ x + (1 | g)", zwsp);
    let ast = assert_parse_ok(formula.trim_start_matches(zwsp));
    assert!(ast.columns.contains_key("y"));
}
