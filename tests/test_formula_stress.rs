//! Systematic and adversarial coverage for Wilkinson formula parsing (`lme_rs::formula::parse`)
//! and the hand-off to design-matrix construction (`build_design_matrices`).
//!
//! These tests document current behavior: some inputs are rejected at parse time, others parse
//! but fail later with a missing-column style error when data do not match.

use lme_rs::formula::parse;
use lme_rs::model_matrix::build_design_matrices;
use lme_rs::LmeError;
use polars::prelude::*;

fn assert_parse_ok(formula: &str) -> lme_rs::formula::FiastoModel {
    parse(formula).unwrap_or_else(|e| panic!("expected parse ok for {:?}, got {:?}", formula, e))
}

fn assert_parse_err(formula: &str) {
    let err = parse(formula).expect_err(&format!("expected parse error for {:?}", formula));
    match err {
        LmeError::NotImplemented { ref feature } => {
            assert!(
                feature.starts_with("Formula parsing error:")
                    || feature.starts_with("JSON parsing error:"),
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
    assert_parse_err("Rey ~ 1 (+ (1 |  "); // fiasto 0.2.7 reached an internal unreachable!()
}

#[test]
fn bare_rhs_tilde_parses_as_intercept_only_model() {
    // fiasto accepts `y ~` as response with implicit intercept only (no predictors).
    let ast = assert_parse_ok("y ~");
    assert!(ast.metadata.has_intercept);
    assert!(!ast.metadata.is_random_effects_model);
}

#[test]
fn empty_or_whitespace_only_strings_rejected_or_fail_parse() {
    // `lmer` treats all-whitespace as EmptyFormula before parse; `parse` itself still runs fiasto.
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
fn three_level_nested_re_grouping_fails_after_expansion() {
    // `expand_nested_re` produces `(1 | district:school:student)`; fiasto then treats `:`
    // inside the grouping side as interaction syntax and errors. Document as current limitation.
    assert_parse_err("y ~ (1 | district/school/student)");
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
fn fixed_effect_zero_plus_x_is_rejected_by_fiasto() {
    // fiasto: "zero term (0) cannot be combined with other terms" for `0 + x` on the RHS.
    assert_parse_err("y ~ 0 + x + (1 | g)");
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
fn bare_colon_interaction_parses() {
    let ast = assert_parse_ok("y ~ a:b + (1 | g)");
    assert!(ast.columns.contains_key("a") || ast.all_generated_columns.iter().any(|c| c == "a"));
    assert!(ast.columns.contains_key("b") || ast.all_generated_columns.iter().any(|c| c == "b"));
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
    assert_eq!(ast.offset.as_deref(), Some("off"));
}

#[test]
fn offset_with_nested_parens_inner_preserved() {
    let ast = assert_parse_ok("y ~ offset(log(w)) + x + (1 | g)");
    assert_eq!(ast.offset.as_deref(), Some("log(w)"));
}

#[test]
fn offset_does_not_clean_double_plus_before_tokenize() {
    assert_parse_err("y ~ x + + offset(off) + (1 | g)");
}

#[test]
fn unsupported_or_non_wilkinson_syntax_rejected() {
    assert_parse_err("y ~ ."); // dot expansion
    assert_parse_err("cbind(y1, y2) ~ x"); // multivariate
    assert_parse_err("y ~ I(x^2)"); // inline I()
    assert_parse_err("y ~ ns(x, df = 3)"); // splines
}

#[test]
fn poly_syntax_is_accepted_by_parser() {
    // Parsed into generated basis column names; not the same as R's `poly()` evaluation.
    let ast = assert_parse_ok("y ~ poly(x, 2) + (1 | g)");
    assert!(ast.columns.contains_key("y"));
    assert!(
        ast.columns.contains_key("x")
            || ast.all_generated_columns.iter().any(|c| c.contains("poly"))
    );
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
