#[test]
fn test_parse_formula_model() {
    let result = lme_rs::formula::parse("Reaction ~ 1 + Days + (1 + Days | Subject)").unwrap();
    assert!(result.metadata.has_intercept);
    assert!(result.metadata.is_random_effects_model);
    assert_eq!(
        result.columns["Subject"].random_effects[0]
            .variables
            .as_slice(),
        &["Days".to_string()]
    );
}

#[test]
fn test_parse_crossed_effects() {
    let result = lme_rs::formula::parse("y ~ 1 + (1 | A) + (1 | B)").unwrap();
    assert!(result.columns.contains_key("A"));
    assert!(result.columns.contains_key("B"));
}

#[test]
fn test_poly_raw_matches_identity_powers() {
    use lme_rs::model_matrix::build_design_matrices;
    let df = polars::df!(
        "y" => &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
        "x" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "g" => &["a", "a", "b", "b", "c", "c"],
    )
    .unwrap();
    let ast = lme_rs::formula::parse("y ~ 0 + poly(x, 2, raw = TRUE)").unwrap();
    let matrices = build_design_matrices(&ast, &df).unwrap();
    assert_eq!(
        matrices.fixed_names,
        vec!["poly(x, 2, raw = TRUE)1", "poly(x, 2, raw = TRUE)2"]
    );
    for i in 0..6 {
        let x = (i + 1) as f64;
        assert!((matrices.x[[i, 0]] - x).abs() < 1e-12);
        assert!((matrices.x[[i, 1]] - x * x).abs() < 1e-12);
    }
}

#[test]
fn test_dot_expands_remaining_columns() {
    use lme_rs::model_matrix::build_design_matrices;
    let df = polars::df!(
        "y" => &[1.0, 2.0, 3.0, 4.0],
        "x" => &[0.1, 0.2, 0.3, 0.4],
        "z" => &[1.0, 1.0, 0.0, 0.0],
        "g" => &["a", "a", "b", "b"],
    )
    .unwrap();
    let ast = lme_rs::formula::parse("y ~ . + (1 | g)").unwrap();
    let matrices = build_design_matrices(&ast, &df).unwrap();
    assert!(matrices.fixed_names.iter().any(|n| n == "x"));
    assert!(matrices.fixed_names.iter().any(|n| n == "z"));
    assert!(!matrices
        .fixed_names
        .iter()
        .any(|n| n == "g" || n.starts_with("g")));
}

#[test]
fn test_orthogonal_poly_and_ns_fit_and_predict() {
    let df = polars::df!(
        "y" => &[1.0, 2.2, 2.8, 5.1, 7.4, 9.0, 12.2, 16.0],
        "x" => &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "g" => &["a", "a", "a", "b", "b", "c", "c", "c"],
    )
    .unwrap();

    let poly_fit = lme_rs::lm_df("y ~ poly(x, 2)", &df).unwrap();
    assert_eq!(poly_fit.coefficients.len(), 3);
    let poly_pred = poly_fit.predict(&df).unwrap();
    for (a, b) in poly_fit.fitted.iter().zip(poly_pred.iter()) {
        assert!((a - b).abs() < 1e-8, "poly predict {a} vs fitted {b}");
    }

    let ns_fit = lme_rs::lm_df("y ~ ns(x, 3)", &df).unwrap();
    assert_eq!(ns_fit.coefficients.len(), 4);
    let ns_pred = ns_fit.predict(&df).unwrap();
    for (a, b) in ns_fit.fitted.iter().zip(ns_pred.iter()) {
        assert!((a - b).abs() < 1e-6, "ns predict {a} vs fitted {b}");
    }

    let mixed = lme_rs::lmer("y ~ poly(x, 2) + (1 | g)", &df, true).unwrap();
    assert!(mixed.converged.unwrap_or(false));
    assert!(mixed
        .fixed_names
        .as_ref()
        .unwrap()
        .iter()
        .any(|n| n.starts_with("poly(x, 2)")));
}
