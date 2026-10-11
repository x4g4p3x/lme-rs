#[test]
fn orthogonal_polynomial_fits_preserve_predictor_units() {
    use polars::prelude::*;
    let x = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5];
    let data = df!("y" => [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1], "x" => x).unwrap();
    let baseline = lme_rs::lm_df("y ~ poly(x, 4)", &data).unwrap();
    let new_x = [-4., 0.25, 5.];
    let expected = baseline.predict(&df!("x" => new_x).unwrap()).unwrap();
    for scale in [1e-100, 1e100] {
        let mut rescaled = data.clone();
        rescaled
            .with_column(Column::new("x".into(), x.map(|v| v * scale)))
            .unwrap();
        let fit = lme_rs::lm_df("y ~ poly(x, 4)", &rescaled).unwrap();
        assert!(
            (&fit.fitted - &baseline.fitted)
                .iter()
                .all(|v| v.abs() < 1e-10),
            "scale {scale}"
        );
        let got = fit
            .predict(&df!("x" => new_x.map(|v| v * scale)).unwrap())
            .unwrap();
        assert!(
            (&got - &expected).iter().all(|v| v.abs() < 1e-9),
            "scale {scale}: {got:?} vs {expected:?}"
        );
    }
}

#[test]
fn natural_spline_fits_preserve_predictor_units() {
    use polars::prelude::*;
    let x = [1., 2., 3., 4., 5., 6., 7., 8.];
    let data = df!("y" => [1.2, 2.1, 4.8, 4.2, 6.3, 5.7, 7.4, 8.1], "x" => x).unwrap();
    let baseline = lme_rs::lm_df("y ~ ns(x, 3)", &data).unwrap();
    let new_x = [-1., 4.25, 10.];
    let expected = baseline.predict(&df!("x" => new_x).unwrap()).unwrap();
    for scale in [1e-200, 1e200, 1e-310] {
        let mut rescaled = data.clone();
        rescaled
            .with_column(Column::new("x".into(), x.map(|v| v * scale)))
            .unwrap();
        let fit = lme_rs::lm_df("y ~ ns(x, 3)", &rescaled).unwrap();
        assert!(
            (&fit.fitted - &baseline.fitted)
                .iter()
                .all(|v| v.abs() < 1e-10),
            "scale {scale}"
        );
        let got = fit
            .predict(&df!("x" => new_x.map(|v| v * scale)).unwrap())
            .unwrap();
        assert!(
            (&got - &expected).iter().all(|v| v.abs() < 1e-9),
            "scale {scale}: {got:?} vs {expected:?}"
        );
    }
}

#[test]
fn natural_spline_with_repeated_observations_matches_truncated_power_space() {
    let x = [0.0_f64, 1., 1., 1., 2., 3., 4., 5., 6., 8., 10., 12.];
    // A natural cubic spline has linear tails. With boundary knots 0 and 12,
    // differences of these truncated cubics span its two nonlinear directions.
    let d = |x: f64, knot: f64| {
        ((x - knot).max(0.0).powi(3) - (x - 12.0).max(0.0).powi(3)) / (12.0 - knot)
    };
    let data = polars::df!("x" => x,
        "y" => [0.3, 1.2, 1.5, 0.8, 2.4, 4.1, 4.8, 4.3, 6.2, 8.3, 9.7, 13.1],
        "b1" => x.map(|x| d(x, 5.0/3.0)-d(x, 0.0)),
        "b2" => x.map(|x| d(x, 16.0/3.0)-d(x, 0.0)))
    .unwrap();
    let expected = lme_rs::lm_df("y ~ x + b1 + b2", &data).unwrap();
    let actual = lme_rs::lm_df("y ~ ns(x, 3)", &data).unwrap();
    assert!((&actual.fitted - &expected.fitted)
        .iter()
        .all(|v| v.abs() < 1e-9));
}

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
fn identity_arithmetic_respects_power_precedence() {
    use lme_rs::model_matrix::build_design_matrices;
    let xs = [0.5_f64, 1.0, 2.0, 3.0];
    let df = polars::df!("y" => [1., 2., 3., 4.], "x" => xs).unwrap();
    for (expression, expected) in [
        ("-x^2", xs.map(|x| -(x * x))),
        ("(-x)^2", xs.map(|x| x * x)),
        ("x^-2", xs.map(|x| 1.0 / (x * x))),
        ("x^-2^2", xs.map(|x| 1.0 / x.powi(4))),
        ("x^2^3", xs.map(|x| x.powi(8))),
        ("(x^2)^3", xs.map(|x| x.powi(6))),
        ("-x^2*3", xs.map(|x| -3.0 * x * x)),
        ("--x^2", xs.map(|x| x * x)),
        ("x^(-2)^2", xs.map(|x| x.powi(4))),
    ] {
        let ast = lme_rs::formula::parse(&format!("y ~ 0 + I({expression})")).unwrap();
        let matrices = build_design_matrices(&ast, &df).unwrap();
        assert_eq!(matrices.x.ncols(), 1);
        for (actual, expected) in matrices.x.column(0).iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "I({expression}): {actual} versus {expected}"
            );
        }
        // Canonical labels must remain equivalent when parsed again.
        let canonical =
            lme_rs::formula::parse(&format!("y ~ 0 + {}", matrices.fixed_names[0])).unwrap();
        let rebuilt = build_design_matrices(&canonical, &df).unwrap();
        assert_eq!(rebuilt.x, matrices.x, "I({expression}) label round trip");
    }
}

#[test]
fn nested_power_fit_and_prediction_match_explicit_covariates() {
    let xs = [
        -1.25_f64, -1., -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1., 1.25,
    ];
    let df = polars::df!(
        "y" => [4., 2., 1.2, 0.7, 0.3, 0.8, 0.9, 1.4, 2.5, 3.8],
        "x" => xs,
        "sixth" => xs.map(|x| x.powi(6)),
        "eighth" => xs.map(|x| x.powi(8)),
    )
    .unwrap();
    let arithmetic = lme_rs::lm_df("y ~ I((x^2)^3) + I(x^2^3)", &df).unwrap();
    let explicit = lme_rs::lm_df("y ~ sixth + eighth", &df).unwrap();
    assert_eq!(arithmetic.coefficients.len(), explicit.coefficients.len());
    for (actual, expected) in arithmetic.coefficients.iter().zip(&explicit.coefficients) {
        assert!((actual - expected).abs() < 1e-10);
    }
    let xs = [0.1_f64, 0.6, 1.1];
    let newdata = polars::df!(
        "x" => xs,
        "sixth" => xs.map(|x| x.powi(6)),
        "eighth" => xs.map(|x| x.powi(8)),
    )
    .unwrap();
    let actual = arithmetic.predict(&newdata).unwrap();
    let expected = explicit.predict(&newdata).unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn parenthesized_power_terms_keep_distinct_design_columns() {
    use lme_rs::model_matrix::build_design_matrices;
    let xs = [0.5_f64, 1.0, 2.0, 3.0];
    let df = polars::df!("y" => [1., 2., 3., 4.], "x" => xs).unwrap();
    let ast = lme_rs::formula::parse("y ~ 0 + I((x^2)^3) + I(x^2^3)").unwrap();
    let matrices = build_design_matrices(&ast, &df).unwrap();
    assert_eq!(matrices.fixed_names, ["I((x^2)^3)", "I(x^2^3)"]);
    for (i, x) in xs.into_iter().enumerate() {
        assert!((matrices.x[[i, 0]] - x.powi(6)).abs() < 1e-12);
        assert!((matrices.x[[i, 1]] - x.powi(8)).abs() < 1e-12);
    }
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
