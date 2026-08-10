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
