#[test]
fn test_print_fiasto_output() {
    let result = fiasto::parse_formula("Reaction ~ 1 + Days + (1 + Days | Subject)").unwrap();
    println!("{}", result);
}

#[test]
fn test_parse_crossed_effects() {
    let result = fiasto::parse_formula("y ~ 1 + (1 | A) + (1 | B)").unwrap();
    println!("{}", result);
}
