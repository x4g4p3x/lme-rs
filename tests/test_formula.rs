use fiasto::parse_formula;

#[test]
fn test_print_fiasto_output() {
    let result = fiasto::parse_formula("Reaction ~ 1 + Days + (1 + Days | Subject)").unwrap();
    println!("{}", result);
}
