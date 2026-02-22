use fiasto::parse_formula;

#[test]
fn test_print_fiasto_output() {
    let result = parse_formula("Reaction ~ Days + (Days | Subject)").unwrap();
    println!("{}", result);
}
