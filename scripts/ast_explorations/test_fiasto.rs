use fiasto::parse_formula;
fn main() {
    let raw = "(1 | Subject) + (0 + Days | Subject)";
    let parsed = parse_formula(raw);
    println!("{:#?}", parsed);
}
