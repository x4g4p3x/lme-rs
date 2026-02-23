use fiasto::parse_formula;
fn main() {
    let raw = "Reaction ~ Days + offset(time) + (1 | Subject)";
    let parsed = parse_formula(raw);
    println!("{:#?}", parsed);
}
