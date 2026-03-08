use lme_rs::lmer;
use polars::prelude::*;

fn load_sleepstudy() -> DataFrame {
    let mut file =
        std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv not found");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("Failed to read CSV")
}

#[test]
fn test_with_intercept_has_k2() {
    let df = load_sleepstudy();
    // (Days | Subject) → default intercept included → k = 2 (intercept + slope)
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    let re_blocks = fit.re_blocks.as_ref().expect("re_blocks should be Some");
    assert_eq!(re_blocks.len(), 1);

    let block = &re_blocks[0];
    assert_eq!(
        block.k, 2,
        "With intercept, k should be 2 (intercept + slope)"
    );
    assert_eq!(block.effect_names[0], "(Intercept)");
    assert_eq!(block.effect_names[1], "Days");
}

#[test]
fn test_no_intercept_suppressed_k1() {
    let df = load_sleepstudy();
    // (0 + Days | Subject) → intercept suppressed → k = 1 (slope only)
    let fit = lmer("Reaction ~ Days + (0 + Days | Subject)", &df, true).unwrap();

    let re_blocks = fit.re_blocks.as_ref().expect("re_blocks should be Some");
    assert_eq!(re_blocks.len(), 1);

    let block = &re_blocks[0];
    assert_eq!(
        block.k, 1,
        "With suppressed intercept, k should be 1 (slope only)"
    );
    assert!(
        !block.effect_names.contains(&"(Intercept)".to_string()),
        "effect_names should NOT contain (Intercept) when suppressed"
    );
    assert_eq!(block.effect_names[0], "Days");
}

#[test]
fn test_no_intercept_theta_length() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (0 + Days | Subject)", &df, true).unwrap();
    let theta = fit.theta.as_ref().unwrap();

    // k=1 → theta_len = k*(k+1)/2 = 1
    assert_eq!(
        theta.len(),
        1,
        "theta should have length 1 for slope-only RE"
    );
}
