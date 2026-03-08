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
fn test_conditional_differs_from_population_for_known_group() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Subject "308" is in the training data — conditional predictions should differ
    let new_days = Series::new("Days".into(), &[0.0, 1.0, 5.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();

    let pop_preds = fit.predict(&newdata).unwrap();
    let cond_preds = fit.predict_conditional(&newdata, true).unwrap();

    assert_eq!(pop_preds.len(), cond_preds.len());

    // Subject 308 should have non-zero random effects, so conditional ≠ population
    let diff: f64 = pop_preds
        .iter()
        .zip(cond_preds.iter())
        .map(|(p, c)| (p - c).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "Conditional predictions should differ from population for a known group, but total |diff| = {:.12}",
        diff
    );

    println!("Population:  {:?}", pop_preds.to_vec());
    println!("Conditional: {:?}", cond_preds.to_vec());
    println!("Total |diff|: {:.6}", diff);
}

#[test]
fn test_conditional_equals_population_for_unknown_group() {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();

    // Subject "999" does NOT exist in training data → should fallback to population-level
    let new_days = Series::new("Days".into(), &[0.0, 5.0, 10.0]);
    let new_subject = Series::new("Subject".into(), &["999", "999", "999"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()]).unwrap();

    let pop_preds = fit.predict(&newdata).unwrap();
    let cond_preds = fit.predict_conditional(&newdata, true).unwrap();

    for i in 0..pop_preds.len() {
        assert!(
            (pop_preds[i] - cond_preds[i]).abs() < 1e-12,
            "For unknown group, conditional should equal population at obs {}: pop={}, cond={}",
            i,
            pop_preds[i],
            cond_preds[i]
        );
    }
}
