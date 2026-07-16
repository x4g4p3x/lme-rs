use lme_rs::nlmm::{sspower_eval, NlmmStart};
use lme_rs::{nlmer_with_options, NlmerOptions};
use polars::prelude::*;

fn power_df() -> DataFrame {
    let a = 2.0;
    let b = 0.5;
    let c = 1.0;
    let n = 50usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = 0.5 + (i as f64) * 0.2;
        let gi = if i < n / 2 { "A" } else { "B" };
        let offset = if gi == "A" { 0.15 } else { -0.1 };
        let (mu, _, _, _) = sspower_eval(a, b, c + offset, xi);
        x.push(xi);
        y.push(mu + 0.05 * (i as f64 - 25.0));
        g.push(gi.to_string());
    }
    DataFrame::new(vec![
        Column::new("y".into(), &y),
        Column::new("x".into(), &x),
        Column::new("g".into(), &g),
    ])
    .unwrap()
}

#[test]
fn sspower_respects_population_box_bounds() {
    let df = power_df();

    let mut start = NlmmStart::new();
    start.insert("a".to_string(), 1.0);
    start.insert("b".to_string(), 0.5);
    start.insert("c".to_string(), 0.0);

    let mut lower = NlmmStart::new();
    lower.insert("a".to_string(), 0.5);
    lower.insert("b".to_string(), 0.1);
    lower.insert("c".to_string(), 0.5);

    let mut upper = NlmmStart::new();
    upper.insert("a".to_string(), 3.0);
    upper.insert("b".to_string(), 1.5);
    upper.insert("c".to_string(), 1.5);

    let opts = NlmerOptions {
        reml: false,
        start,
        lower: Some(lower),
        upper: Some(upper),
        ..NlmerOptions::default()
    };

    let fit = nlmer_with_options("y ~ SSpower(x, a, b, c) ~ c|g", &df, &opts).unwrap();
    let names = fit.fixed_names.as_ref().unwrap();
    for (i, name) in names.iter().enumerate() {
        let v = fit.coefficients[i];
        match name.as_str() {
            "a" => assert!((0.5..=3.0).contains(&v), "a={v}"),
            "b" => assert!((0.1..=1.5).contains(&v), "b={v}"),
            "c" => assert!((0.5..=1.5).contains(&v), "c={v}"),
            _ => {}
        }
    }
}

#[test]
fn rejects_inverted_bounds() {
    let df = power_df();

    let mut lower = NlmmStart::new();
    lower.insert("a".to_string(), 5.0);
    let mut upper = NlmmStart::new();
    upper.insert("a".to_string(), 1.0);

    let opts = NlmerOptions {
        lower: Some(lower),
        upper: Some(upper),
        ..NlmerOptions::default()
    };
    let err = nlmer_with_options("y ~ SSpower(x, a, b, c) ~ c|g", &df, &opts).unwrap_err();
    assert!(format!("{err}").contains("bounds"));
}

#[test]
fn sspower_respects_group_level_box_bounds() {
    let df = power_df();

    let mut start = NlmmStart::new();
    start.insert("a".to_string(), 1.0);
    start.insert("b".to_string(), 0.5);
    start.insert("c".to_string(), 1.0);

    // Tight bounds on group-level c = β_c + b_g around the DGP (~0.9–1.15).
    let mut group_lower = NlmmStart::new();
    group_lower.insert("c".to_string(), 0.85);
    let mut group_upper = NlmmStart::new();
    group_upper.insert("c".to_string(), 1.25);

    let opts = NlmerOptions {
        reml: false,
        start,
        group_lower: Some(group_lower),
        group_upper: Some(group_upper),
        ..NlmerOptions::default()
    };

    let fit = nlmer_with_options("y ~ SSpower(x, a, b, c) ~ c|g", &df, &opts).unwrap();
    let c_idx = fit
        .fixed_names
        .as_ref()
        .unwrap()
        .iter()
        .position(|n| n == "c")
        .unwrap();
    let beta_c = fit.coefficients[c_idx];
    let ranef = fit.ranef.as_ref().expect("ranef");
    let terms = ranef.column("term").unwrap();
    let vals = ranef.column("condval").unwrap().f64().unwrap();
    for row in 0..ranef.height() {
        let term = terms
            .get(row)
            .unwrap()
            .to_string()
            .trim_matches('"')
            .to_string();
        if term != "c" {
            continue;
        }
        let b = vals.get(row).unwrap();
        let phi = beta_c + b;
        assert!(
            (0.85..=1.25).contains(&phi),
            "group-level c out of bounds: {phi}"
        );
    }
}

#[test]
fn rejects_group_bound_on_fixed_only_param() {
    let df = power_df();
    let mut group_lower = NlmmStart::new();
    group_lower.insert("a".to_string(), 0.0);
    let opts = NlmerOptions {
        group_lower: Some(group_lower),
        ..NlmerOptions::default()
    };
    let err = nlmer_with_options("y ~ SSpower(x, a, b, c) ~ c|g", &df, &opts).unwrap_err();
    assert!(format!("{err}").contains("random-effect"));
}
