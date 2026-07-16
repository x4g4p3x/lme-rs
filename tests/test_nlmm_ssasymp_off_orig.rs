//! Smoke tests for `SSasympOff` / `SSasympOrig`.

use lme_rs::nlmer;
use lme_rs::nlmm::{ssasympoff_eval, ssasymporig_eval, NlmmStart};
use polars::prelude::*;

fn asympoff_df() -> DataFrame {
    let n = 40usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) * 0.25;
        let gi = if i < n / 2 { "A" } else { "B" };
        let (mu, _) = ssasympoff_eval(90.0, 0.4_f64.ln(), 0.5, xi);
        x.push(xi);
        y.push(mu + if gi == "A" { 0.4 } else { -0.3 });
        g.push(gi.to_string());
    }
    DataFrame::new(vec![
        Column::new("y".into(), &y),
        Column::new("x".into(), &x),
        Column::new("g".into(), &g),
    ])
    .unwrap()
}

fn asymporig_df() -> DataFrame {
    let n = 40usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) * 0.25;
        let gi = if i < n / 2 { "A" } else { "B" };
        let (mu, _) = ssasymporig_eval(90.0, 0.4_f64.ln(), xi);
        x.push(xi);
        y.push(mu + if gi == "A" { 0.4 } else { -0.3 });
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
fn ssasympoff_nlmer_runs() {
    let df = asympoff_df();
    let mut start = NlmmStart::new();
    start.insert("Asym".into(), 88.0);
    start.insert("lrc".into(), 0.35_f64.ln());
    start.insert("c0".into(), 0.4);
    let fit = nlmer(
        "y ~ SSasympOff(x, Asym, lrc, c0) ~ Asym|g",
        &df,
        start,
        false,
    )
    .unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 3);
}

#[test]
fn ssasymporig_nlmer_runs() {
    let df = asymporig_df();
    let mut start = NlmmStart::new();
    start.insert("Asym".into(), 88.0);
    start.insert("lrc".into(), 0.35_f64.ln());
    let fit = nlmer("y ~ SSasympOrig(x, Asym, lrc) ~ Asym|g", &df, start, false).unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 2);
}
