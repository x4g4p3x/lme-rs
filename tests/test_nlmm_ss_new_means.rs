//! Synthetic `SSfpl` / `SSbiexp` / `SSweibull` nlmer smoke tests.

use lme_rs::nlmer;
use lme_rs::nlmm::{ssbiexp_eval, ssfpl_eval, ssweibull_eval, NlmmStart};
use polars::prelude::*;

fn fpl_df() -> DataFrame {
    let n = 36usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) * 0.4;
        let gi = if i < n / 2 { "A" } else { "B" };
        let (mu, _) = ssfpl_eval(10.0, 50.0, 6.0, 2.0, xi);
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

fn biexp_df() -> DataFrame {
    let n = 40usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) * 0.25;
        let gi = if i < n / 2 { "A" } else { "B" };
        let (mu, _) = ssbiexp_eval(5.0, 0.5_f64.ln(), 3.0, 0.1_f64.ln(), xi);
        x.push(xi);
        y.push(mu + if gi == "A" { 0.05 } else { -0.04 });
        g.push(gi.to_string());
    }
    DataFrame::new(vec![
        Column::new("y".into(), &y),
        Column::new("x".into(), &x),
        Column::new("g".into(), &g),
    ])
    .unwrap()
}

fn weibull_df() -> DataFrame {
    let n = 40usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) * 0.3;
        let gi = if i < n / 2 { "A" } else { "B" };
        let (mu, _) = ssweibull_eval(100.0, 80.0, -1.0, 1.5, xi);
        x.push(xi);
        y.push(mu + if gi == "A" { 0.5 } else { -0.4 });
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
fn ssfpl_nlmer_runs() {
    let df = fpl_df();
    let mut start = NlmmStart::new();
    start.insert("A".into(), 12.0);
    start.insert("B".into(), 48.0);
    start.insert("xmid".into(), 5.5);
    start.insert("scal".into(), 2.2);
    let fit = nlmer("y ~ SSfpl(x, A, B, xmid, scal) ~ A|g", &df, start, false).unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 4);
}

#[test]
fn ssbiexp_nlmer_runs() {
    let df = biexp_df();
    let mut start = NlmmStart::new();
    start.insert("A1".into(), 4.5);
    start.insert("lrc1".into(), 0.4_f64.ln());
    start.insert("A2".into(), 3.2);
    start.insert("lrc2".into(), 0.12_f64.ln());
    let fit = nlmer(
        "y ~ SSbiexp(x, A1, lrc1, A2, lrc2) ~ A1|g",
        &df,
        start,
        false,
    )
    .unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 4);
}

#[test]
fn ssweibull_nlmer_runs() {
    let df = weibull_df();
    let mut start = NlmmStart::new();
    start.insert("Asym".into(), 95.0);
    start.insert("Drop".into(), 75.0);
    start.insert("lrc".into(), -0.8);
    start.insert("pwr".into(), 1.4);
    let fit = nlmer(
        "y ~ SSweibull(x, Asym, Drop, lrc, pwr) ~ Asym|g",
        &df,
        start,
        false,
    )
    .unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 4);
}

#[test]
fn ssfpl_nlmer_self_start() {
    let df = fpl_df();
    let fit = nlmer(
        "y ~ SSfpl(x, A, B, xmid, scal) ~ A|g",
        &df,
        NlmmStart::new(),
        false,
    )
    .unwrap();
    assert!(fit.deviance.unwrap().is_finite());
}
