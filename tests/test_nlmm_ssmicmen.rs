//! Synthetic `SSmicmen` nlmer smoke test.

use lme_rs::nlmer;
use lme_rs::nlmm::{ssmicmen_eval, NlmmStart};
use polars::prelude::*;

fn micmen_df() -> DataFrame {
    let n = 40usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64 + 1.0) * 0.5;
        let gi = if i < n / 2 { "A" } else { "B" };
        let (mu, _, _) = ssmicmen_eval(12.0, 2.0, xi);
        x.push(xi);
        y.push(mu + if gi == "A" { 0.3 } else { -0.2 });
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
fn ssmicmen_nlmer_runs() {
    let df = micmen_df();
    let mut start = NlmmStart::new();
    start.insert("Vmax".into(), 10.0);
    start.insert("K".into(), 1.5);
    let fit = nlmer("y ~ SSmicmen(x, Vmax, K) ~ Vmax|g", &df, start, false).unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    assert_eq!(fit.coefficients.len(), 2);
}
