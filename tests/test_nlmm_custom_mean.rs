//! Custom nonlinear mean via [`nlmer_with_mean`].

use lme_rs::nlmm::{nlmer_with_mean, CustomNlmmMean, NlmerFormula, NlmerOptions, NlmmStart};
use polars::prelude::*;

#[test]
fn custom_exponential_mean_runs() {
    let n = 30usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64 * 0.2;
        x.push(xi);
        y.push(3.0 * (-0.5 * xi).exp() + 0.1);
        g.push(if i < 15 { "1" } else { "2" }.to_string());
    }
    let df = DataFrame::new(vec![
        Column::new("y".into(), &y),
        Column::new("x".into(), &x),
        Column::new("g".into(), &g),
    ])
    .unwrap();

    let mean = CustomNlmmMean::new(2, |x, p| {
        let mu = p[0] * (-p[1] * x).exp();
        let da = (-p[1] * x).exp();
        let db = -x * p[0] * (-p[1] * x).exp();
        (mu, vec![da, db])
    })
    .into_arc();

    let parsed = NlmerFormula {
        response: "y".into(),
        covariate: "x".into(),
        fixed_param_names: vec!["a".into(), "b".into()],
        re_params: vec!["a".into()],
        re_group: "g".into(),
    };
    let mut start = NlmmStart::new();
    start.insert("a".into(), 2.0);
    start.insert("b".into(), 0.4);
    let opts = NlmerOptions {
        start,
        ..NlmerOptions::default()
    };
    let fit = nlmer_with_mean(&parsed, mean, &df, None, &opts).unwrap();
    assert!(fit.deviance.unwrap().is_finite());
    let pred = fit.predict(&df).unwrap();
    assert_eq!(pred.len(), n);
}
