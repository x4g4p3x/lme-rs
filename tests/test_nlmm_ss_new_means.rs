//! Synthetic `SSfpl` / `SSbiexp` / `SSweibull` nlmer smoke tests.

use lme_rs::nlmer;
use lme_rs::nlmm::{ssbiexp_eval, ssfpl_eval, sslogis_eval, ssweibull_eval, NlmmStart};
use polars::prelude::*;

#[test]
fn logistic_tail_gradients_retain_representable_values() {
    for z in [-400.0_f64, 400.0] {
        // Independently, the logistic derivative is 1 / (2 + 2*cosh(z)).
        // At these tails it remains representable, even though exp(z)^2 may not.
        let slope = 1.0 / (2.0 + 2.0 * z.cosh());
        let (_, _, dxmid, dscal) = sslogis_eval(40.0, 0.0, 2.0, -2.0 * z);
        let (_, fpl) = ssfpl_eval(10.0, 50.0, 0.0, 2.0, -2.0 * z);
        // Reflection exchanges the two asymptote derivatives; the small one
        // must survive even when subtracting the large one from 1 would cancel.
        let small_asymptote = if z < 0.0 { fpl[0] } else { fpl[1] };
        assert!((small_asymptote / (-z.abs()).exp() - 1.0).abs() < 1e-12);
        let expected_xmid = -20.0 * slope;
        for (name, actual_xmid, actual_scal) in
            [("SSlogis", dxmid, dscal), ("SSfpl", fpl[2], fpl[3])]
        {
            assert!(
                (actual_xmid / expected_xmid - 1.0).abs() < 1e-12,
                "{name}, z={z}: xmid gradient {actual_xmid}, expected {expected_xmid}"
            );
            assert!((actual_scal / (-z * expected_xmid) - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn logistic_saturated_tails_have_finite_limiting_gradients() {
    for (x, expected_logis, expected_fpl, da, db) in [
        (-1000.0, 0.0, 10.0, 1.0, 0.0),
        (1000.0, 40.0, 50.0, 0.0, 1.0),
        (-f64::MAX, 0.0, 10.0, 1.0, 0.0),
        (f64::MAX, 40.0, 50.0, 0.0, 1.0),
    ] {
        for scal in [1.0, 0.5] {
            // MAX / 0.5 overflows the standardized covariate; its logistic
            // limit and zero derivatives must still be evaluated without 0*inf.
            let (mu, asym, dxmid, dscal) = sslogis_eval(40.0, 0.0, scal, x);
            assert_eq!(mu, expected_logis);
            assert_eq!(asym, db);
            assert_eq!(dxmid, 0.0);
            assert_eq!(dscal, 0.0);
            let (mu, gradient) = ssfpl_eval(10.0, 50.0, 0.0, scal, x);
            assert_eq!(mu, expected_fpl);
            assert_eq!(gradient, [da, db, 0.0, 0.0]);
        }
    }
}

#[test]
fn logistic_gradients_transform_with_predictor_units() {
    let (_, _, logis_xmid, logis_scal) = sslogis_eval(40.0, 5.0, 2.0, 4.0);
    let (fpl_mu, fpl_gradient) = ssfpl_eval(10.0, 50.0, 5.0, 2.0, 4.0);
    for units in [1e-200, 1e200] {
        let (_, _, dxmid, dscal) = sslogis_eval(40.0, 5.0 * units, 2.0 * units, 4.0 * units);
        let (mu, gradient) = ssfpl_eval(10.0, 50.0, 5.0 * units, 2.0 * units, 4.0 * units);
        assert!((mu - fpl_mu).abs() < 1e-12);
        for (actual, expected) in [
            (dxmid, logis_xmid / units),
            (dscal, logis_scal / units),
            (gradient[2], fpl_gradient[2] / units),
            (gradient[3], fpl_gradient[3] / units),
        ] {
            assert!(
                (actual / expected - 1.0).abs() < 1e-12,
                "units={units}: gradient {actual}, expected {expected}"
            );
        }
    }
}
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
