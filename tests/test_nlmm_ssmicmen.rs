//! Synthetic `SSmicmen` nlmer smoke test.

use lme_rs::nlmer;
use lme_rs::nlmm::{ssmicmen_eval, NlmmStart};
use polars::prelude::*;

#[test]
fn ssmicmen_gradients_transform_with_predictor_units() {
    // Vmax=10, K=2, x=3 gives mu=6, dVmax=3/5, dK=-6/5.
    // Scaling both K and x preserves the curve and divides dK by the scale.
    for units in [1.0, 1e-200, 1e200, -1e-200, -1e200] {
        let (mu, dv, dk) = ssmicmen_eval(10.0, 2.0 * units, 3.0 * units);
        assert!((mu / 6.0 - 1.0).abs() < 1e-14, "units={units}: mu={mu}");
        assert!((dv / 0.6 - 1.0).abs() < 1e-14);
        assert!(
            (dk / (-1.2 / units) - 1.0).abs() < 1e-14,
            "units={units}: dK={dk}"
        );
    }
}

#[test]
fn ssmicmen_retains_finite_results_when_intermediates_overflow() {
    // The numerator overflows, but the rational mean and gradient do not.
    let (mu, dv, dk) = ssmicmen_eval(1e308, 2.0, 3.0);
    assert!((mu / 6e307 - 1.0).abs() < 1e-14, "mu={mu}");
    assert_eq!(dv, 0.6);
    assert!((dk / -1.2e307 - 1.0).abs() < 1e-14);

    // At x=K, the mean is Vmax/2 and dK=-Vmax/(4*K), even if K+x overflows.
    for k in [1e308, -1e308] {
        let (mu, dv, dk) = ssmicmen_eval(10.0, k, k);
        assert_eq!(mu, 5.0);
        assert_eq!(dv, 0.5);
        assert!((dk / (-2.5 / k) - 1.0).abs() < 1e-14, "dK={dk}");
    }
}

#[test]
fn ssmicmen_preserves_zero_origin_and_singular_denominator() {
    assert_eq!(ssmicmen_eval(10.0, 2.0, 0.0), (0.0, 0.0, 0.0));
    let (mu, dv, dk) = ssmicmen_eval(10.0, -2.0, 2.0);
    assert!(!mu.is_finite() && !dv.is_finite() && !dk.is_finite());
}

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
