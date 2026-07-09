//! Two-term power-series mean (MATLAB Curve Fitter `power2`: `a * x^b + c`).

/// Evaluate μ = `a * x^b + c` and partial derivatives w.r.t. `a`, `b`, `c`.
///
/// Requires `x > 0` for a real-valued gradient when `b` is not an integer.
#[inline]
pub fn sspower_eval(a: f64, b: f64, c: f64, x: f64) -> (f64, f64, f64, f64) {
    if x <= 0.0 || !x.is_finite() {
        return (f64::NAN, 0.0, 0.0, 0.0);
    }
    let xb = x.powf(b);
    let mu = a * xb + c;
    let d_a = xb;
    let d_b = a * xb * x.ln();
    let d_c = 1.0;
    (mu, d_a, d_b, d_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_matlab_power2_example_shape() {
        // f(x) = a*x^b + c at a=2, b=0.5, c=1, x=4 => 2*2 + 1 = 5
        let (mu, da, db, dc) = sspower_eval(2.0, 0.5, 1.0, 4.0);
        assert!((mu - 5.0).abs() < 1e-12);
        assert!((da - 2.0).abs() < 1e-12);
        assert!((dc - 1.0).abs() < 1e-12);
        assert!(db.is_finite());
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let a = 2.0;
        let b = 0.5;
        let c = 1.0;
        let x = 3.0;
        let h = 1e-6;
        let (mu, da, db, dc) = sspower_eval(a, b, c, x);
        assert!((da - (sspower_eval(a + h, b, c, x).0 - mu) / h).abs() < 1e-5);
        assert!((db - (sspower_eval(a, b + h, c, x).0 - mu) / h).abs() < 1e-4);
        assert!((dc - (sspower_eval(a, b, c + h, x).0 - mu) / h).abs() < 1e-5);
    }
}
