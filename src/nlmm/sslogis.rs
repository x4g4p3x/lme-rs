//! Three-parameter logistic mean (`stats::SSlogis` / `nlmer` examples).

/// Logistic fraction, its complement, and the positive derivative magnitude.
/// Evaluate the smaller tail directly so neither exponentials nor their squares
/// overflow, and retain the complement when the fraction rounds to one.
#[inline]
pub(super) fn logistic_parts(t: f64) -> (f64, f64, f64) {
    let e = (-t.abs()).exp();
    let denom = 1.0 + e;
    let small = e / denom;
    let large = 1.0 / denom;
    let (fraction, complement) = if t >= 0.0 {
        (small, large)
    } else {
        (large, small)
    };
    (fraction, complement, small * large)
}

/// Evaluate μ = `Asym / (1 + exp((xmid - x) / scal))` and partial derivatives.
///
/// Random effects enter additively on `Asym`: use `a = Asym + b_group` when building μ.
#[inline]
pub fn sslogis_eval(a: f64, xmid: f64, scal: f64, x: f64) -> (f64, f64, f64, f64) {
    let t = (xmid - x) / scal;
    let (fraction, _, slope) = logistic_parts(t);
    let mu = a * fraction;
    let d_mu_d_a = fraction;
    let scaled_slope = a * slope / scal;
    let d_mu_d_xmid = -scaled_slope;
    let d_mu_d_scal = if t.is_infinite() && slope == 0.0 {
        0.0
    } else {
        scaled_slope * t
    };
    (mu, d_mu_d_a, d_mu_d_xmid, d_mu_d_scal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let a = 192.0;
        let xmid = 728.0;
        let scal = 348.0;
        let x = 500.0;
        let h = 1e-6;
        let (mu, da, dx, ds) = sslogis_eval(a, xmid, scal, x);
        let mu_a = sslogis_eval(a + h, xmid, scal, x).0;
        let mu_x = sslogis_eval(a, xmid + h, scal, x).0;
        let mu_s = sslogis_eval(a, xmid, scal + h, x).0;
        assert!((da - (mu_a - mu) / h).abs() < 1e-5);
        assert!((dx - (mu_x - mu) / h).abs() < 1e-4);
        assert!((ds - (mu_s - mu) / h).abs() < 1e-4);
    }
}
