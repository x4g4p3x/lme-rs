//! Four-parameter logistic mean (`stats::SSfpl`).
//!
//! μ = `A + (B - A) / (1 + exp((xmid - x) / scal))`

use super::sslogis::logistic_parts;

/// Evaluate μ and partials w.r.t. `A`, `B`, `xmid`, `scal`.
#[inline]
pub fn ssfpl_eval(a: f64, b: f64, xmid: f64, scal: f64, x: f64) -> (f64, Vec<f64>) {
    let z = (xmid - x) / scal;
    let (frac, complement, slope) = logistic_parts(z);
    let mu = a + (b - a) * frac;
    let d_a = complement;
    let d_b = frac;
    let scaled_slope = (b - a) * slope / scal;
    let d_xmid = -scaled_slope;
    let d_scal = if z.is_infinite() && slope == 0.0 {
        0.0
    } else {
        scaled_slope * z
    };
    (mu, vec![d_a, d_b, d_xmid, d_scal])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let a = 10.0;
        let b = 50.0;
        let xmid = 5.0;
        let scal = 2.0;
        let x = 4.0;
        let h = 1e-6;
        let (_mu, g) = ssfpl_eval(a, b, xmid, scal, x);
        let params = [a, b, xmid, scal];
        for i in 0..4 {
            let mut p_lo = params;
            let mut p_hi = params;
            p_lo[i] -= h;
            p_hi[i] += h;
            let mu_lo = ssfpl_eval(p_lo[0], p_lo[1], p_lo[2], p_lo[3], x).0;
            let mu_hi = ssfpl_eval(p_hi[0], p_hi[1], p_hi[2], p_hi[3], x).0;
            let fd = (mu_hi - mu_lo) / (2.0 * h);
            assert!(
                (g[i] - fd).abs() < 1e-4,
                "param {i}: analytic={} fd={fd}",
                g[i]
            );
        }
    }
}
