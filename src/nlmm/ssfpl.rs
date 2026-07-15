//! Four-parameter logistic mean (`stats::SSfpl`).
//!
//! μ = `A + (B - A) / (1 + exp((xmid - x) / scal))`

/// Evaluate μ and partials w.r.t. `A`, `B`, `xmid`, `scal`.
#[inline]
pub fn ssfpl_eval(a: f64, b: f64, xmid: f64, scal: f64, x: f64) -> (f64, Vec<f64>) {
    let z = (xmid - x) / scal;
    let e = z.exp();
    let denom = 1.0 + e;
    let frac = 1.0 / denom;
    let mu = a + (b - a) * frac;
    let d_frac_dz = -e / (denom * denom);
    let d_a = 1.0 - frac;
    let d_b = frac;
    let d_xmid = (b - a) * d_frac_dz / scal;
    let d_scal = (b - a) * d_frac_dz * (-(xmid - x) / (scal * scal));
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
