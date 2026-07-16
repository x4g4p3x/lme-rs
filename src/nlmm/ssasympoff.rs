//! Offset asymptotic regression (`stats::SSasympOff`).
//!
//! μ = `Asym * (1 - exp(-exp(lrc) * (x - c0)))`

/// Evaluate μ and partials w.r.t. `Asym`, `lrc`, `c0`.
#[inline]
pub fn ssasympoff_eval(asym: f64, lrc: f64, c0: f64, x: f64) -> (f64, Vec<f64>) {
    let k = lrc.exp();
    let z = x - c0;
    let e = (-k * z).exp();
    let mu = asym * (1.0 - e);
    let d_asym = 1.0 - e;
    let d_lrc = asym * e * (k * z);
    let d_c0 = -asym * e * k;
    (mu, vec![d_asym, d_lrc, d_c0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let asym = 90.0;
        let lrc = (0.4_f64).ln();
        let c0 = 0.5;
        let x = 2.0;
        let h = 1e-6;
        let (_mu, g) = ssasympoff_eval(asym, lrc, c0, x);
        let params = [asym, lrc, c0];
        for i in 0..3 {
            let mut p_lo = params;
            let mut p_hi = params;
            p_lo[i] -= h;
            p_hi[i] += h;
            let mu_lo = ssasympoff_eval(p_lo[0], p_lo[1], p_lo[2], x).0;
            let mu_hi = ssasympoff_eval(p_hi[0], p_hi[1], p_hi[2], x).0;
            let fd = (mu_hi - mu_lo) / (2.0 * h);
            assert!(
                (g[i] - fd).abs() < 1e-4,
                "param {i}: analytic={} fd={fd}",
                g[i]
            );
        }
    }
}
