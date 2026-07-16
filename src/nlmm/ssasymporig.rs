//! Origin asymptotic regression (`stats::SSasympOrig`).
//!
//! μ = `Asym * (1 - exp(-exp(lrc) * x))`

/// Evaluate μ and partials w.r.t. `Asym`, `lrc`.
#[inline]
pub fn ssasymporig_eval(asym: f64, lrc: f64, x: f64) -> (f64, Vec<f64>) {
    let k = lrc.exp();
    let e = (-k * x).exp();
    let mu = asym * (1.0 - e);
    let d_asym = 1.0 - e;
    let d_lrc = asym * e * (k * x);
    (mu, vec![d_asym, d_lrc])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let asym = 90.0;
        let lrc = (0.4_f64).ln();
        let x = 2.0;
        let h = 1e-6;
        let (_mu, g) = ssasymporig_eval(asym, lrc, x);
        let params = [asym, lrc];
        for i in 0..2 {
            let mut p_lo = params;
            let mut p_hi = params;
            p_lo[i] -= h;
            p_hi[i] += h;
            let mu_lo = ssasymporig_eval(p_lo[0], p_lo[1], x).0;
            let mu_hi = ssasymporig_eval(p_hi[0], p_hi[1], x).0;
            let fd = (mu_hi - mu_lo) / (2.0 * h);
            assert!(
                (g[i] - fd).abs() < 1e-4,
                "param {i}: analytic={} fd={fd}",
                g[i]
            );
        }
    }
}
