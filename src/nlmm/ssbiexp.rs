//! Bi-exponential mean (`stats::SSbiexp`).
//!
//! μ = `A1 * exp(-exp(lrc1) * x) + A2 * exp(-exp(lrc2) * x)`

/// Evaluate μ and partials w.r.t. `A1`, `lrc1`, `A2`, `lrc2`.
#[inline]
pub fn ssbiexp_eval(a1: f64, lrc1: f64, a2: f64, lrc2: f64, x: f64) -> (f64, Vec<f64>) {
    let k1 = lrc1.exp();
    let k2 = lrc2.exp();
    let e1 = (-k1 * x).exp();
    let e2 = (-k2 * x).exp();
    let mu = a1 * e1 + a2 * e2;
    let d_a1 = e1;
    let d_lrc1 = a1 * e1 * (-k1 * x);
    let d_a2 = e2;
    let d_lrc2 = a2 * e2 * (-k2 * x);
    (mu, vec![d_a1, d_lrc1, d_a2, d_lrc2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let a1 = 5.0;
        let lrc1 = 0.5_f64.ln();
        let a2 = 3.0;
        let lrc2 = 0.1_f64.ln();
        let x = 2.0;
        let h = 1e-6;
        let (_mu, g) = ssbiexp_eval(a1, lrc1, a2, lrc2, x);
        let params = [a1, lrc1, a2, lrc2];
        for i in 0..4 {
            let mut p_lo = params;
            let mut p_hi = params;
            p_lo[i] -= h;
            p_hi[i] += h;
            let mu_lo = ssbiexp_eval(p_lo[0], p_lo[1], p_lo[2], p_lo[3], x).0;
            let mu_hi = ssbiexp_eval(p_hi[0], p_hi[1], p_hi[2], p_hi[3], x).0;
            let fd = (mu_hi - mu_lo) / (2.0 * h);
            assert!(
                (g[i] - fd).abs() < 1e-4,
                "param {i}: analytic={} fd={fd}",
                g[i]
            );
        }
    }
}
