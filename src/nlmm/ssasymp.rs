//! Asymptotic regression mean (`stats::SSasymp`).

/// Evaluate μ = `Asym - (Asym - R0) * exp(-exp(lrc) * x)` and partial derivatives.
///
/// Random effects enter additively on any parameter via the caller (typically `Asym`).
#[inline]
pub fn ssasymp_eval(asym: f64, r0: f64, lrc: f64, x: f64) -> (f64, f64, f64, f64) {
    let k = lrc.exp();
    let e = (-k * x).exp();
    let mu = asym - (asym - r0) * e;
    let d_mu_d_asym = 1.0 - e;
    let d_mu_d_r0 = e;
    let d_mu_d_lrc = (asym - r0) * x * k * e;
    (mu, d_mu_d_asym, d_mu_d_r0, d_mu_d_lrc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let asym = 100.0;
        let r0 = 15.0;
        let lrc = -0.5;
        let x = 1.5;
        let h = 1e-6;
        let (mu, da, dr, dl) = ssasymp_eval(asym, r0, lrc, x);
        assert!((da - (ssasymp_eval(asym + h, r0, lrc, x).0 - mu) / h).abs() < 1e-5);
        assert!((dr - (ssasymp_eval(asym, r0 + h, lrc, x).0 - mu) / h).abs() < 1e-5);
        assert!((dl - (ssasymp_eval(asym, r0, lrc + h, x).0 - mu) / h).abs() < 1e-5);
    }
}
