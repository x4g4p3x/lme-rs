//! Three-parameter logistic mean (`stats::SSlogis` / `nlmer` examples).

/// Evaluate μ = `Asym / (1 + exp((xmid - x) / scal))` and partial derivatives.
///
/// Random effects enter additively on `Asym`: use `a = Asym + b_group` when building μ.
#[inline]
pub fn sslogis_eval(a: f64, xmid: f64, scal: f64, x: f64) -> (f64, f64, f64, f64) {
    let t = (xmid - x) / scal;
    let e = t.exp();
    let d = 1.0 + e;
    let mu = a / d;
    let d_mu_d_a = 1.0 / d;
    let d_mu_d_xmid = -a * e / (scal * d * d);
    let d_mu_d_scal = a * e * t / (scal * d * d);
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
