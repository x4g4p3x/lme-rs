//! Michaelis–Menten mean (`stats::SSmicmen`).

/// Evaluate μ = `Vmax * x / (K + x)` and partial derivatives w.r.t. `Vmax`, `K`.
#[inline]
pub fn ssmicmen_eval(vmax: f64, k: f64, x: f64) -> (f64, f64, f64) {
    let denom = k + x;
    let mu = vmax * x / denom;
    let d_vmax = x / denom;
    let d_k = -vmax * x / (denom * denom);
    (mu, d_vmax, d_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let vmax = 10.0;
        let k = 2.0;
        let x = 3.0;
        let h = 1e-6;
        let (mu, dv, dk) = ssmicmen_eval(vmax, k, x);
        assert!((dv - (ssmicmen_eval(vmax + h, k, x).0 - mu) / h).abs() < 1e-5);
        assert!((dk - (ssmicmen_eval(vmax, k + h, x).0 - mu) / h).abs() < 1e-5);
    }
}
