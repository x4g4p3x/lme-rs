//! Michaelis–Menten mean (`stats::SSmicmen`).

/// Evaluate μ = `Vmax * x / (K + x)` and partial derivatives w.r.t. `Vmax`, `K`.
#[inline]
pub fn ssmicmen_eval(vmax: f64, k: f64, x: f64) -> (f64, f64, f64) {
    let denom = k + x;
    // Form the dimensionless fraction before multiplying by Vmax, and avoid
    // squaring the denominator: both intermediates can overflow/underflow
    // despite representable means and gradients after a change of units.
    let (d_vmax, scale, scaled_denom) = if denom.is_infinite() && k.is_finite() && x.is_finite() {
        // A same-sign finite sum can overflow. Normalize only in that case,
        // preserving the original denominator near cancellation and at zero.
        let scale = k.abs().max(x.abs());
        let scaled_denom = k / scale + x / scale;
        ((x / scale) / scaled_denom, scale, scaled_denom)
    } else {
        (x / denom, 1.0, denom)
    };
    let mu = vmax * d_vmax;
    let d_k = -(mu / scale) / scaled_denom;
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
