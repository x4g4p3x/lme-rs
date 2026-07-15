//! Weibull growth mean (`stats::SSweibull`).
//!
//! μ = `Asym - Drop * exp(-exp(lrc) * x^pwr)`

/// Evaluate μ and partials w.r.t. `Asym`, `Drop`, `lrc`, `pwr`.
#[inline]
pub fn ssweibull_eval(asym: f64, drop: f64, lrc: f64, pwr: f64, x: f64) -> (f64, Vec<f64>) {
    let x_safe = x.max(0.0);
    let xp = if x_safe == 0.0 { 0.0 } else { x_safe.powf(pwr) };
    let k = lrc.exp();
    let e = (-k * xp).exp();
    let mu = asym - drop * e;
    let d_asym = 1.0;
    let d_drop = -e;
    let d_lrc = -drop * e * (-k * xp);
    let d_pwr = if x_safe <= 0.0 {
        0.0
    } else {
        -drop * e * (-k * xp * x_safe.ln())
    };
    (mu, vec![d_asym, d_drop, d_lrc, d_pwr])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let asym = 100.0;
        let drop = 80.0;
        let lrc = -1.0;
        let pwr = 1.5;
        let x = 2.0;
        let h = 1e-6;
        let (_mu, g) = ssweibull_eval(asym, drop, lrc, pwr, x);
        let params = [asym, drop, lrc, pwr];
        for i in 0..4 {
            let mut p_lo = params;
            let mut p_hi = params;
            p_lo[i] -= h;
            p_hi[i] += h;
            let mu_lo = ssweibull_eval(p_lo[0], p_lo[1], p_lo[2], p_lo[3], x).0;
            let mu_hi = ssweibull_eval(p_hi[0], p_hi[1], p_hi[2], p_hi[3], x).0;
            let fd = (mu_hi - mu_lo) / (2.0 * h);
            assert!(
                (g[i] - fd).abs() < 1e-3,
                "param {i}: analytic={} fd={fd}",
                g[i]
            );
        }
    }
}
