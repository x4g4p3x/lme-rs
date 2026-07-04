//! Gompertz growth mean (`stats::SSgompertz`: `Asym * exp(-b2 * b3^x)`).

/// Evaluate μ and partials w.r.t. `Asym`, `b2`, `b3`.
#[inline]
pub fn ssgompertz_eval(asym: f64, b2: f64, b3: f64, x: f64) -> (f64, Vec<f64>) {
    let pow = b3.powf(x);
    let t = (-b2 * pow).exp();
    let mu = asym * t;
    let d_asym = t;
    let d_b2 = -asym * t * pow;
    let d_b3 = if x == 0.0 {
        0.0
    } else {
        -asym * t * b2 * x * b3.powf(x - 1.0)
    };
    (mu, vec![d_asym, d_b2, d_b3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let asym = 50.0;
        let b2 = 2.0;
        let b3 = 0.3;
        let x = 2.0;
        let h = 1e-6;
        let (_mu, g) = ssgompertz_eval(asym, b2, b3, x);
        let params = [asym, b2, b3];
        for i in 0..3 {
            let mut p_lo = params;
            let mut p_hi = params;
            p_lo[i] -= h;
            p_hi[i] += h;
            let mu_lo = ssgompertz_eval(p_lo[0], p_lo[1], p_lo[2], x).0;
            let mu_hi = ssgompertz_eval(p_hi[0], p_hi[1], p_hi[2], x).0;
            let fd = (mu_hi - mu_lo) / (2.0 * h);
            assert!(
                (g[i] - fd).abs() < 1e-3,
                "param {i}: analytic={} fd={fd}",
                g[i]
            );
        }
    }
}
