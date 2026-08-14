//! Studentized-range CDF (`stats::ptukey`) for Tukey–Kramer p-values.
//!
//! `P(Q ≤ q)` for `Q = W / S` where `W` is the range of `nmeans` i.i.d. standard
//! normals and `S = √(χ²_ν / ν)` is independent. Infinite `df` uses `W` alone.
//! Quadrature matches R `ptukey` to about `1e-6` (typically much tighter for
//! `nmeans ≤ 5`).

#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

const SQRT2: f64 = std::f64::consts::SQRT_2;
const SQRT_PI: f64 = 1.772_453_850_905_516;

// 48-point Gauss–Legendre on [-1, 1] (Golub–Welsch).
#[rustfmt::skip]
const LEGENDRE_X: [f64; 48] = [
    -9.987710072524261e-1, -9.935301722663501e-1, -9.841245837228265e-1,
    -9.705915925462469e-1, -9.529877031604304e-1, -9.313866907065542e-1,
    -9.058791367155692e-1, -8.765720202742471e-1, -8.435882616243935e-1,
    -8.070662040294421e-1, -7.671590325157398e-1, -7.240341309238145e-1,
    -6.778723796326633e-1, -6.288673967765130e-1, -5.772247260839725e-1,
    -5.231609747222328e-1, -4.669029047509585e-1, -4.086864819907172e-1,
    -3.487558862921607e-1, -2.873624873554561e-1, -2.247637903946895e-1,
    -1.612223560688915e-1, -9.700469920946264e-2, -3.238017096286971e-2,
     3.238017096286960e-2,  9.700469920946275e-2,  1.612223560688919e-1,
     2.247637903946893e-1,  2.873624873554553e-1,  3.487558862921606e-1,
     4.086864819907169e-1,  4.669029047509587e-1,  5.231609747222330e-1,
     5.772247260839727e-1,  6.288673967765137e-1,  6.778723796326640e-1,
     7.240341309238147e-1,  7.671590325157402e-1,  8.070662040294427e-1,
     8.435882616243936e-1,  8.765720202742479e-1,  9.058791367155696e-1,
     9.313866907065542e-1,  9.529877031604309e-1,  9.705915925462472e-1,
     9.841245837228269e-1,  9.935301722663508e-1,  9.987710072524262e-1,
];
#[rustfmt::skip]
const LEGENDRE_W: [f64; 48] = [
    3.153346052304077e-3, 7.327553901275349e-3, 1.147723457923511e-2,
    1.557931572294446e-2, 1.961616045735513e-2, 2.357076083932421e-2,
    2.742650970835639e-2, 3.116722783279734e-2, 3.477722256477007e-2,
    3.824135106583208e-2, 4.154508294346503e-2, 4.467456085669406e-2,
    4.761665849249135e-2, 5.035903555385637e-2, 5.289018948519387e-2,
    5.519950369998428e-2, 5.727729210040373e-2, 5.911483969839561e-2,
    6.070443916589431e-2, 6.203942315989348e-2, 6.311419228625238e-2,
    6.392423858464734e-2, 6.446616443595025e-2, 6.473769681268304e-2,
    6.473769681268347e-2, 6.446616443594938e-2, 6.392423858464751e-2,
    6.311419228625456e-2, 6.203942315989391e-2, 6.070443916589393e-2,
    5.911483969839613e-2, 5.727729210040344e-2, 5.519950369998431e-2,
    5.289018948519353e-2, 5.035903555385442e-2, 4.761665849249060e-2,
    4.467456085669422e-2, 4.154508294346481e-2, 3.824135106583044e-2,
    3.477722256477062e-2, 3.116722783279807e-2, 2.742650970835680e-2,
    2.357076083932451e-2, 1.961616045735571e-2, 1.557931572294382e-2,
    1.147723457923440e-2, 7.327553901276181e-3, 3.153346052305805e-3,
];

// 32-point Gauss–Hermite: ∫ e^{-x²} f(x) dx (Golub–Welsch).
#[rustfmt::skip]
const HERMITE_X: [f64; 32] = [
    -7.125813909830725, -6.409498149269657, -5.812225949515918, -5.275550986515878,
    -4.777164503502592, -4.305547953351194, -3.853755485471442, -3.417167492818564,
    -2.992490825002372, -2.577249537732312, -2.169499183606110, -1.767654109463201,
    -1.370376410952867, -0.976500463589677, -0.584978765435928, -0.194840741569399,
     0.194840741569402,  0.584978765435933,  0.976500463589684,  1.370376410952874,
     1.767654109463202,  2.169499183606113,  2.577249537732318,  2.992490825002374,
     3.417167492818572,  3.853755485471446,  4.305547953351199,  4.777164503502595,
     5.275550986515878,  5.812225949515912,  6.409498149269659,  7.125813909830728,
];
#[rustfmt::skip]
const HERMITE_W: [f64; 32] = [
    7.310676427383912e-23, 9.231736536518524e-19, 1.197344017092760e-15,
    4.215010211326317e-13, 5.933291463396425e-11, 4.098832164770885e-9,
    1.574167792545528e-7,  3.650585129562385e-6,  5.416584061819871e-5,
    5.362683655279629e-4,  3.654890326654330e-3,  1.755342883157280e-2,
    6.045813095591266e-2,  1.512697340766450e-1,  2.774581423025262e-1,
    3.752383525928041e-1,  3.752383525927986e-1,  2.774581423025272e-1,
    1.512697340766430e-1,  6.045813095591362e-2,  1.755342883157378e-2,
    3.654890326654487e-3,  5.362683655279718e-4,  5.416584061819994e-5,
    3.650585129562392e-6,  1.574167792545590e-7,  4.098832164770894e-9,
    5.933291463396705e-11, 4.215010211326561e-13, 1.197344017092878e-15,
    9.231736536518381e-19, 7.310676427384068e-23,
];

fn normal_cdf(x: f64) -> f64 {
    Normal::new(0.0, 1.0).expect("standard normal").cdf(x)
}

/// CDF of the range of `nmeans` i.i.d. N(0, 1).
fn range_of_normals_cdf(w: f64, nmeans: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    if !w.is_finite() {
        return 1.0;
    }
    if (nmeans - 2.0).abs() < 1e-12 {
        return (2.0 * normal_cdf(w / SQRT2) - 1.0).clamp(0.0, 1.0);
    }
    let mut acc = 0.0;
    for (&t, &wt) in HERMITE_X.iter().zip(HERMITE_W.iter()) {
        let x = t * SQRT2;
        let d = normal_cdf(x) - normal_cdf(x - w);
        if d > 0.0 {
            acc += wt * (d.ln() * (nmeans - 1.0)).exp();
        }
    }
    (nmeans / SQRT_PI * acc).clamp(0.0, 1.0)
}

fn log_s_density(s: f64, df: f64) -> f64 {
    if s <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (df / 2.0) * df.ln() - (df / 2.0 - 1.0) * std::f64::consts::LN_2 - ln_gamma(df / 2.0)
        + (df - 1.0) * s.ln()
        - df * s * s / 2.0
}

/// `P(Q ≤ q)` for the studentized range with `nmeans` groups and `df` residual df.
pub(crate) fn ptukey(q: f64, nmeans: f64, df: f64) -> f64 {
    if !q.is_finite() {
        return if q.is_sign_positive() { 1.0 } else { 0.0 };
    }
    if q <= 0.0 || nmeans < 2.0 {
        return 0.0;
    }
    if !df.is_finite() || df > 1.0e8 {
        return range_of_normals_cdf(q, nmeans);
    }
    if df < 1.0 {
        return f64::NAN;
    }
    let half_width = 12.0 / (2.0 * df).sqrt();
    let a = (1.0 - half_width).max(1e-12);
    let b = 1.0 + half_width;
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut acc = 0.0;
    for (&z, &wt) in LEGENDRE_X.iter().zip(LEGENDRE_W.iter()) {
        let s = half * z + mid;
        let lp = log_s_density(s, df);
        if lp > -700.0 {
            acc += wt * lp.exp() * range_of_normals_cdf(q * s, nmeans);
        }
    }
    (acc * half).clamp(0.0, 1.0)
}

/// Two-sided Tukey–Kramer p-value from a Wald `t` (or `z` with infinite df).
pub(crate) fn tukey_kramer_p(t: f64, n_groups: usize, df: f64) -> f64 {
    if n_groups < 2 {
        return f64::NAN;
    }
    let q = SQRT2 * t.abs();
    (1.0 - ptukey(q, n_groups as f64, df)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{ptukey, tukey_kramer_p};

    fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
        let diff = (got - expected).abs();
        assert!(
            diff <= tol,
            "{name}: got {got}, expected {expected} (|Δ|={diff} > {tol})"
        );
    }

    #[test]
    fn ptukey_matches_r_stats() {
        // R 4.3.3 stats::ptukey
        assert_close(
            "k2 inf q2",
            ptukey(2.0, 2.0, f64::INFINITY),
            0.842_700_792_949_714_8,
            1e-12,
        );
        assert_close(
            "k3 inf q3",
            ptukey(3.0, 3.0, f64::INFINITY),
            0.914_457_428_345_042_4,
            1e-10,
        );
        assert_close(
            "k3 df48 q3",
            ptukey(3.0, 3.0, 48.0),
            0.903_852_487_876_580_8,
            1e-9,
        );
        assert_close(
            "k3 df20 q2.5",
            ptukey(2.5, 3.0, 20.0),
            0.794_197_913_064_306_0,
            1e-9,
        );
        assert_close(
            "k2 df10 q0.5",
            ptukey(0.5, 2.0, 10.0),
            0.268_986_189_324_594_8,
            1e-9,
        );
        assert_close(
            "k2 df120 q4",
            ptukey(4.0, 2.0, 120.0),
            0.994_516_311_294_083_1,
            1e-9,
        );
        assert_close(
            "k8 df10 q4",
            ptukey(4.0, 8.0, 10.0),
            0.810_560_649_322_259_5,
            2e-6,
        );
    }

    #[test]
    fn tukey_kramer_two_groups_equals_two_sided_normal() {
        use statrs::distribution::{ContinuousCDF, Normal};
        let z = 1.96;
        let p = tukey_kramer_p(z, 2, f64::INFINITY);
        let expected = 2.0 * (1.0 - Normal::new(0.0, 1.0).unwrap().cdf(z));
        assert_close("k=2 tukey vs z", p, expected, 1e-10);
    }
}
