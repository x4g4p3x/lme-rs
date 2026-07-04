//! Gauss–Hermite quadrature helpers (shared by GLMM and NLMM).

/// Maximum spherical RE dimension for joint multivariate AGQ.
pub(crate) const AGQ_JOINT_MAX_Q: usize = 8;

pub(crate) fn resolve_gh_order(n_agq: usize) -> Option<usize> {
    if n_agq < 2 {
        return None;
    }
    const SUPPORTED: [usize; 5] = [3, 5, 7, 9, 11];
    for &s in &SUPPORTED {
        if s >= n_agq {
            return Some(s);
        }
    }
    Some(11)
}

mod gh_rule_tables {
    #![allow(clippy::excessive_precision)]
    // lme4 / fastGHQuad-style Gauss–Hermite rules: z = sqrt(2) * x (physicist roots x), weights sum to 1.
    pub(crate) const GH_Z3: [f64; 3] = [-1.7320508075688772, 0.0, 1.7320508075688772];
    pub(crate) const GH_W3: [f64; 3] =
        [0.16666666666666669, 0.6666666666666665, 0.16666666666666669];
    pub(crate) const GH_Z5: [f64; 5] = [
        -2.8569700138728056,
        -1.3556261799742659,
        0.0,
        1.3556261799742659,
        2.8569700138728056,
    ];
    pub(crate) const GH_W5: [f64; 5] = [
        0.011257411327720693,
        0.2220759220056126,
        0.5333333333333333,
        0.2220759220056126,
        0.011257411327720693,
    ];
    pub(crate) const GH_Z7: [f64; 7] = [
        -3.7504397177257425,
        -2.3667594107345416,
        -1.1544053947399682,
        0.0,
        1.1544053947399682,
        2.3667594107345416,
        3.7504397177257425,
    ];
    pub(crate) const GH_W7: [f64; 7] = [
        0.0005482688559722182,
        0.03075712396758651,
        0.24012317860501273,
        0.45714285714285713,
        0.24012317860501273,
        0.03075712396758651,
        0.0005482688559722182,
    ];
    pub(crate) const GH_Z9: [f64; 9] = [
        -4.512745863399783,
        -3.2054290028564703,
        -2.07684797867783,
        -1.0232556637891326,
        0.0,
        1.0232556637891326,
        2.07684797867783,
        3.2054290028564703,
        4.512745863399783,
    ];
    pub(crate) const GH_W9: [f64; 9] = [
        2.2345844007746576e-5,
        0.0027891413212317653,
        0.04991640676521791,
        0.2440975028949394,
        0.4063492063492064,
        0.2440975028949394,
        0.04991640676521791,
        0.0027891413212317653,
        2.2345844007746576e-5,
    ];
    pub(crate) const GH_Z11: [f64; 11] = [
        -5.188001224374871,
        -3.9361666071299766,
        -2.8651231606436456,
        -1.876035020154846,
        -0.9288689973810641,
        0.0,
        0.9288689973810641,
        1.876035020154846,
        2.8651231606436456,
        3.9361666071299766,
        5.188001224374871,
    ];
    pub(crate) const GH_W11: [f64; 11] = [
        8.121849790214923e-7,
        0.00019567193027122338,
        0.006720285235537264,
        0.06613874607105782,
        0.24224029987396992,
        0.3694083694083694,
        0.24224029987396992,
        0.06613874607105782,
        0.006720285235537264,
        0.00019567193027122338,
        8.121849790214923e-7,
    ];
}

pub(crate) fn gh_rule(order: usize) -> Option<(&'static [f64], &'static [f64])> {
    use gh_rule_tables::*;
    match order {
        3 => Some((&GH_Z3[..], &GH_W3[..])),
        5 => Some((&GH_Z5[..], &GH_W5[..])),
        7 => Some((&GH_Z7[..], &GH_W7[..])),
        9 => Some((&GH_Z9[..], &GH_W9[..])),
        11 => Some((&GH_Z11[..], &GH_W11[..])),
        _ => None,
    }
}

/// Picks a 1D quadrature order; for `k > 1`, reduces `order` so `order^k` stays bounded.
pub(crate) fn resolve_gh_order_product(n_agq: usize, k: usize) -> Option<usize> {
    let base = resolve_gh_order(n_agq)?;
    if k == 1 {
        return Some(base);
    }
    const MAX_POINTS: usize = 400;
    let mut ord = base;
    loop {
        if ord.pow(k as u32) <= MAX_POINTS {
            return Some(ord);
        }
        ord = match ord {
            11 => 9,
            9 => 7,
            7 => 5,
            5 => 3,
            3 => return None,
            _ => return None,
        };
    }
}

/// Product-grid size cap for joint integration over all `q` random effects at once.
pub(crate) fn resolve_gh_order_joint(n_agq: usize, q: usize) -> Option<usize> {
    if q == 0 || q > AGQ_JOINT_MAX_Q {
        return None;
    }
    let base = resolve_gh_order(n_agq)?;
    const MAX_POINTS: usize = 7000;
    let mut ord = base;
    loop {
        if ord.pow(q as u32) <= MAX_POINTS {
            return Some(ord);
        }
        ord = match ord {
            11 => 9,
            9 => 7,
            7 => 5,
            5 => 3,
            3 => return None,
            _ => return None,
        };
    }
}

pub(crate) fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_basic() {
        let xs = [0.0_f64, 1.0, 2.0];
        let lse = log_sum_exp(&xs);
        let expected = (1.0 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((lse - expected).abs() < 1e-12);
    }
}
