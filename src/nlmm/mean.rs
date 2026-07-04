//! Nonlinear mean evaluation with optional random-effect offsets on parameters.

use crate::nlmm::formula::NlmmMeanKind;
use crate::nlmm::ssasymp::ssasymp_eval;
use crate::nlmm::sslogis::sslogis_eval;

/// μ and ∂μ/∂(each fixed parameter), with RE offsets already applied to effective parameters.
pub(crate) fn eval_mean(
    kind: NlmmMeanKind,
    x: f64,
    params: &[f64],
    re_param_indices: &[usize],
    re_offsets: &[f64],
) -> (f64, Vec<f64>) {
    let mut effective = params.to_vec();
    for (idx, off) in re_param_indices.iter().zip(re_offsets.iter()) {
        effective[*idx] += *off;
    }

    match kind {
        NlmmMeanKind::Sslogis => {
            let (mu, da, dx, ds) = sslogis_eval(effective[0], effective[1], effective[2], x);
            (mu, vec![da, dx, ds])
        }
        NlmmMeanKind::Ssasymp => {
            let (mu, da, dr, dl) = ssasymp_eval(effective[0], effective[1], effective[2], x);
            (mu, vec![da, dr, dl])
        }
    }
}

/// Default starting values when the user omits `start`.
pub(crate) fn default_start(kind: NlmmMeanKind, names: &[String]) -> Vec<f64> {
    let defaults: &[(&str, f64)] = match kind {
        NlmmMeanKind::Sslogis => &[("Asym", 200.0), ("xmid", 725.0), ("scal", 350.0)],
        NlmmMeanKind::Ssasymp => &[("Asym", 90.0), ("R0", 20.0), ("lrc", (0.4_f64).ln())],
    };
    names
        .iter()
        .map(|name| {
            defaults
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .unwrap_or(1.0)
        })
        .collect()
}
