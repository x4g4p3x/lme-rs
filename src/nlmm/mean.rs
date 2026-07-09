//! Default starting values for built-in nonlinear means.

use crate::nlmm::formula::NlmmMeanKind;

/// Default starting values when the user omits `start`.
pub(crate) fn default_start(kind: NlmmMeanKind, names: &[String]) -> Vec<f64> {
    let defaults: &[(&str, f64)] = match kind {
        NlmmMeanKind::Sslogis => &[("Asym", 200.0), ("xmid", 725.0), ("scal", 350.0)],
        NlmmMeanKind::Ssasymp | NlmmMeanKind::Ssfol => {
            &[("Asym", 90.0), ("R0", 20.0), ("lrc", (0.4_f64).ln())]
        }
        NlmmMeanKind::Ssmicmen => &[("Vmax", 10.0), ("K", 1.0)],
        NlmmMeanKind::Ssgompertz => &[("Asym", 50.0), ("b2", 1.0), ("b3", 0.3)],
        NlmmMeanKind::Sspower => &[("a", 1.0), ("b", 1.0), ("c", 0.0)],
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
