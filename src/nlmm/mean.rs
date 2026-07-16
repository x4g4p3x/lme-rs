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
        NlmmMeanKind::Ssfpl => &[("A", 10.0), ("B", 50.0), ("xmid", 5.0), ("scal", 2.0)],
        NlmmMeanKind::Ssbiexp => &[
            ("A1", 5.0),
            ("lrc1", (0.5_f64).ln()),
            ("A2", 3.0),
            ("lrc2", (0.1_f64).ln()),
        ],
        NlmmMeanKind::Ssweibull => &[("Asym", 100.0), ("Drop", 80.0), ("lrc", -1.0), ("pwr", 1.5)],
        NlmmMeanKind::Ssasympoff => &[("Asym", 90.0), ("lrc", (0.4_f64).ln()), ("c0", 0.5)],
        NlmmMeanKind::Ssasymporig => &[("Asym", 90.0), ("lrc", (0.4_f64).ln())],
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
