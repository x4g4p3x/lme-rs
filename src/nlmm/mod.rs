//! Nonlinear mixed-effects models (`nlmer`-style).

mod fit;
mod formula;
mod sslogis;

pub use fit::{fit_nlmer, NlmerOptions, NlmmStart};
pub use formula::{parse_nlmer_formula, NlmerFormula, NlmmMeanKind};

use polars::prelude::DataFrame;

use crate::LmeFit;

/// Fit a nonlinear mixed-effects model (Gaussian, Laplace / PIRLS-style).
///
/// Formula syntax (three parts separated by `~`):
///
/// ```text
/// response ~ SSlogis(covariate, Asym, xmid, scal) ~ Asym|group
/// ```
///
/// Only random effects on `Asym` are supported in this release, matching the
/// canonical `nlmer(Orange)` example.
pub fn nlmer(
    formula: &str,
    data: &DataFrame,
    start: NlmmStart,
    reml: bool,
) -> crate::Result<LmeFit> {
    let (parsed, mean) = parse_nlmer_formula(formula)?;
    let opts = NlmerOptions {
        reml,
        start,
        ..NlmerOptions::default()
    };
    fit_nlmer(&parsed, mean, data, formula, &opts)
}
