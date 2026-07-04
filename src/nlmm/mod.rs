//! Nonlinear mixed-effects models (`nlmer`-style).

mod fit;
mod formula;
mod mean;
pub(crate) mod predict;
pub(crate) mod re_cov;
mod self_start;
mod ssasymp;
mod sslogis;
pub use re_cov::sigma_from_theta;
pub use ssasymp::ssasymp_eval;
/// Alias for [`ssasymp_eval`] (`stats::SSfol` uses the same mean function).
pub use ssasymp::ssasymp_eval as ssfol_eval;
pub use sslogis::sslogis_eval;

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
/// response ~ SSlogis(covariate, Asym, xmid, scal) ~ Asym + xmid | group
/// response ~ SSasymp(covariate, Asym, R0, lrc) ~ Asym|group
/// response ~ SSfol(covariate, Asym, R0, lrc) ~ Asym|group
/// ```
///
/// Supported means: `SSlogis`, `SSasymp`, `SSfol`. When `start` is empty,
/// data-driven `selfStart` heuristics (R `stats::getInitial`) are used.
/// Random effects are additive on the named nonlinear parameters (one or more
/// per grouping factor).
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
