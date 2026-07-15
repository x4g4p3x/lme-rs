//! Nonlinear mixed-effects models (`nlmer`-style).

mod fit;
mod formula;
mod mean;
mod mean_fn;
pub(crate) mod predict;
pub(crate) mod re_cov;
mod self_start;
mod ssasymp;
mod ssbiexp;
mod ssfpl;
mod ssgompertz;
mod sslogis;
mod ssmicmen;
mod sspower;
mod ssweibull;
pub use re_cov::sigma_from_theta;
pub use ssasymp::ssasymp_eval;
/// Alias for [`ssasymp_eval`] (`stats::SSfol` uses the same mean function).
pub use ssasymp::ssasymp_eval as ssfol_eval;
pub use ssbiexp::ssbiexp_eval;
pub use ssfpl::ssfpl_eval;
pub use ssgompertz::ssgompertz_eval;
pub use sslogis::sslogis_eval;
pub use ssmicmen::ssmicmen_eval;
pub use sspower::sspower_eval;
pub use ssweibull::ssweibull_eval;

pub use fit::{fit_nlmer, NlmerOptions, NlmmStart};
pub use formula::{parse_nlmer_custom_formula, parse_nlmer_formula, NlmerFormula, NlmmMeanKind};
pub use mean_fn::{builtin_mean, CustomNlmmMean, NlmmMeanEval};

use std::sync::Arc;

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
/// response ~ SSmicmen(covariate, Vmax, K) ~ Vmax|group
/// response ~ SSgompertz(covariate, Asym, b2, b3) ~ Asym|group
/// response ~ SSpower(covariate, a, b, c) ~ c|group
/// response ~ SSfpl(covariate, A, B, xmid, scal) ~ A|group
/// response ~ SSbiexp(covariate, A1, lrc1, A2, lrc2) ~ A1|group
/// response ~ SSweibull(covariate, Asym, Drop, lrc, pwr) ~ Asym|group
/// ```
///
/// Supported means: `SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`, `SSpower`,
/// `SSfpl`, `SSbiexp`, `SSweibull`.
/// When `start` is empty, R-style data-driven starting heuristics are used with
/// multistart fallback to static defaults.
/// Random effects are additive on the named nonlinear parameters (one or more
/// per grouping factor).
pub fn nlmer(
    formula: &str,
    data: &DataFrame,
    start: NlmmStart,
    reml: bool,
) -> crate::Result<LmeFit> {
    let opts = NlmerOptions {
        reml,
        start,
        ..NlmerOptions::default()
    };
    nlmer_with_options(formula, data, &opts)
}

/// Fit a nonlinear mixed model from a formula string and [`NlmerOptions`].
pub fn nlmer_with_options(
    formula: &str,
    data: &DataFrame,
    opts: &NlmerOptions,
) -> crate::Result<LmeFit> {
    let (parsed, kind) = parse_nlmer_formula(formula)?;
    fit_nlmer(&parsed, builtin_mean(kind), data, formula, opts)
}

/// Fit a nonlinear mixed model with a custom mean evaluator.
///
/// Build [`NlmerFormula`] programmatically and pass any type implementing
/// [`NlmmMeanEval`] (see [`CustomNlmmMean`]).
pub fn nlmer_with_mean(
    parsed: &NlmerFormula,
    mean: Arc<dyn NlmmMeanEval>,
    data: &DataFrame,
    formula_label: Option<&str>,
    opts: &NlmerOptions,
) -> crate::Result<LmeFit> {
    let label = formula_label.unwrap_or("(custom nlmm mean)");
    fit_nlmer(parsed, mean, data, label, opts)
}
