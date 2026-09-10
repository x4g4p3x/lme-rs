Linear, generalized linear, and nonlinear mixed-effects models with
`lme4`-style formulas and Polars DataFrames.

<div class="lme-hero">
<p class="lme-eyebrow">lme-rs · Rust API reference</p>
<p class="lme-title">From formula to fitted model.</p>
<p>Fit a model, inspect its numerical diagnostics, and choose the prediction or
inference method that matches your question.</p>
<nav class="lme-jump" aria-label="Getting started">
<a href="#choose-a-model">Choose a model</a>
<a href="#first-fit">First fit</a>
<a href="#use-the-result">Use the result</a>
</nav>
</div>

# Choose a model

<div class="lme-pathways">
<div class="lme-path">

**Linear regression**

[`lm_df`] fits ordinary least squares from a formula. Use [`lm`] when you already
have response and design arrays.

</div>
<div class="lme-path">

**Linear mixed models**

[`lmer`] fits Gaussian mixed models using REML or maximum likelihood.
[`lmer_weighted`] adds observation precision weights.

</div>
<div class="lme-path">

**Generalized mixed models**

[`glmer`] fits a selected [`family::Family`]. [`glmer_with_link`] selects the link;
[`FitDiagnostics`] records the quadrature order actually used.

</div>
<div class="lme-path">

**Nonlinear mixed models**

[`nlmer`] fits built-in nonlinear mean functions. [`nlmer_with_mean`] accepts a
custom mean; [`NlmerOptions`] configures fitting.

</div>
</div>

# First fit

This small example fits a random intercept for each group. The final `true`
selects REML; use `false` for maximum likelihood.

```
use lme_rs::lmer;
use polars::prelude::*;

# fn main() -> anyhow::Result<()> {
let data = df!(
    "y" => [
        10.0, 12.0, 13.0, 15.0,
         9.0, 11.0, 14.0, 17.0,
         8.0, 10.0, 12.0, 14.0,
    ],
    "x" => [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
            0.0, 1.0, 2.0, 3.0],
    "group" => ["a", "a", "a", "a", "b", "b", "b", "b",
                "c", "c", "c", "c"],
)?;

let fit = lmer("y ~ x + (1 | group)", &data, true)?;
let population_predictions = fit.predict(&data)?;
assert_eq!(population_predictions.len(), data.height());
# Ok(())
# }
```

The result is an [`LmeFit`]. Inspect its coefficients, variance estimates, and
`diagnostics` before interpreting a fit. With [`FitControl`], you can set iteration
limits, supply starting values, and require numerical convergence through
[`fit_prepared_with_control`] or [`fit_prepared_glmer_with_control`].

# Use the result

| Your question | API and interpretation |
| --- | --- |
| Predictions from fixed effects? | [`LmeFit::predict`] omits group effects. For GLMMs it returns the linear predictor, including any offset. |
| Predictions for known groups? | [`LmeFit::predict_conditional`] includes stored group effects. Its `allow_new_levels` option controls the fallback for unseen groups. |
| GLMM response means? | [`LmeFit::predict_response`] and [`LmeFit::predict_conditional_response`] apply the inverse link, with group effects omitted or included respectively. |
| Confidence intervals? | [`LmeFit::confint`] gives Wald intervals. [`profile_ci`] provides profile-likelihood alternatives and documents supported parameters. |
| Fixed-effect hypotheses? | [`contrast`] provides Wald F-tests; [`emmeans`] provides reference-grid marginal means and pairwise comparisons. |
| Compare nested LMMs? | [`anova()`] compares likelihoods. Use the same observations and fit with ML when fixed effects differ; REML comparisons require identical fixed-effects designs. |
| Assess predictions on held-out groups? | [`cv_grouped`] and [`cv_grouped_glmer`] preserve grouping structure when splitting folds. |
| Simulate or bootstrap? | [`LmeFit::simulate`] draws responses from fitted conditional means. [`boot_lmer`] and [`boot_glmer`] perform bootstrap refits. |

# Reuse a design

For repeated fits with the same design, start with [`prepare_lmer`] or
[`prepare_glmer`]. The resulting [`LmerPrepared`] and [`GlmerPrepared`] objects
retain the design and model metadata. Fit with [`fit_prepared`] or
[`fit_prepared_glmer`], or replace only the response with
[`fit_prepared_with_response`] or [`fit_prepared_glmer_with_response`].
[`LmerWorkspace`] provides reusable buffers for repeated LMM responses.

# Scope and versions

<div class="lme-note">

**Read the documentation for the version you use.**

On docs.rs, the version selector identifies the published crate documented on
this page. Changes on the repository's development branch appear here only after
a new crate release. The [development changelog](https://github.com/x4g4p3x/lme-rs/blob/master/CHANGELOG.md)
separates **Unreleased** changes from published releases.

</div>

Numerical compatibility is established for tested models and reference fixtures.
Formula syntax, quadrature, and post-fit inference cover a subset of R's mixed-model
ecosystem; support varies by model and method. See each API's documented limits
and check fit diagnostics, including approximation warnings.

Continue with the [Rust guide](https://github.com/x4g4p3x/lme-rs/blob/master/GUIDE.md),
[documentation index](https://github.com/x4g4p3x/lme-rs/blob/master/docs/README.md), or
[workflow assessment](https://github.com/x4g4p3x/lme-rs/blob/master/USABILITY.md).
These repository links follow `master`; select the matching release tag when
reading about a published version. The API index below belongs to this build.
