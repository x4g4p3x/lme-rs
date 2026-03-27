# lme-rs User Guide

This guide covers the practical Rust workflow for fitting linear and generalized linear mixed-effects models with `lme-rs`.

## Table of Contents

- [Getting Started](#getting-started)
- [Data Requirements](#data-requirements)
- [Modeling Workflows](#modeling-workflows)
- [Predictions](#predictions)
- [Inference and Diagnostics](#inference-and-diagnostics)
- [Model Comparison](#model-comparison)
- [Limitations and Compatibility Notes](#limitations-and-compatibility-notes)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [API Surface Summary](#api-surface-summary)

## Getting Started

### Installation

```toml
[dependencies]
lme-rs = "0.1.6"
polars = { version = "0.46", features = ["csv"] }
anyhow = "1"
```

### Loading data

`lme-rs` uses `polars::DataFrame` as its data container.

```rust
use polars::prelude::*;

let mut file = std::fs::File::open("tests/data/sleepstudy.csv")?;
let df = CsvReadOptions::default()
    .with_has_header(true)
    .into_reader_with_file_handle(&mut file)
    .finish()?;
```

### First model

```rust
use lme_rs::lmer;

let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
println!("{}", fit);
```

The `reml` flag controls LMM estimation mode:

- `true`: REML. Prefer this for reporting variance components.
- `false`: ML. Use this when comparing nested models with likelihood ratio tests.

## Data Requirements

`lme-rs` expects a rectangular `polars::DataFrame` with column names that match the variables referenced in the formula.

### Column types

- Response and numeric predictor columns should be numeric (`Float64` or `Int64` are the safe defaults).
- Grouping columns for random effects should be string-like or otherwise castable to strings.
- Prediction data must contain the same fixed-effect and grouping columns implied by the fitted formula.

### Formula expectations

The parser supports the standard R-style Wilkinson syntax used by `lme4` for the covered cases:

- `y ~ x`
- `y ~ x + (1 | group)`
- `y ~ x + (x | group)`
- `y ~ x + (1 | a) + (1 | b)`
- `y ~ x + (1 | a/b)`

### Practical advice

- Keep grouping identifiers stable and explicit. Converting them to strings before modeling is often the least surprising choice.
- Centering or scaling continuous predictors can help optimizer behavior for more complex random-effects structures.
- When building prediction frames, mirror the training column names exactly.

## Modeling Workflows

### `lm()` for fixed-effects-only linear models

```rust
use lme_rs::lm;
use ndarray::array;

let y = array![1.0, 2.0, 3.0, 4.0];
let x = array![
    [1.0, 1.0],
    [1.0, 2.0],
    [1.0, 3.0],
    [1.0, 4.0],
];

let fit = lm(&y, &x)?;
println!("Coefficients: {:?}", fit.coefficients);
```

### `lmer()` for linear mixed models

```rust
use lme_rs::lmer;

let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
```

Common patterns:

```rust
// Random intercept only
let fit = lmer("Yield ~ 1 + (1 | Batch)", &df, true)?;

// Random intercept and slope
let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;

// Nested random effects
let fit = lmer("strength ~ 1 + (1 | batch/cask)", &df, true)?;

// Crossed random effects
let fit = lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", &df, true)?;
```

### `lmer_weighted()` for weighted LMMs

```rust
use lme_rs::lmer_weighted;
use ndarray::Array1;

let weights = Array1::from_vec(vec![1.0; df.height()]);
let fit = lmer_weighted("y ~ x + (1 | group)", &df, true, Some(weights))?;
```

### `glmer()` for generalized linear mixed models

```rust
use lme_rs::{family::Family, glmer};

let poisson_fit = glmer(
    "TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
    &df,
    Family::Poisson,
)?;

let binomial_fit = glmer(
    "y ~ x1 + x2 + (1 | group)",
    &df,
    Family::Binomial,
)?;

let gamma_fit = glmer(
    "cost ~ age + (1 | clinic)",
    &df,
    Family::Gamma,
)?;
```

Built-in family support through the public enum:

| Family | Default link | Typical use |
| :----- | :----------- | :---------- |
| `Family::Binomial` | logit | binary outcomes and proportions |
| `Family::Poisson` | log | count data |
| `Family::Gaussian` | identity | continuous Gaussian responses |
| `Family::Gamma` | inverse | positive continuous responses |

At the public API level, `glmer()` currently dispatches through these built-in families and their default links.

### Understanding model output

The `Display` implementation intentionally mirrors the shape of R output.

```text
Linear mixed model fit by REML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)

     AIC      BIC   logLik deviance
  1755.6   1774.8   -871.8   1743.6
REML criterion at convergence: 1743.6283
...
```

Useful fields on `LmeFit` include:

```rust
let beta = &fit.coefficients;
let sigma2 = fit.sigma2;
let theta = fit.theta.as_ref();
let beta_se = fit.beta_se.as_ref();
let deviance = fit.deviance;
let converged = fit.converged;
let iterations = fit.iterations;
```

## Predictions

### Population-level predictions

`predict()` uses fixed effects only.

```rust
let newdata = DataFrame::new(vec![
    Series::new("Days".into(), &[0.0, 1.0, 5.0, 10.0]).into(),
    Series::new("Subject".into(), &["308", "308", "308", "308"]).into(),
])?;

let preds = fit.predict(&newdata)?;
```

### Conditional predictions

`predict_conditional()` includes stored random effects.

```rust
let preds = fit.predict_conditional(&newdata, false)?;
```

`allow_new_levels` controls how unseen groups are handled:

- `false`: return an error when a new grouping level appears
- `true`: use zero random-effect contribution for unseen groups

### Response-scale GLMM predictions

For GLMMs, `predict()` returns the linear predictor. Use `predict_response()` when you need probabilities, rates, or means on the response scale.

```rust
let eta = poisson_fit.predict(&newdata)?;
let mu = poisson_fit.predict_response(&newdata)?;
let mu_cond = poisson_fit.predict_conditional_response(&newdata, true)?;
```

## Inference and Diagnostics

### Confidence intervals

`confint()` returns Wald intervals for the fixed effects.

```rust
let ci = fit.confint(0.95)?;
println!("{}", ci);
```

### Parametric simulation

`simulate()` draws new response vectors from the fitted model.

```rust
let sims = fit.simulate(100)?;
println!("generated {} simulations", sims.simulations.len());
```

### Satterthwaite degrees of freedom

```rust
let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
fit.with_satterthwaite(&df)?;
println!("{}", fit);
```

Use this when you want approximate denominator degrees of freedom and p-values for LMM fixed effects.

### Kenward-Roger path

```rust
fit.with_kenward_roger(&df)?;
println!("{}", fit);
```

The Kenward-Roger path produces denominator degrees of freedom that match R's `pbkrtest` on the covered LMM configurations. As with Satterthwaite, results are derived via numerical differentiation of the REML objective.

### Robust standard errors

```rust
fit.with_robust_se(&df, None)?;
fit.with_robust_se(&df, Some("Subject"))?;
```

Use the first form for observation-level HC0 robust errors and the second for clustered robust errors.

### Type III ANOVA table

```rust
use lme_rs::DdfMethod;

fit.with_satterthwaite(&df)?;
let table = fit.anova(DdfMethod::Satterthwaite)?;
println!("{}", table);
```

Current scope:

- Type III fixed-effect tables only
- 1-DoF fixed-effect terms only
- Requires a denominator degrees of freedom method to be computed first

## Model Comparison

Use `anova(fit_a, fit_b)` for likelihood ratio tests between nested models. Both models should be fit with `reml = false`.

```rust
use lme_rs::{anova, lmer};

let fit0 = lmer("Reaction ~ 1 + (1 | Subject)", &df, false)?;
let fit1 = lmer("Reaction ~ Days + (Days | Subject)", &df, false)?;

let lrt = anova(&fit0, &fit1)?;
println!("{}", lrt);
```

## Limitations and Compatibility Notes

### Numerical comparisons with R

`lme-rs` is designed to track `lme4` behavior closely on the covered workflows. The strongest evidence for parity is in the repository tests and comparison fixtures, not in a blanket claim that every `lme4` feature is already mirrored.

See [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) for concrete side-by-side output.

### GLMM information criteria

For GLMMs, `lme-rs` computes the optimization target from a Laplace-approximated conditional deviance. R includes additional data-dependent constants in `logLik()` and downstream `AIC()` and `BIC()` reporting. That means absolute information criteria can differ even when the fitted parameters agree.

### ANOVA scope

Fixed-effects ANOVA support currently means Type III tests over the current 1-DoF fixed-effect design. If you need a richer ANOVA design matrix story, treat this as a current limitation rather than as a hidden assumption.

### Kenward-Roger status

`with_kenward_roger()` produces denominator degrees of freedom that match R's `pbkrtest` on the covered LMM configurations. Numerical precision is consistent with the finite-difference Hessian used by the implementation; results have been validated against the `sleepstudy` reference to within 0.01 df. As always, validating against an R reference is sensible for any publication-critical workflow.

### Python bindings

The Python package is intentionally smaller than the Rust crate. If you need the full inference surface, use the Rust API directly.

## Troubleshooting

### Convergence issues

If a mixed model does not converge or reports unstable estimates:

- center or scale continuous predictors
- simplify the random-effects structure
- verify that grouping columns are coded as expected
- look for quasi-complete or complete separation in binomial models
- compare against an `lme4` fit on the same data when debugging a hard case

### Prediction errors

Prediction failures usually come from one of two issues:

- the prediction frame is missing a fixed-effect or grouping column required by the stored formula
- `allow_new_levels` is `false` and the prediction frame contains unseen groups

### Data shape issues

If you see dimension mismatch or underdetermined-system errors, inspect the effective design matrix implied by the formula. These errors often mean that the data does not contain enough information for the requested structure.

## Performance Notes

`lme-rs` uses sparse matrix machinery for grouped random-effects structure, which helps substantially for larger mixed models. Even so, performance depends heavily on:

- the number of observations
- the number of grouping levels
- whether you fit random slopes as well as intercepts
- how well-scaled the predictors are for optimization

For concrete parity outputs, use the scripts and datasets in `comparisons/` and `tests/data/`.

## API Surface Summary

### Free functions

| Function | Description |
| :------- | :---------- |
| `lm(y, x)` | fixed-effects-only linear regression |
| `lmer(formula, data, reml)` | linear mixed model |
| `lmer_weighted(formula, data, reml, weights)` | weighted linear mixed model |
| `glmer(formula, data, family)` | generalized linear mixed model |
| `anova(fit_a, fit_b)` | likelihood ratio test between nested models |

### `LmeFit` methods

| Method | Description |
| :----- | :---------- |
| `predict(newdata)` | fixed-effects-only prediction |
| `predict_conditional(newdata, allow_new_levels)` | prediction with stored random effects |
| `predict_response(newdata)` | GLMM response-scale prediction |
| `predict_conditional_response(newdata, allow_new_levels)` | conditional response-scale GLMM prediction |
| `confint(level)` | Wald confidence intervals |
| `simulate(nsim)` | parametric simulation |
| `with_satterthwaite(data)` | Satterthwaite degrees of freedom and p-values |
| `with_kenward_roger(data)` | Kenward-Roger denominator degrees of freedom and p-values |
| `with_robust_se(data, cluster_col)` | robust or cluster-robust standard errors |
| `anova(ddf_method)` | Type III fixed-effects ANOVA table |

### Selected fields on `LmeFit`

| Field | Meaning |
| :---- | :------ |
| `coefficients` | fixed-effect coefficients |
| `residuals` | model residuals |
| `fitted` | fitted values |
| `sigma2` | residual variance when defined |
| `theta` | relative covariance parameters |
| `b` | estimated random effects |
| `ranef` | random effects as a `DataFrame` |
| `var_corr` | variance-correlation summary as a `DataFrame` |
| `aic`, `bic`, `log_likelihood`, `deviance` | fit statistics |
| `converged`, `iterations` | optimizer diagnostics |
