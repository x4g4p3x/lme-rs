# lme-rs User Guide

A comprehensive guide to using `lme-rs` for fitting linear and generalized linear mixed-effects models in Rust.

## Table of Contents

- [Getting Started](#getting-started)
- [Linear Mixed Models (LMMs)](#linear-mixed-models-lmms)
- [Generalized Linear Mixed Models (GLMMs)](#generalized-linear-mixed-models-glmms)
- [Predictions](#predictions)
- [Complex Random Effects](#complex-random-effects)
- [Diagnostics & Inference](#diagnostics--inference)
- [Model Comparison (ANOVA)](#model-comparison-anova)
- [API Reference](#api-reference)

---

## Getting Started

### Installation

Add `lme-rs` to your `Cargo.toml`:

```toml
[dependencies]
lme-rs = "0.1.0"
polars = { version = "0.46", features = ["csv"] }
anyhow = "1"
```

### Loading Data

`lme-rs` uses [Polars](https://pola.rs) DataFrames as its data input:

```rust
use polars::prelude::*;

let mut file = std::fs::File::open("data.csv")?;
let df = CsvReadOptions::default()
    .with_has_header(true)
    .into_reader_with_file_handle(&mut file)
    .finish()?;
```

### Your First Model

```rust
use lme_rs::lmer;

// Fit: Reaction ~ Days + (Days | Subject)
let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;

// Print R-style summary
println!("{}", fit);
```

The third argument controls REML vs ML estimation:
- `true` = Restricted Maximum Likelihood (REML) — default, better for variance estimation
- `false` = Maximum Likelihood (ML) — needed for model comparison via `anova()`

---

## Linear Mixed Models (LMMs)

### `lmer()` — Standard LMM

```rust
use lme_rs::lmer;

// Random intercept only
let fit = lmer("Yield ~ 1 + (1 | Batch)", &df, true)?;

// Random intercept + slope
let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;

// ML estimation (for model comparison)
let fit_ml = lmer("Reaction ~ Days + (Days | Subject)", &df, false)?;
```

### `lmer_weighted()` — Weighted LMM

Apply prior observation weights:

```rust
use lme_rs::lmer_weighted;
use ndarray::Array1;

let weights = Array1::from_vec(vec![1.0; df.height()]);
let fit = lmer_weighted("y ~ x + (1 | group)", &df, true, Some(weights))?;
```

### `lm()` — Ordinary Least Squares

For simple fixed-effects-only models:

```rust
use lme_rs::lm;
use ndarray::{array, Array1, Array2};

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

### Understanding the Output

The `Display` implementation mirrors R's `summary()`:

```text
Linear mixed model fit by REML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)

     AIC      BIC   logLik deviance
  1755.6   1774.8   -871.8   1743.6
REML criterion at convergence: 1743.6283
Scaled residuals:
    Min      1Q  Median      3Q     Max
-3.9536 -0.4628  0.0296  0.4659  5.1793

Random effects:
 Groups   Name        Variance Std.Dev.
 Subject  (Intercept) 611.9033 24.7367
          Days        35.0801  5.9228
 Corr:
  Days  0.066
 Residual             654.9417 25.5918
Number of obs: 180, groups: Subject, 18

Fixed effects:
            Estimate Std. Error t value
(Intercept) 251.4051     6.8238   36.84
Days         10.4673     1.5459    6.77
```

### Accessing Model Components

```rust
// Fixed effects coefficients (β)
let beta = &fit.coefficients;

// Variance of residuals (σ²)
let sigma2 = fit.sigma2.unwrap();

// Random effects (b)
let b = fit.b.as_ref().unwrap();

// Optimized theta (relative covariance parameters)
let theta = fit.theta.as_ref().unwrap();

// Standard errors and t-values
let se = fit.beta_se.as_ref().unwrap();
let t_vals = fit.beta_t.as_ref().unwrap();

// Model fit statistics
let aic = fit.aic.unwrap();
let bic = fit.bic.unwrap();
let deviance = fit.deviance.unwrap();
let loglik = fit.log_likelihood.unwrap();

// Convergence info
let converged = fit.converged.unwrap();
let iterations = fit.iterations.unwrap();
```

---

## Generalized Linear Mixed Models (GLMMs)

### `glmer()` — GLMM with Family/Link

```rust
use lme_rs::{glmer, family::Family};

// Poisson GLMM (count data, log link)
let fit = glmer("TICKS ~ YEAR + HEIGHT + (1 | BROOD)", &df, Family::Poisson)?;

// Binomial GLMM (binary data, logit link)
let fit = glmer("y ~ x1 + x2 + (1 | group)", &df, Family::Binomial)?;

// Gamma GLMM (positive continuous data, inverse link)
let fit = glmer("cost ~ age + (1 | clinic)", &df, Family::Gamma)?;
```

### Available Families

| Family | Link (canonical) | Use Case |
|:-------|:-----------------|:---------|
| `Family::Binomial` | Logit | Binary outcomes, proportions |
| `Family::Poisson` | Log | Count data |
| `Family::Gaussian` | Identity | Continuous data (equivalent to `lmer`) |
| `Family::Gamma` | Inverse | Positive continuous with variance ∝ μ² |

### GLMM Output

```text
Generalized linear mixed model fit by ML (Laplace) ['glmerMod']
 Family: poisson ( log )
Formula: TICKS ~ YEAR + HEIGHT + (1 | BROOD)

     AIC      BIC   logLik deviance
  1108.1   1124.1   -550.1   1100.1

Random effects:
 Groups   Name        Variance Std.Dev.
 BROOD    (Intercept) 1.5547   1.2469
Number of obs: 403, groups: BROOD, 118

Fixed effects:
            Estimate Std. Error z value
(Intercept)  55.3223    15.8417    3.49
YEAR         -0.4538     0.1630   -2.78
HEIGHT       -0.0239     0.0037   -6.47
```

> **Note**: GLMM fixed effects report **z-values** (not t-values) since the Laplace approximation uses the normal distribution.

> **AIC convention**: `lme-rs` computes AIC from the Laplace-approximated conditional deviance used for optimization. R's `logLik()`/`AIC()` additionally includes data-dependent constants (e.g., `lgamma(y+1)` for Poisson). This means **absolute AIC values differ**, but model fit parameters (β, θ) are identical. See the footnote in [COMPARISONS.md](examples/COMPARISONS.md) for details.

---

## Predictions

### Population-Level Predictions (`re.form=NA`)

Predictions using only fixed effects (Xβ), ignoring random effects:

```rust
let newdata = DataFrame::new(vec![
    Series::new("Days".into(), &[0.0, 1.0, 5.0, 10.0]).into(),
    Series::new("Subject".into(), &["308", "308", "308", "308"]).into(),
])?;

let preds = fit.predict(&newdata)?;
```

### Conditional Predictions (`re.form=NULL`)

Predictions including subject-specific random effects (Xβ + Zb):

```rust
let preds = fit.predict_conditional(&newdata, false)?;
```

The `allow_new_levels` parameter controls behavior for unknown groups:
- `false` — Returns an error if a group not seen during training appears
- `true` — Unknown groups receive zero random-effect contributions (population-level)

```rust
// Predict for a group that wasn't in the training data
let new_subject_data = DataFrame::new(vec![
    Series::new("Days".into(), &[3.0]).into(),
    Series::new("Subject".into(), &["NEW_SUBJECT"]).into(),
])?;

// This would error with allow_new_levels=false:
let preds = fit.predict_conditional(&new_subject_data, true)?;
```

### Response-Scale Predictions (GLMMs)

For GLMMs, predictions are on the linear predictor scale by default. Use `predict_response()` to transform to the response scale:

```rust
// Linear predictor (log-counts for Poisson)
let eta = fit.predict(&newdata)?;

// Response scale (actual expected counts)
let counts = fit.predict_response(&newdata)?;

// Conditional response scale (with random effects)
let cond_counts = fit.predict_conditional_response(&newdata, true)?;
```

---

## Complex Random Effects

### Wilkinson Formula Syntax

`lme-rs` supports R-style Wilkinson notation:

| Formula | Meaning |
|:--------|:--------|
| `(1 \| group)` | Random intercept per group |
| `(x \| group)` | Random intercept + random slope for `x`, correlated |
| `(1 \| a) + (1 \| b)` | Crossed random effects: independent intercepts for `a` and `b` |
| `(1 \| a/b)` | Nested random effects: expands to `(1 \| a) + (1 \| a:b)` |

### Nested Random Effects

```rust
// Nested: cask within batch
let fit = lmer("strength ~ 1 + (1 | batch/cask)", &df, true)?;
// Equivalent to: strength ~ 1 + (1 | batch) + (1 | batch:cask)
```

### Crossed Random Effects

```rust
// Two independent random intercepts
let fit = lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", &df, true)?;
```

---

## Diagnostics & Inference

### Confidence Intervals (`confint`)

Wald confidence intervals for fixed effects:

```rust
let ci = fit.confint(0.95)?;  // 95% CI
println!("{}", ci);
// Output:
//                    2.5 %       97.5 %
//  (Intercept)    238.0290     264.7812
//  Days             7.4374      13.4972
```

### Parametric Bootstrap (`simulate`)

Generate simulated response vectors:

```rust
let sims = fit.simulate(1000)?;
for sim in &sims.simulations {
    // Each `sim` is an Array1<f64> of length n_obs
    println!("mean = {:.2}", sim.mean().unwrap());
}
```

### Satterthwaite Degrees of Freedom

Approximate p-values for fixed effects using Satterthwaite's method (matches R's `lmerTest`):

```rust
let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
fit.with_satterthwaite(&df)?;
println!("{}", fit);
// Fixed effects now show df and p-values:
//             Estimate Std. Error       df t value Pr(>|t|) [Satterthwaite]
// (Intercept) 251.4051     6.8238    17.00   36.84   0.0000
// Days         10.4673     1.5459    17.00    6.77   0.0000
```

### Kenward-Roger Degrees of Freedom

A more conservative alternative to Satterthwaite:

```rust
fit.with_kenward_roger(&df)?;
println!("{}", fit);
```

### Robust Standard Errors (Sandwich Estimator)

```rust
// Observation-level robust SE (HC0)
fit.with_robust_se(&df, None)?;

// Cluster-robust SE (clustered by Subject)
fit.with_robust_se(&df, Some("Subject"))?;

println!("{}", fit);
// Fixed effects now show robust SE:
//             Estimate Std. Error t value Pr(>|t|) [Robust]
```

### Type III ANOVA Table

Requires Satterthwaite or Kenward-Roger to be computed first:

```rust
use lme_rs::DdfMethod;

fit.with_satterthwaite(&df)?;
let anova_table = fit.anova(DdfMethod::Satterthwaite)?;
println!("{}", anova_table);
// Type III Analysis of Variance Table with Satterthwaite's method
// Term              NumDF     DenDF   F value    Pr(>F)
// Days                  1   17.0000   45.8533  3.26e-06 ***
```

---

## Model Comparison (ANOVA)

Use `anova()` for Likelihood Ratio Tests (LRT) between nested models. Both models **must** be fit with `reml=false` (ML):

```rust
use lme_rs::{lmer, anova};

let fit0 = lmer("Reaction ~ 1 + (1 | Subject)", &df, false)?;
let fit1 = lmer("Reaction ~ Days + (Days | Subject)", &df, false)?;

let lrt = anova(&fit0, &fit1)?;
println!("{}", lrt);
// Data: n = 180
//
// Models:
//   H0: Reaction ~ 1 + (1 | Subject)
//   H1: Reaction ~ Days + (Days | Subject)
//
//     npar  deviance  Chisq  Df  Pr(>Chisq)
// H0     3   1802.1
// H1     6   1751.9  50.184   3   7.16e-11 ***
```

---

## API Reference

### Core Functions

| Function | Description |
|:---------|:------------|
| `lm(y, x)` | OLS regression: y = Xβ + ε |
| `lmer(formula, data, reml)` | Linear mixed model |
| `lmer_weighted(formula, data, reml, weights)` | Weighted LMM |
| `glmer(formula, data, family)` | Generalized LMM |
| `anova(fit_a, fit_b)` | Likelihood ratio test between nested models |

### `LmeFit` Methods

| Method | Description |
|:-------|:------------|
| `predict(newdata)` | Population-level prediction (Xβ) |
| `predict_conditional(newdata, allow_new_levels)` | Conditional prediction (Xβ + Zb) |
| `predict_response(newdata)` | Population prediction on response scale (GLMM) |
| `predict_conditional_response(newdata, allow_new_levels)` | Conditional prediction on response scale |
| `confint(level)` | Wald confidence intervals for fixed effects |
| `simulate(nsim)` | Parametric bootstrap simulation |
| `with_satterthwaite(data)` | Compute Satterthwaite df and p-values |
| `with_kenward_roger(data)` | Compute Kenward-Roger df and p-values |
| `with_robust_se(data, cluster_col)` | Compute robust sandwich standard errors |
| `anova(ddf_method)` | Type III ANOVA table for fixed effects |

### `LmeFit` Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `coefficients` | `Array1<f64>` | Fixed effects (β) |
| `residuals` | `Array1<f64>` | Residuals (y − ŷ) |
| `fitted` | `Array1<f64>` | Fitted values (ŷ) |
| `sigma2` | `Option<f64>` | Residual variance (σ²) |
| `theta` | `Option<Array1<f64>>` | Relative covariance parameters |
| `b` | `Option<Array1<f64>>` | Random effects (b = Λu) |
| `reml` | `Option<f64>` | REML criterion (None for ML fits) |
| `deviance` | `Option<f64>` | Model deviance |
| `aic` / `bic` | `Option<f64>` | Information criteria |
| `log_likelihood` | `Option<f64>` | Log-likelihood |
| `beta_se` | `Option<Array1<f64>>` | Standard errors of β |
| `beta_t` | `Option<Array1<f64>>` | t-values (or z-values for GLMM) |
| `ranef` | `Option<DataFrame>` | Random effects as DataFrame |
| `var_corr` | `Option<DataFrame>` | Variance-correlation as DataFrame |
| `converged` | `Option<bool>` | Whether optimizer converged |
| `iterations` | `Option<u64>` | Number of optimizer iterations |

### Family/Link Combinations

| `Family` Enum | Default Link | Alternative Links (via custom family) |
|:-------------|:-------------|:--------------------------------------|
| `Binomial` | Logit | Probit, Cloglog |
| `Poisson` | Log | Sqrt |
| `Gaussian` | Identity | — |
| `Gamma` | Inverse | Log |

---

## Tips & Best Practices

1. **REML vs ML**: Use REML (default) for reporting variance estimates. Use ML for model comparison with `anova()`.

2. **Formula syntax**: The formula parser supports standard Wilkinson notation. Use `+` for additive terms, `|` for random effects grouping, and `/` for nesting.

3. **Data types**: Ensure numeric columns are `Float64` or `Int64` in your DataFrame. Grouping columns can be `String` or `Utf8`.

4. **Convergence**: If the optimizer doesn't converge (rare), try:
   - Centering/scaling continuous predictors
   - Simplifying the random effects structure
   - Checking for perfect separation (in Binomial models)

5. **Performance**: Large datasets with complex random effects benefit from the sparse matrix architecture — `lme-rs` uses compressed sparse column format and sparse Cholesky decomposition to avoid O(N²) blowouts.
