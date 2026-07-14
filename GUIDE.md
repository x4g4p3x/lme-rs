# lme-rs User Guide

This guide covers the practical Rust workflow for fitting linear and generalized linear mixed-effects models with `lme-rs`.

## Table of Contents

- [Getting Started](#getting-started)
- [Data Requirements](#data-requirements)
- [Modeling Workflows](#modeling-workflows)
- [Predictions](#predictions)
- [Inference and Diagnostics](#inference-and-diagnostics)
- [Model Comparison](#model-comparison)
- [Repeated Fits and Cross-Validation](#repeated-fits-and-cross-validation)
- [Limitations and Compatibility Notes](#limitations-and-compatibility-notes)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)
- [API Surface Summary](#api-surface-summary)

## Getting Started

### Installation

```toml
[dependencies]
lme-rs = "0.1.9"
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

### `nlmer()` for nonlinear mixed models

`nlmer` fits Gaussian nonlinear mixed models with a three-part formula (response, nonlinear mean, random part):

```text
circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree
circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree
y ~ SSasymp(x, Asym, R0, lrc) ~ Asym|id
y ~ SSfol(x, Asym, R0, lrc) ~ Asym|id
y ~ SSmicmen(x, Vmax, K) ~ Vmax|id
y ~ SSgompertz(x, Asym, b2, b3) ~ Asym|id
y ~ SSpower(x, a, b, c) ~ c|id   # MATLAB Curve Fitter power2: a*x^b + c
```

```rust
use lme_rs::nlmer;
use std::collections::HashMap;

let mut start = HashMap::new();
start.insert("Asym".to_string(), 200.0);
start.insert("xmid".to_string(), 725.0);
start.insert("scal".to_string(), 350.0);

// ML (matches R `nlmer` default); pass `true` for REML profiling
let fit = nlmer(
    "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
    &df,
    start,
    false,
)?;
```

For scalar adaptive quadrature on a single random effect (`k = 1`), use [`nlmer_with_options`](src/nlmm/mod.rs) and set [`NlmerOptions::n_agq`](src/nlmm/fit.rs) to a value `≥ 2` (default `1` is Laplace only). Python: `nlmer(..., n_agq=7)`.

Custom mean functions implement [`NlmmMeanEval`](src/nlmm/mean_fn.rs) or wrap a closure with [`CustomNlmmMean`](src/nlmm/mean_fn.rs), then call [`nlmer_with_mean`](src/nlmm/mod.rs). In Python, pass a callable to [`nlmer_with_mean`](python/src/lib.rs) with the `response ~ covariate ~ re | group` formula layout (see [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)).

Current limitations:

- Built-in mean functions: `SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`, `SSpower` (`a * x^b + c`, MATLAB `power2`; not in R `stats::SS*`). User-defined means via [`nlmer_with_mean`](src/nlmm/mod.rs) / [`CustomNlmmMean`](src/nlmm/mean_fn.rs) in Rust, or `lme_python.nlmer_with_mean` in Python.
- Starting values: pass an empty `NlmmStart` / `start=None` in Python to use R-style `selfStart` heuristics (`stats::getInitial`); the fitter also tries static defaults and keeps the lowest-deviance result.
- Random effects: one grouping factor; multiple parameters before `|` use a multivariate Cholesky covariance (`Asym + xmid | Tree`). θ matches `lme4::getME(., "theta")` (relative Λ; VarCorr SDs are reported through `σ²ΛΛᵀ`). Orange scalar and correlated multi-RE fits, plus `SSasymp` / `SSfol` / `SSmicmen` / `SSgompertz` / **`SSpower`**, are covered by lme4 parity tests (`SSpower` via custom R `selfStart`; see [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md)).
- **`SSpower`:** μ = `a * x^b + c` (MATLAB Curve Fitter `power2`). Requires **covariate x > 0**. Not in R `stats::SS*`; grouped calibration only — not bounded single-curve NLS (lmfit / MATLAB Curve Fitter). For **independent per-sensor fits** vs **pooled** `nlmer`, and why CUDA batch fitters are a different lane, see [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md).
- `predict()` evaluates the mean at fixed parameters only (`re.form = NA`); `predict_conditional()` adds stored random effects (`re.form = NULL`).
- Scalar AGQ (`n_agq ≥ 2`, `k = 1` RE) is applied in the final profile evaluation at the optimized θ, not inside the θ search (same pattern as `glmer`). Default `n_agq = 1` is Laplace / penalized Gauss–Newton (`nAGQ = 0` style).

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

`glmer()` uses each family's canonical (default) link. Non-canonical links are available via
[`glmer_with_link`](src/lib.rs) and [`family::Link`](src/family.rs):

| Family | Default link | Also supported |
| :----- | :----------- | :------------- |
| `Family::Binomial` | logit | probit, cloglog |
| `Family::Poisson` | log | identity, sqrt |
| `Family::Gaussian` | identity | log, inverse |
| `Family::Gamma` | inverse | identity, log |

```rust
use lme_rs::{glmer_with_link, family::{Family, Link}};

let probit_fit = glmer_with_link(
    "y ~ period2 + period3 + period4 + (1 | herd)",
    &df,
    Family::Binomial,
    Link::Probit,
    1,
)?;
```

### `glmer_weighted()` for prior observation weights

```rust
use lme_rs::{family::Family, glmer_weighted};
use ndarray::Array1;

let w = Array1::from_vec(vec![1.0; df.height()]);
let fit = glmer_weighted(
    "y ~ period2 + period3 + period4 + (1 | herd)",
    &df,
    Family::Binomial,
    1,
    Some(w),
)?;
```

Weights must be strictly positive and match the number of rows. For `Family::Gaussian`, fitting delegates to `lmer_weighted` with ML.

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

`confint()` returns Wald intervals for the fixed effects. When Kenward–Roger or Satterthwaite denominator degrees of freedom are stored on the fit (via `with_kenward_roger()` or `with_satterthwaite()`), intervals use **t** critical values with those dfs; otherwise the normal approximation is used.

```rust
let mut fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
fit.with_satterthwaite(&df)?;
let ci = fit.confint(0.95)?;
println!("{}", ci);
```

### Parametric simulation

`simulate()` draws new response vectors from the fitted conditional means. It does **not** refit the model. For bootstrap inference (simulate or resample → refit → summarize), use [`boot_lmer`](#bootstrap-refits-boot_lmer) instead.

```rust
// All logical CPUs by default when n_jobs is None
let sims = fit.simulate_with(10_000, None, Some(42))?;
println!("generated {} simulations", sims.simulations.len());

// Stream batches without holding all draws in memory
fit.simulate_batched(50_000, 1_000, Some(4), Some(42), |batch_idx, batch| {
    println!("batch {batch_idx}: {} draws", batch.len());
    Ok(())
})?;
```

### Bootstrap refits (`boot_lmer`)

[`boot_lmer`](src/bootstrap.rs) mirrors R's `lme4::bootMer` workflow for **Gaussian LMMs**: draw bootstrap responses, refit on each replicate, and summarize fixed-effect (and variance-component) draws. Design-matrix setup is amortized with [`prepare_lmer`](#amortized-fitting-prepare_lmer--fit_prepared); each replicate calls [`fit_prepared_with_response`](src/lib.rs) to swap the response vector without rebuilding `X` / `Z`.

```rust
use lme_rs::{boot_lmer, lmer, BootLmerMethod};

let formula = "Reaction ~ Days + (1 | Subject)";
let fit = lmer(formula, &df, true)?;

let boot = boot_lmer(
    formula,
    &df,
    &fit,
    500,
    BootLmerMethod::Parametric, // or BootLmerMethod::Residual
    true,                       // reml for each refit
    Some(42),                   // optional seed (reproducible across n_jobs)
    None,                       // n_jobs: None = all CPUs, Some(1) = sequential
)?;

println!("converged: {:.1}%", 100.0 * boot.prop_converged);
let ci = boot.confint_percentile(0.95)?;
println!("{}", ci);

// Convenience on the fit:
let boot = fit.boot(formula, &df, 500, BootLmerMethod::Parametric, true, Some(42), None)?;
```

| Method | Resampling | R analogue |
|:-------|:-----------|:-----------|
| `BootLmerMethod::Parametric` | New Gaussian responses from fitted conditional means + σ² | `bootMer(..., type = "parametric")` |
| `BootLmerMethod::Residual` | Fitted values + resampled residuals (with replacement) | `bootMer(..., type = "residual")` |

**Result fields:** `t0` (original fixed effects), `replicates` (per-draw `coefficients`, `theta`, `sigma2`, `converged`), `prop_converged`, and `confint_percentile(level)` for percentile intervals on converged draws.

**Scope:** LMMs only (not GLMM/NLMM). Requires the same `formula` and `DataFrame` used for the reference fit. Does not resample fixed-effect or variance-component parameters directly (parametric draws are on the **response** scale, then the model is refit). Validate against R `bootMer` on publication-critical workflows.

**Parallelism:** same `n_jobs` semantics as [`cv_grouped`](#group-structure-preserving-cv-cv_grouped); BLAS/OpenMP pinned to one thread per worker when `n_jobs > 1`.

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

### Fixed-effects ANOVA (Type I / II / III)

```rust
use lme_rs::{AnovaType, DdfMethod};

fit.with_satterthwaite(&df)?;
let table = fit.anova(DdfMethod::Satterthwaite)?; // Type III (default)
let table_ii = fit.anova_typed(AnovaType::Type2, DdfMethod::Satterthwaite)?;
let table_i = fit.anova_typed(AnovaType::Type1, DdfMethod::Satterthwaite)?;
let cask_test = fit.linear_hypothesis("cask", DdfMethod::Satterthwaite)?;
println!("{}", table);
```

Current scope:

- Type **I**, **II**, and **III** tables (`anova_typed` / `AnovaType`)
- `linear_hypothesis(term)` for single-term Wald tests (`car::linearHypothesis`-style)
- 1-DoF marginal tests for continuous predictors; joint multi-DoF Wald F-tests for categorical predictors when dummy columns are grouped in the fit
- Multi-DoF Satterthwaite denominator df follows **`lmerTest::contestMD()`** (orthogonal contrast directions + `get_Fstat_ddf()` pooling); call `with_satterthwaite()` before `anova()` so the vcov Jacobian is available
- Kenward–Roger multi-DoF rows use `pbkrtest::KRmodcomp` / `.KR_adjust` via [`kr_modcomp`](src/kr_modcomp.rs) and `vcovAdj16` via [`kr_vcov_adj`](src/kr_vcov_adj.rs); when `vcovAdj` ≈ `vcov`, DenDF matches marginal KR pooling (pastes `cask` reference)
- Golden regression for categorical multi-DoF ANOVA: manifest case `pastes_cask_multi_dof_reml` in [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) (pastes / `cask`)
- **User-defined contrasts:** any **q × p** matrix `L` via [`LmeFit::test_contrast`](src/contrast.rs) / [`test_contrast_vs`](src/contrast.rs) (`lmerTest::contestMD` / `KRmodcomp`); helpers [`contrast_matrix`](src/contrast.rs) and [`contrast_matrix_from_names`](src/contrast.rs). Python: `fit.test_contrast(l_matrix, ddf_method=...)`.

## Model Comparison

Use `anova(fit_a, fit_b)` for likelihood ratio tests between nested models. Both models should be fit with `reml = false`.

```rust
use lme_rs::{anova, lmer};

let fit0 = lmer("Reaction ~ 1 + (1 | Subject)", &df, false)?;
let fit1 = lmer("Reaction ~ Days + (Days | Subject)", &df, false)?;

let lrt = anova(&fit0, &fit1)?;
println!("{}", lrt);
```

## Repeated Fits and Cross-Validation

### Amortized fitting (`prepare_lmer` / `fit_prepared`)

When you fit the **same formula and data** many times (grid search over REML vs ML, hyperparameter tuning, bootstrap replicates on fixed data), build the design matrices once and reuse them:

```rust
use lme_rs::{fit_prepared, prepare_lmer};

let prepared = prepare_lmer("Reaction ~ Days + (1 | Subject)", &df)?;
let fit_reml = fit_prepared(&prepared, true)?;
let fit_ml = fit_prepared(&prepared, false)?;
```

[`LmerPrepared`](src/lib.rs) caches `LmmData`, the fixed/random design matrices, and blocked-kernel metadata. Hot `fit_prepared` wall time on fair tier-A crossed fixtures matches or beats MixedModels.jl; see [OPTIMIZATION.md](OPTIMIZATION.md).

[`refit_lmer`](src/cv.rs) is a convenience wrapper that calls `prepare_lmer` followed by `fit_prepared` when you do not need to hold the prepared object.

### Group-structure-preserving CV (`cv_grouped`)

[`cv_grouped`](src/cv.rs) performs k-fold cross-validation that **splits by grouping units** (e.g. all rows for one `Subject` stay in train or test). This avoids the common mistake of row-wise k-fold on clustered data, which leaks information across folds.

```rust
use lme_rs::cv_grouped;

let cv = cv_grouped(
    "Reaction ~ Days + (1 | Subject)",
    &df,
    "Subject",   // column whose levels define folds
    5,           // n_splits (must be ≤ number of unique groups)
    true,        // reml
    Some(42),    // optional seed for reproducible group shuffling
    None,        // n_jobs: None = all CPUs, Some(1) = sequential
)?;

println!("OOF RMSE: {:.2}", cv.rmse);
// cv.oof_predictions — one population-level prediction per observation
// cv.test_fold       — which fold each row was held out in (0 .. n_splits-1)
// cv.folds           — per-fold RMSE, MAE, convergence, group/obs counts
```

**Prediction semantics on held-out groups:** test subjects were not in the training fold, so each out-of-fold prediction uses population-level fixed-effects prediction (`LmeFit::predict`), i.e. no subject-specific random effect.

**Parallelism:** pass `n_jobs: Some(k)` to fit up to `k` folds concurrently (`None` uses all logical CPUs, capped at fold count). When `n_jobs > 1`, OpenBLAS/MKL/OpenMP are pinned to one thread per worker to avoid CPU oversubscription.

**Scope:** LMMs only (not GLMM/NLMM). Requires `n_splits ≥ 2` and `n_splits` ≤ number of unique levels in `group_col`. Each training fold is fit via `prepare_lmer` + `fit_prepared`.

### Custom parallel refits (grids, manual bootstrap)

For refits that **do not** follow the standard `bootMer` response-resampling pattern, amortize setup with `prepare_lmer` and parallelize across replicates with `rayon` or a thread pool. To swap only the response vector on prepared data, use [`fit_prepared_with_response`](src/lib.rs):

```rust
use lme_rs::{fit_prepared, fit_prepared_with_response, prepare_lmer};
use rayon::prelude::*;

let prepared = prepare_lmer("Reaction ~ Days + (1 | Subject)", &df)?;
let y_boot = /* custom response vector */;
let fit = fit_prepared_with_response(&prepared, Some(y_boot), true)?;
```

For identical refits (e.g. REML vs ML grid on fixed data):

```rust
use lme_rs::{fit_prepared, prepare_lmer};
use rayon::prelude::*;

let prepared = prepare_lmer("Reaction ~ Days + (1 | Subject)", &df)?;
let fits: Vec<_> = (0..100)
    .into_par_iter()
    .map(|_| fit_prepared(&prepared, true))
    .collect::<Result<_, _>>()?;
```

Set `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` when combining outer parallelism with BLAS-backed linear algebra.

## Limitations and Compatibility Notes

### Numerical comparisons with R

`lme-rs` is designed to track `lme4` behavior closely on the covered workflows. The strongest evidence for parity is in the repository tests and comparison fixtures, not in a blanket claim that every `lme4` feature is already mirrored.

See [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) for concrete side-by-side output.

### GLMM information criteria

For GLMMs, `lme-rs` computes the optimization target from a Laplace-approximated conditional deviance. R includes additional data-dependent constants in `logLik()` and downstream `AIC()` and `BIC()` reporting. That means absolute information criteria can differ even when the fitted parameters agree.

### ANOVA scope

Fixed-effects ANOVA supports Type **I**, **II**, and **III** (`AnovaType`). Type II uses `lmerTest`-style contrasts (marginal for non-contained terms, Doolittle reordering for contained terms). Continuous terms use 1-DoF tests where applicable; categorical predictors use joint multi-DoF Wald rows. Arbitrary **q × p** contrast matrices are supported via [`test_contrast`](src/contrast.rs); named-term tests via [`linear_hypothesis`](src/anova.rs).

### Kenward-Roger status

`with_kenward_roger()` produces denominator degrees of freedom that match R's `pbkrtest` on the covered LMM configurations. Numerical precision is consistent with the finite-difference Hessian used by the implementation; results have been validated against the `sleepstudy` reference to within 0.01 df. As always, validating against an R reference is sensible for any publication-critical workflow.

### Python bindings

The Python package mirrors most of the Rust formula API, including `prepare_lmer`, `fit_prepared`, `refit_lmer`, `cv_grouped`, and `boot_lmer`. Matrix-only `lm(y, x)` without a DataFrame remains Rust-only. See [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md).

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

`lme-rs` uses sparse matrix machinery for grouped random-effects structure, which helps substantially for larger mixed models. The θ optimizer reuses a single [`LmmData`](src/math.rs) (precomputed `Z^T Z`, `Z^T X`, `Z^T y`) across Nelder–Mead evaluations; intercept-only random-effects models use a diagonal-Λ fast path that scales `Z^T Z` directly instead of rebuilding sparse triple products each step.

Even with those optimizations, performance depends heavily on:

- the number of observations
- the number of grouping levels
- whether you fit random slopes as well as intercepts
- how well-scaled the predictors are for optimization

Fair fit-only timings vs MixedModels.jl are documented in [BENCHMARKS.md](BENCHMARKS.md#fair-rust-vs-julia-reference-results) (reference workstation medians as of 2026-07-06).

For concrete parity outputs, use the scripts and datasets in `comparisons/` and `tests/data/`.

## API Surface Summary

### Free functions

| Function | Description |
| :------- | :---------- |
| `lm(y, x)` | fixed-effects-only linear regression |
| `lmer(formula, data, reml)` | linear mixed model |
| `prepare_lmer(formula, data)` | cache design matrices for repeated fits |
| `fit_prepared(prepared, reml)` | fit from a prior `prepare_lmer` call |
| `refit_lmer(formula, data, reml)` | `prepare_lmer` + `fit_prepared` convenience |
| `cv_grouped(formula, data, group_col, n_splits, reml, seed, n_jobs)` | group-preserving k-fold CV (LMM); parallel folds when `n_jobs > 1` |
| `boot_lmer(formula, data, fit, nsim, method, reml, seed, n_jobs)` | parametric/residual bootstrap refits (LMM); percentile CIs on `BootLmerResult` |
| `fit_prepared_with_response(prepared, y_response, reml)` | refit from prepared design with a new response vector |
| `lmer_weighted(formula, data, reml, weights)` | weighted linear mixed model |
| `nlmer(formula, data, start, reml)` | nonlinear mixed model (`SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`, `SSpower`; multivariate RE; empty `start` → `selfStart`) |
| `nlmer_with_options(formula, data, opts)` | `nlmer` with [`NlmerOptions`](src/nlmm/fit.rs) (`n_agq`, `max_inner`, …) |
| `nlmer_with_mean(parsed, mean, data, formula_label, opts)` | `nlmer` with a custom [`NlmmMeanEval`](src/nlmm/mean_fn.rs) |
| `glmer(formula, data, family)` | generalized linear mixed model (canonical link) |
| `glmer_with_link(formula, data, family, link)` | GLMM with explicit link |
| `glmer_weighted(formula, data, family, n_agq, weights)` | GLMM with prior observation weights |
| `anova(fit_a, fit_b)` | likelihood ratio test between nested models |

### `LmeFit` methods

| Method | Description |
| :----- | :---------- |
| `predict(newdata)` | fixed-effects-only prediction |
| `predict_conditional(newdata, allow_new_levels)` | prediction with stored random effects |
| `predict_response(newdata)` | GLMM response-scale prediction |
| `predict_conditional_response(newdata, allow_new_levels)` | conditional response-scale GLMM prediction |
| `confint(level)` | Wald CIs; uses t with KR/Satterthwaite dfs when stored on fit |
| `simulate(nsim)` | parametric simulation (sequential; use `simulate_with` for parallel) |
| `simulate_with(nsim, n_jobs, seed)` | parallel parametric simulation |
| `simulate_batched(nsim, batch_size, n_jobs, seed, on_batch)` | stream simulation batches |
| `boot(formula, data, nsim, method, reml, seed, n_jobs)` | parametric/residual bootstrap refits (`bootMer`-style; LMM only) |
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
