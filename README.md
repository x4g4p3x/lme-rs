# lme-rs - mixed-effects models in Rust

[![crates.io](https://img.shields.io/crates/v/lme-rs.svg)](https://crates.io/crates/lme-rs)
[![docs.rs](https://docs.rs/lme-rs/badge.svg)](https://docs.rs/lme-rs/latest/lme_rs/)
[![CI](https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/lme-rs.svg)](LICENSE)

`lme-rs` is a Rust library for linear and generalized linear mixed-effects models, modeled after R's `lme4` workflow. It fits models from `polars::DataFrame` inputs and includes several downstream inference helpers that are often spread across `lme4`, `lmerTest`, and `car` in R.

## What it covers

- `lm()` for fixed-effects-only linear models
- `lmer()` and `lmer_weighted()` for linear mixed models
- `glmer()` for binomial, poisson, gaussian, and gamma mixed models
- Wilkinson formulas with nested and crossed random effects
- Population-level and conditional prediction APIs
- Wald confidence intervals, parametric simulation, robust standard errors, Satterthwaite degrees of freedom, and Kenward-Roger denominator degrees of freedom
- Likelihood ratio tests between nested models and Type III ANOVA (1-DoF tests for continuous terms; joint multi-DoF Wald tests for grouped categorical fixed effects)

## Quick start

```bash
cargo add lme-rs
```

```rust
use lme_rs::{lm_df, lmer};
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    let mut file = std::fs::File::open("tests/data/sleepstudy.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // Fixed-effects-only OLS (formula + DataFrame, no random effects)
    let ols = lm_df("Reaction ~ Days", &df)?;
    println!("{}", ols);

    // Linear mixed model
    let mixed = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
    println!("{}", mixed);

    Ok(())
}
```

## Why this crate exists

`lme-rs` aims to make mixed-effects modeling usable in a native Rust workflow without giving up the modeling conventions people already know from `lme4`:

- formulas look like R formulas
- grouped random effects map to sparse matrix machinery
- model summaries and downstream helpers are designed to feel familiar to `lme4` users

## Current status

The core modeling surface is in place and exercised by the test suite, examples, and cross-language comparisons in [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md). The crate is usable today, but some features are intentionally narrower than the R ecosystem wrappers they resemble.

For a subjective, area-by-area view of what is implemented versus still open (including items not started), see [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md).

## Limitations and compatibility notes

- Numerical parity is the goal for the covered LMM and GLMM workflows, but the guarantee is scoped to the models and examples exercised by the repository tests and comparison fixtures.
- `glmer()` uses a Laplace approximation. Absolute AIC, BIC, and log-likelihood values can differ from R because `lme-rs` optimizes a deviance expression that omits data-dependent constants. Coefficients and variance parameters are the quantities to compare.
- Fixed-effects ANOVA is **Type III** only (no Type II table). Continuous fixed effects use 1-DoF marginal tests; categorical predictors encoded as multiple dummies are summarized with **joint multi-DoF Wald F-tests** when those columns are grouped in the fit. Very general contrast or multi-df designs beyond that are not covered.
- `with_kenward_roger()` produces denominator degrees of freedom that match R's `pbkrtest` to within the precision of numerical differentiation on the covered LMM models.
- The Rust crate exposes a broader surface than the Python bindings. The Python package is useful, but it is not yet a full mirror of the Rust API.
- Built-in GLMM families currently use their default links through the public `glmer()` API.

## Documentation map

- Rust API docs: [docs.rs](https://docs.rs/lme-rs/latest/lme_rs/)
- Rust usage guide: [GUIDE.md](GUIDE.md)
- Python bindings guide: [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)
- Cross-language numerical comparisons: [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md)
- Approximate completion by repository area: [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md)
- Benchmark scope and methodology: [BENCHMARKS.md](BENCHMARKS.md)
- Benchmark CI artifacts (uploaded on version tags): [GitHub Releases](https://github.com/x4g4p3x/lme-rs/releases/latest) (see [CHANGELOG.md](CHANGELOG.md) for what each release ships)
- Release history: [CHANGELOG.md](CHANGELOG.md)
- Contributor setup: [CONTRIBUTING.md](CONTRIBUTING.md)
- Release workflow: [RELEASING.md](RELEASING.md)

## Examples

The `comparisons/` directory contains cross-language reference fits for common datasets:

- `sleepstudy`
- `dyestuff`
- `pastes`
- `penicillin`
- `cbpp`
- `grouseticks`

Each example is mirrored across Rust, R, Python, and Julia where that comparison is useful.

## Development notes

Repository metadata on GitHub is synced from `Cargo.toml` by the workflow in [.github/workflows/repo-metadata.yml](.github/workflows/repo-metadata.yml). If you change the package description, homepage, keywords, or categories, the GitHub About box will be updated on the next metadata sync run.

To run the same checks as CI locally (format, clippy, build, tests, docs), use [`scripts/local_ci.sh`](scripts/local_ci.sh) or [`scripts/local_ci.ps1`](scripts/local_ci.ps1) (see [CHANGELOG.md](CHANGELOG.md) 0.1.6).
