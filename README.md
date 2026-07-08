# lme-rs - mixed-effects models in Rust

[![crates.io](https://img.shields.io/crates/v/lme-rs.svg)](https://crates.io/crates/lme-rs)
[![docs.rs](https://docs.rs/lme-rs/badge.svg)](https://docs.rs/lme-rs/latest/lme_rs/)
[![Release CI](https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/lme-rs.svg)](LICENSE)

`lme-rs` is a Rust library for linear and generalized linear mixed-effects models, modeled after R's `lme4` workflow. It fits models from `polars::DataFrame` inputs and includes several downstream inference helpers that are often spread across `lme4`, `lmerTest`, and `car` in R.

## What it covers

- `lm()` for fixed-effects-only linear models
- `lmer()` and `lmer_weighted()` for linear mixed models
- `prepare_lmer()` / `fit_prepared()` to amortize design-matrix setup when fitting the same formula and data repeatedly (see [OPTIMIZATION.md](OPTIMIZATION.md))
- `nlmer()` for nonlinear mixed models (`SSlogis` / `SSasymp` / `SSfol` / `SSmicmen` / `SSgompertz` means; optional scalar AGQ; `nlmer_with_mean` for custom μ in Rust and Python; scalar or multivariate random effects on nonlinear parameters, e.g. Orange-tree growth)
- `glmer()` and `glmer_weighted()` for binomial, poisson, gaussian, and gamma mixed models
- Wilkinson formulas with nested and crossed random effects
- Population-level and conditional prediction APIs
- Wald confidence intervals, parametric simulation, robust standard errors, Satterthwaite degrees of freedom, and Kenward-Roger denominator degrees of freedom
- Likelihood ratio tests between nested models and Type I / II / III fixed-effects ANOVA (1-DoF tests for continuous terms; joint multi-DoF Wald tests for grouped categorical fixed effects)

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

The core modeling surface is in place and exercised by the test suite, examples, and cross-language comparisons in [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md). For **whether your workflow is in scope** — and the distinction between repository test coverage and real-world field experience — see **[USABILITY.md](USABILITY.md)**.

On the fair MixedModels.jl harness, **`crossed_20k` hot fits** (`prepare_lmer` + `fit_prepared`) are **~10 ms** — faster than Julia on the same fixture — while one-shot `lmer()` is **~1.3×** Julia because setup and post-fit work are included in wall time. Axis (3) cold-fit target: **≤1.5×** Julia ([BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md)). See [OPTIMIZATION.md](OPTIMIZATION.md) and [BENCHMARKS.md](BENCHMARKS.md#fair-rust-julia-2026-07-09-prepare-fast-path).

[REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) is an internal **coverage** map (how much of the intended API exists), not a usability score.

## Limitations and compatibility notes

- Numerical parity is the goal for the covered LMM and GLMM workflows, but the guarantee is scoped to the models and examples exercised by the repository tests and comparison fixtures.
- `glmer()` uses a Laplace approximation. Absolute AIC, BIC, and log-likelihood values can differ from R because `lme-rs` optimizes a deviance expression that omits data-dependent constants. Coefficients and variance parameters are the quantities to compare.
- Fixed-effects ANOVA supports **Type I**, **II**, and **III** (`anova_typed` / `AnovaType`). Continuous fixed effects use 1-DoF tests where applicable; categorical predictors encoded as multiple dummies use **joint multi-DoF Wald F-tests**, with multi-DoF Satterthwaite denominator df following **`lmerTest::contestMD()`** (see [GUIDE.md](GUIDE.md) and [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) §4). Arbitrary user-defined **q × p** contrast matrices are supported via `test_contrast()` (Rust) / `fit.test_contrast()` (Python); named-term tests via `linear_hypothesis()` / `fit.linear_hypothesis()`.
- `with_kenward_roger()` produces denominator degrees of freedom that match R's `pbkrtest` to within the precision of numerical differentiation on the covered LMM models.
- The Python bindings mirror the Rust API (`lm`, `lm_matrix`, `lmer`, `glmer`, `nlmer`, contrasts, ANOVA, prediction, simulation) with structured result types and [`lme_python.pyi`](python/lme_python.pyi) stubs.
- Built-in GLMM families cover binomial, Poisson, Gaussian, and gamma with canonical links; non-canonical links are selectable via `glmer_with_link` / `link_name=` ([`GUIDE.md`](GUIDE.md)).

## Documentation map

- Rust API docs: [docs.rs](https://docs.rs/lme-rs/latest/lme_rs/)
- Rust usage guide: [GUIDE.md](GUIDE.md)
- Python bindings guide: [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)
- Cross-language numerical comparisons: [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md)
- **Usability** (workflows in scope, validation posture, field experience): [USABILITY.md](USABILITY.md)
- Approximate **coverage** by repository area (not usability): [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md)
- Benchmark scope and methodology: [BENCHMARKS.md](BENCHMARKS.md); **coverage map** [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md)
- LMM fit optimization (engineering notes): [OPTIMIZATION.md](OPTIMIZATION.md)
- Benchmark CI artifacts (uploaded on version tags): [GitHub Releases](https://github.com/x4g4p3x/lme-rs/releases/latest) (see [CHANGELOG.md](CHANGELOG.md) for what each release ships)
- Release history: [CHANGELOG.md](CHANGELOG.md)
- Contributor setup: [CONTRIBUTING.md](CONTRIBUTING.md) (also lists security audit and crate publish dry-run workflows)
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

Repository metadata on GitHub is synced from `Cargo.toml` by [.github/workflows/repo-metadata.yml](.github/workflows/repo-metadata.yml) on `v*` release tags or manual dispatch. Preflight with `task repo-metadata`; the workflow needs a valid **`REPO_ADMIN_TOKEN`** secret (see [CONTRIBUTING.md](CONTRIBUTING.md)).

GitHub Actions are release-oriented: automatic runs are limited to `v*` tag pushes, with `workflow_dispatch` available for manual runs. Local checks carry the day-to-day gate in layers: **`task preflight`** before push (lint, compile graph, `cargo audit`, metadata dry-run), **`task ci`** before large PRs or release tags ([AGENTS.md](AGENTS.md) for hook details). Install `cargo-audit` once: `cargo install cargo-audit`.
