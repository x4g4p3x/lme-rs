# lme-rs - mixed-effects models in Rust

[![crates.io](https://img.shields.io/crates/v/lme-rs.svg)](https://crates.io/crates/lme-rs)
[![docs.rs](https://docs.rs/lme-rs/badge.svg)](https://docs.rs/lme-rs/latest/lme_rs/)
[![Release CI](https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/lme-rs.svg)](LICENSE)

`lme-rs` is a Rust library for linear and generalized linear mixed-effects models, modeled after R's `lme4` workflow. It fits models from `polars::DataFrame` inputs and includes several downstream inference helpers that are often spread across `lme4`, `lmerTest`, and `car` in R.

> **Repository completion (evidence-weighted): 88% (207/236 scope units).** This is a deterministic implementation-coverage score, calculated from the checked binary criteria in [`completion_manifest.json`](completion_manifest.json), not a usability or production-readiness claim. See [`REPO_COMPLETION_BY_AREA.md`](REPO_COMPLETION_BY_AREA.md).

## What it covers

- `lm()` for fixed-effects-only linear models
- `lmer()` and `lmer_weighted()` for linear mixed models
- `prepare_lmer()` / `fit_prepared()` and `prepare_glmer()` / `fit_prepared_glmer()` to amortize design-matrix setup for repeated LMM/GLMM fits (see [OPTIMIZATION.md](OPTIMIZATION.md))
- `cv_grouped()` / `cv_grouped_glmer()` for group-structure-preserving k-fold CV (see [GUIDE.md](GUIDE.md#repeated-fits-and-cross-validation))
- `boot_lmer()` / `boot_glmer()` for parametric (and LMM residual) bootstrap refits with percentile CIs (see [GUIDE.md](GUIDE.md#bootstrap-refits-boot_lmer--boot_glmer))
- `nlmer()` for nonlinear mixed models (`SSlogis` / `SSasymp` / `SSfol` / `SSmicmen` / `SSgompertz` / `SSpower` / `SSfpl` / `SSbiexp` / `SSweibull` / `SSasympOff` / `SSasympOrig`; optional population and group-level bounds; optional scalar AGQ; `nlmer_with_mean` for custom μ; scalar or multivariate RE)
- `glmer()` and `glmer_weighted()` for binomial, poisson, gaussian, and gamma mixed models (Laplace or scalar AGQ via `n_agq`)
- Wilkinson formulas with nested and crossed random effects
- Population-level and conditional prediction APIs
- Wald and **profile-likelihood** confidence intervals (`parms=` subset), parametric simulation, bootstrap refits, robust standard errors, Satterthwaite / Kenward–Roger dfs
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

On the fair MixedModels.jl harness, every case in the current **12-case tier-A suite** passed the strict Rust `cold_fit` **&lt;1.0× Julia** gate, including all 10 LMM and both GLMM cases ([2026-07-22 full reference](benchmarks/fair-rust-julia-reference-2026-07-22-full-tier-a.json)). Hot `prepare_lmer` + `fit_prepared` also beat Julia on every LMM case in that run. See [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md) and [OPTIMIZATION.md](OPTIMIZATION.md).

[REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) is an internal **coverage** map (how much of the intended API exists), not a usability score.

## Limitations and compatibility notes

- Numerical parity is the goal for the covered LMM and GLMM workflows, but the guarantee is scoped to the models and examples exercised by the repository tests and comparison fixtures.
- `glmer()` uses Laplace by default (`n_agq = 1`). For **scalar** random effects, `n_agq ≥ 2` optimizes θ under adaptive Gauss–Hermite quadrature (matching `lme4`). Absolute AIC, BIC, and log-likelihood values can differ from R because `lme-rs` optimizes a deviance expression that omits data-dependent constants. Coefficients and variance parameters are the quantities to compare.
- Fixed-effects ANOVA supports **Type I**, **II**, and **III** (`anova_typed` / `AnovaType`). Continuous fixed effects use 1-DoF tests where applicable; categorical predictors encoded as multiple dummies use **joint multi-DoF Wald F-tests**, with multi-DoF Satterthwaite denominator df following **`lmerTest::contestMD()`** (see [GUIDE.md](GUIDE.md) and [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) §4). Arbitrary user-defined **q × p** contrast matrices are supported via `test_contrast()` (Rust) / `fit.test_contrast()` (Python); named-term tests via `linear_hypothesis()` / `fit.linear_hypothesis()`.
- `with_kenward_roger()` produces denominator degrees of freedom that match R's `pbkrtest` to within the precision of numerical differentiation on the covered LMM models.
- The Python bindings mirror the Rust API (`lm`, `lm_matrix`, `lmer`, `prepare_lmer` / `fit_prepared`, `prepare_glmer` / `fit_prepared_glmer`, `cv_grouped` / `cv_grouped_glmer`, `boot_lmer` / `boot_glmer`, `glmer`, `nlmer`, contrasts, ANOVA, prediction, simulation, profile CIs) with structured result types and [`lme_python.pyi`](python/lme_python.pyi) stubs.
- Built-in GLMM families cover binomial, Poisson, Gaussian, and gamma with canonical links; non-canonical links are selectable via `glmer_with_link` / `link_name=` ([`GUIDE.md`](GUIDE.md)).

## Documentation map

- Rust API docs: [docs.rs](https://docs.rs/lme-rs/latest/lme_rs/)
- Rust usage guide: [GUIDE.md](GUIDE.md)
- Python bindings guide: [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)
- Cross-language numerical comparisons: [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md)
- **Usability** (workflows in scope, validation posture, field experience): [USABILITY.md](USABILITY.md)
- Evidence-weighted **implementation coverage** by repository area (not usability): [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md)
- Benchmark scope and methodology: [BENCHMARKS.md](BENCHMARKS.md); **coverage map** [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md)
- LMM fit optimization (engineering notes): [OPTIMIZATION.md](OPTIMIZATION.md)
- **Calo / sensor calibration** (MATLAB `power2` vs `nlmer`, CUDA batch fitting): [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md)
- Benchmark CI artifacts (uploaded on version tags): [GitHub Releases](https://github.com/x4g4p3x/lme-rs/releases/latest) (see [CHANGELOG.md](CHANGELOG.md) for what each release ships)
- Release history: [CHANGELOG.md](CHANGELOG.md)
- **MCP server (agents / Cursor):** optional companion repo [lme-rs-mcp](https://github.com/x4g4p3x/lme-rs-mcp) — stdio tools for `lme_fit`, ANOVA, and bootstrap on local CSVs ([GUIDE](https://github.com/x4g4p3x/lme-rs-mcp/blob/main/GUIDE.md))
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

GitHub Actions run automatically for pull requests and `v*` tag pushes, with `workflow_dispatch` available for manual runs. Local checks carry the day-to-day gate in layers: **`task preflight`** before push (lint, compile graph, `cargo audit`, metadata dry-run), **`task ci`** before large PRs or release tags ([AGENTS.md](AGENTS.md) for hook details). Install `cargo-audit` once: `cargo install cargo-audit`.
