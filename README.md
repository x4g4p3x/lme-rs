<p align="center">
  <img src="lme-rs.png" alt="lme-rs: linear, generalized and nonlinear mixed models" width="100%">
</p>

# lme-rs

<p align="center">
  <a href="https://crates.io/crates/lme-rs"><img src="https://img.shields.io/crates/v/lme-rs.svg" alt="crates.io"></a>
  <a href="https://docs.rs/lme-rs/latest/lme_rs/"><img src="https://docs.rs/lme-rs/badge.svg" alt="docs.rs"></a>
  <a href="https://pypi.org/project/lme-python/"><img src="https://img.shields.io/pypi/v/lme-python.svg" alt="PyPI"></a>
  <a href="https://github.com/x4g4p3x/lme-rs/releases/latest"><img src="https://img.shields.io/github/v/release/x4g4p3x/lme-rs.svg" alt="GitHub release"></a>
  <a href="https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml"><img src="https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml/badge.svg" alt="Release CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/crates/l/lme-rs.svg" alt="license"></a>
</p>

<p align="center">
  <a href="GUIDE.md"><img src="https://img.shields.io/badge/docs-Rust%20guide-0ea5e9" alt="Rust guide"></a>
  <a href="python/PYTHON_GUIDE.md"><img src="https://img.shields.io/badge/docs-Python%20guide-3776AB" alt="Python guide"></a>
  <a href="USABILITY.md"><img src="https://img.shields.io/badge/docs-Usability-16a34a" alt="Usability"></a>
  <a href="comparisons/COMPARISONS.md"><img src="https://img.shields.io/badge/docs-Comparisons-7c3aed" alt="Comparisons"></a>
</p>

`lme-rs` is a Rust library for **linear, generalized linear, and nonlinear mixed-effects models**, modeled after R's `lme4` workflow. It fits models from `polars::DataFrame` inputs and includes downstream inference helpers that are often spread across `lme4`, `lmerTest`, and `car` in R. Python bindings ship as [`lme-python`](https://pypi.org/project/lme-python/).

| | |
|:--|:--|
| **Latest release** | **[0.2.2](https://github.com/x4g4p3x/lme-rs/releases/tag/v0.2.2)** (2026-08-14) · [changelog](CHANGELOG.md) |
| **Install** | `cargo add lme-rs` · `pip install lme-python` |
| **Data** | `polars::DataFrame` / Polars |
| **Formulas** | Wilkinson / `lme4`-style |

0.2.2 adds LMM estimated marginal means, Tukey / Dunnett `glht`, profile CIs for variance components, Wilkinson `poly()` / `ns()` / `y ~ .` (and related transforms), multivariate AGQ-in-θ, Python `lm(y, x)`, and Gamma GLMM θ parity with lme4.

> **Repository completion (evidence-weighted): 100% (236/236 scope units).** This is a deterministic implementation-coverage score, calculated from the checked binary criteria in [`completion_manifest.json`](completion_manifest.json), not a usability or production-readiness claim. See [`REPO_COMPLETION_BY_AREA.md`](REPO_COMPLETION_BY_AREA.md).

## Contents

- [What it covers](#what-it-covers)
- [Install](#install)
- [Quick start](#quick-start)
- [Why this crate exists](#why-this-crate-exists)
- [Current status](#current-status)
- [Limitations and compatibility notes](#limitations-and-compatibility-notes)
- [Documentation](#documentation)
- [Examples](#examples)
- [Development](#development)

## What it covers

| Area | API |
|:-----|:----|
| **Linear models** | `lm()` / `lm_df()` — fixed-effects-only OLS |
| **Linear mixed models** | `lmer()`, `lmer_weighted()` |
| **GLMMs** | `glmer()`, `glmer_weighted()` — binomial, Poisson, Gaussian, and gamma; Laplace or AGQ via `n_agq` |
| **NLMMs** | `nlmer()` — `SSlogis` / `SSasymp` / `SSfol` / `SSmicmen` / `SSgompertz` / `SSpower` / `SSfpl` / `SSbiexp` / `SSweibull` / `SSasympOff` / `SSasympOrig`; optional population and group-level bounds; optional AGQ including vector RE; `nlmer_with_mean` for custom μ; scalar or multivariate RE |
| **Repeated fits** | `prepare_lmer()` / `fit_prepared()` and `prepare_glmer()` / `fit_prepared_glmer()` amortize design-matrix setup ([OPTIMIZATION.md](OPTIMIZATION.md)); `cv_grouped()` / `cv_grouped_glmer()` preserve group structure ([GUIDE.md](GUIDE.md#repeated-fits-and-cross-validation)); `boot_lmer()` / `boot_glmer()` parametric (and LMM residual) bootstrap refits with percentile CIs ([GUIDE.md](GUIDE.md#bootstrap-refits-boot_lmer--boot_glmer)) |
| **Formulas** | Nested and crossed random effects, two-way `*` / `:` interactions, `log` / `sqrt` / `exp`, `I()` arithmetic, `poly()` / `ns()`, `y ~ .`, and transformed `offset()` terms |
| **Prediction** | Population-level and conditional APIs |
| **Uncertainty** | Wald and **profile-likelihood** confidence intervals (`parms=` subset), parametric simulation, bootstrap refits, robust standard errors, Satterthwaite / Kenward–Roger dfs |
| **Tests & comparisons** | Likelihood ratio tests between nested models; Type I / II / III fixed-effects ANOVA (1-DoF tests for continuous terms; joint multi-DoF Wald tests for grouped categorical fixed effects); Tukey / Dunnett `glht`; LMM estimated marginal means (`emmeans`) with reference-grid pairwise contrasts |

## Install

**Rust**

```bash
cargo add lme-rs
```

**Python**

```bash
pip install lme-python
```

The package imports as `lme_python` and uses Polars DataFrames. See [python/README.md](python/README.md) for wheel coverage and [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md) for the full API.

## Quick start

### Rust

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

    // Linear mixed model (REML)
    let mixed = lmer("Reaction ~ Days + (Days | Subject)", &df, true)?;
    println!("{}", mixed);

    Ok(())
}
```

### Python

```python
import lme_python
import polars as pl

df = pl.DataFrame(
    {
        "y": [10.0, 12.0, 13.0, 15.0, 9.0, 11.0, 14.0, 17.0],
        "x": [0.0, 1.0, 2.0, 3.0] * 2,
        "group": ["a"] * 4 + ["b"] * 4,
    }
)

ols = lme_python.lm("y ~ x", data=df)
mixed = lme_python.lmer("y ~ x + (1 | group)", data=df, reml=True)
print(ols.summary())
print(mixed.summary())
```

## Why this crate exists

`lme-rs` aims to make mixed-effects modeling usable in a native Rust workflow without giving up the modeling conventions people already know from `lme4`:

- formulas look like R formulas
- grouped random effects map to sparse matrix machinery
- model summaries and downstream helpers are designed to feel familiar to `lme4` users

### If you already know lme4

| In R | In lme-rs |
|:-----|:----------|
| `lm` / `lmer` / `glmer` / `nlmer` | `lm` / `lm_df`, `lmer`, `glmer`, `nlmer` |
| `lmerTest` Satterthwaite / Kenward–Roger | `with_satterthwaite()` / `with_kenward_roger()` |
| `car::Anova` Types I–III | `anova_typed` / `AnovaType` |
| `multcomp::glht` Tukey / Dunnett | `glht` (pairwise / vs-control family) |
| `emmeans` | LMM `emmeans()` / `emmeans_pairs()` (reference-grid subset) |
| `bootMer` | `boot_lmer()` / `boot_glmer()` |
| `confint(..., method = "profile")` | Profile CIs, including variance components |

These helpers are intentionally familiar, not full replacements for the R packages. See [Limitations and compatibility notes](#limitations-and-compatibility-notes).

## Current status

The core modeling surface is in place and exercised by the test suite, examples, and cross-language comparisons in [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md). For **whether your workflow is in scope** — and the distinction between repository test coverage and real-world field experience — see **[USABILITY.md](USABILITY.md)**.

On the fair MixedModels.jl harness, every case in the current **12-case tier-A suite** passed the strict Rust `cold_fit` **&lt;1.0× Julia** gate, including all 10 LMM and both GLMM cases ([2026-07-22 full reference](benchmarks/fair-rust-julia-reference-2026-07-22-full-tier-a.json)). Hot `prepare_lmer` + `fit_prepared` also beat Julia on every LMM case in that run. See [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md) and [OPTIMIZATION.md](OPTIMIZATION.md).

[REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) is an internal **coverage** map (how much of the intended API exists), not a usability score.

## Limitations and compatibility notes

### Numerical parity

Numerical parity is the goal for the covered LMM and GLMM workflows, but the guarantee is scoped to the models and examples exercised by the repository tests and comparison fixtures.

### GLMM quadrature and information criteria

`glmer()` uses Laplace by default (`n_agq = 1`). For `n_agq ≥ 2`, θ is optimized under adaptive Gauss–Hermite quadrature when the quadrature grid fits: scalar RE (matching `lme4`), vector RE via a product rule per group, and multiple RE terms when total `q` is small. Larger crossed models stay on the Laplace θ path.

Absolute AIC, BIC, and log-likelihood values can differ from R because `lme-rs` optimizes a deviance expression that omits data-dependent constants. **Coefficients and variance parameters** are the quantities to compare.

### ANOVA, contrasts, and emmeans

Fixed-effects ANOVA supports **Type I**, **II**, and **III** (`anova_typed` / `AnovaType`). Continuous fixed effects use 1-DoF tests where applicable; categorical predictors encoded as multiple dummies use **joint multi-DoF Wald F-tests**, with multi-DoF Satterthwaite denominator df following **`lmerTest::contestMD()`** (see [GUIDE.md](GUIDE.md) and [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) §4).

Arbitrary user-defined **q × p** contrast matrices are supported via `test_contrast()` (Rust) / `fit.test_contrast()` (Python); named-term tests via `linear_hypothesis()` / `fit.linear_hypothesis()`.

LMM `emmeans()` uses an equal-weight categorical reference grid with numeric covariates at their means, and `emmeans_pairs()` provides Tukey–Kramer / Holm / Bonferroni comparisons. Tukey and Dunnett `glht` covers the `multcomp::mcp` pairwise family. This is **not** a full `multcomp` / `emmeans` replacement (no single-step `mvtnorm`, compact-letter display, or GLMM response-scale marginalization).

### Kenward–Roger

`with_kenward_roger()` produces denominator degrees of freedom that match R's `pbkrtest` to within the precision of numerical differentiation on the covered LMM models.

### Python bindings

The Python bindings mirror the Rust API (`lm`, `lm_matrix`, `lmer`, `prepare_lmer` / `fit_prepared`, `prepare_glmer` / `fit_prepared_glmer`, `cv_grouped` / `cv_grouped_glmer`, `boot_lmer` / `boot_glmer`, `glmer`, `nlmer`, contrasts, ANOVA, prediction, simulation, profile CIs) with structured result types and [`lme_python.pyi`](python/lme_python.pyi) stubs.

### Families and links

Built-in GLMM families cover binomial, Poisson, Gaussian, and gamma with canonical links; non-canonical links are selectable via `glmer_with_link` / `link_name=` ([GUIDE.md](GUIDE.md)).

## Documentation

| Audience | Document |
|:---------|:---------|
| Rust API reference | [docs.rs/lme-rs](https://docs.rs/lme-rs/latest/lme_rs/) |
| Rust usage | [GUIDE.md](GUIDE.md) |
| Python usage | [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md) · [python/README.md](python/README.md) |
| “Can I use this for my problem?” | [USABILITY.md](USABILITY.md) |
| Numerical comparisons | [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) |
| Implementation coverage (not usability) | [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) |
| Benchmarks | [BENCHMARKS.md](BENCHMARKS.md) · [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md) |
| LMM fit optimization | [OPTIMIZATION.md](OPTIMIZATION.md) |
| Calo / sensor calibration | [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md) (MATLAB `power2` vs `nlmer`, CUDA batch fitting) |
| MCP server (agents / Cursor) | Companion repo [lme-rs-mcp](https://github.com/x4g4p3x/lme-rs-mcp) — stdio tools for `lme_fit`, ANOVA, and bootstrap on local CSVs ([guide](https://github.com/x4g4p3x/lme-rs-mcp/blob/main/GUIDE.md)) |
| Releases | [CHANGELOG.md](CHANGELOG.md) · [GitHub Releases](https://github.com/x4g4p3x/lme-rs/releases/latest) (benchmark CI artifacts on version tags) |
| Maintainers | [CONTRIBUTING.md](CONTRIBUTING.md) · [RELEASING.md](RELEASING.md) · [AGENTS.md](AGENTS.md) |

## Examples

The [`comparisons/`](comparisons/) directory contains cross-language reference fits for common datasets. Each example is mirrored across Rust, R, Python, and Julia where that comparison is useful.

| Dataset | Typical workflow |
|:--------|:-----------------|
| `sleepstudy` | LMM with random slope |
| `dyestuff` | Intercept LMM / gamma GLMM |
| `pastes` | Nested random effects; ANOVA, `glht`, `emmeans` |
| `penicillin` | Crossed random effects |
| `cbpp` | Binomial GLMM |
| `grouseticks` | Poisson GLMM |

## Development

Repository metadata on GitHub is synced from `Cargo.toml` by [.github/workflows/repo-metadata.yml](.github/workflows/repo-metadata.yml) on `v*` release tags or manual dispatch. Preflight with `task repo-metadata`; the workflow needs a valid **`REPO_ADMIN_TOKEN`** secret (see [CONTRIBUTING.md](CONTRIBUTING.md)).

GitHub Actions run automatically for pull requests and `v*` tag pushes, with `workflow_dispatch` available for manual runs. Local checks carry the day-to-day gate in layers:

| When | Command |
|:-----|:--------|
| Before push | **`task preflight`** (lint, compile graph, `cargo audit`, metadata dry-run) |
| Before large PRs or release tags | **`task ci`** ([AGENTS.md](AGENTS.md) for hook details) |

Install `cargo-audit` once: `cargo install cargo-audit`.
