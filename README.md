<p align="center">
  <img src="lme-rs.png" alt="lme-rs: linear, generalized and nonlinear mixed models" width="100%">
</p>

<p align="center">
  <a href="https://crates.io/crates/lme-rs"><img src="https://img.shields.io/crates/v/lme-rs.svg" alt="crates.io"></a>
  <a href="https://docs.rs/lme-rs/latest/lme_rs/"><img src="https://docs.rs/lme-rs/badge.svg" alt="Rust API reference"></a>
  <a href="https://pypi.org/project/lme-python/"><img src="https://img.shields.io/pypi/v/lme-python.svg" alt="PyPI"></a>
  <a href="https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml"><img src="https://github.com/x4g4p3x/lme-rs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/crates/l/lme-rs.svg" alt="MIT license"></a>
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> ·
  <a href="GUIDE.md">Rust guide</a> ·
  <a href="python/PYTHON_GUIDE.md">Python guide</a> ·
  <a href="USABILITY.md">Supported workflows</a> ·
  <a href="https://github.com/sponsors/x4g4p3x">Sponsor</a>
</p>

# Mixed-effects models in Rust and Python

**lme-rs** fits linear, generalized linear, and nonlinear mixed-effects models
using familiar `lme4`-style formulas. Use it in a native Rust pipeline or through
the `lme_python` Python module. Both interfaces use the same Rust fitting engine.

Start with a DataFrame, describe the fixed and random effects, then inspect
estimates, predict outcomes, and run inference. No R installation is needed to
use the library.

## What it covers

| Your task | Entry point |
|:----------|:------------|
| Ordinary least squares | Rust `lm_df` / matrix `lm`; Python `lm` |
| Gaussian mixed model | `lmer`, `lmer_weighted` |
| Binomial, Poisson, Gaussian, or Gamma mixed model | `glmer`, `glmer_weighted` |
| Nonlinear mixed model with built-in or custom means | `nlmer`, `nlmer_with_mean` |
| Predict for existing or new groups | Population and conditional prediction |
| Quantify uncertainty | Wald/profile intervals, simulation, bootstrap refits |
| Test effects and compare groups | ANOVA, contrasts, Tukey/Dunnett comparisons, LMM estimated marginal means |
| Repeat an analysis efficiently | Prepared fits, grouped cross-validation, parallel bootstrap |

For formula syntax and workflow limits, read [the Rust guide](GUIDE.md),
[the Python guide](python/PYTHON_GUIDE.md), and [the workflow assessment](USABILITY.md).

## Install

### Rust

Add these dependencies to your application's `Cargo.toml`:

```toml
[dependencies]
lme-rs = "0.2.2"
polars = { version = "0.46", features = ["csv"] }
anyhow = "1"
```

Use the compatible Polars **0.46** Rust dependency when passing DataFrames to
`lme-rs`. Run numerical examples with `cargo run --release`.
See [Rust setup](GUIDE.md#getting-started) for build requirements.

### Python

```bash
python -m pip install lme-python
```

The install name is `lme-python`; the import name is `lme_python`.
A matching binary wheel does not require Rust. See
[Python installation](python/README.md#install) for wheel availability and source builds.

## Quick start

These examples create their own small dataset and work without cloning this
repository. They demonstrate the API; use more groups and observations for
a substantive mixed-model analysis.

### Rust

Save as `src/main.rs` in the application configured above:

```rust
use lme_rs::lmer;
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    let data = df!(
        "y" => [10.0, 12.0, 13.0, 15.0, 9.0, 11.0, 14.0, 17.0, 8.0, 10.0, 12.0, 14.0],
        "x" => [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        "group" => ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
    )?;
    let fit = lmer("y ~ x + (1 | group)", &data, true)?;
    println!("{fit}");
    println!("Population predictions: {:?}", fit.predict(&data)?);
    Ok(())
}
```

### Python

```python
import lme_python
import polars as pl

data = pl.DataFrame(
    {
        "y": [10.0, 12.0, 13.0, 15.0, 9.0, 11.0, 14.0, 17.0, 8.0, 10.0, 12.0, 14.0],
        "x": [0.0, 1.0, 2.0, 3.0] * 3,
        "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
    }
)
fit = lme_python.lmer("y ~ x + (1 | group)", data=data, reml=True)
print(fit.summary())
print(fit.predict(data))
```

Here, `y ~ x` fits an intercept and a common slope; `(1 | group)` adds a
group-specific random intercept. `true` / `reml=True` selects REML estimation.
`predict` uses fixed effects; `predict_conditional` includes fitted group effects.

Continue with [Rust modeling workflows](GUIDE.md#modeling-workflows),
[Python modeling workflows](python/PYTHON_GUIDE.md#fitting-models), or
[the runnable example catalog](docs/EXAMPLES.md).

## Why this crate exists

The library brings familiar mixed-model formulas and inference into Rust
applications and Python data workflows. Sparse random-effects algebra and
prepared fitting support analyses that repeatedly fit the same design.

### If you already know lme4

| Familiar R workflow | Corresponding lme-rs workflow |
|:--------------------|:------------------------------|
| `lmer` / `glmer` / `nlmer` | Formula fitting with the same function names |
| `lmerTest` / `pbkrtest` | Satterthwaite and Kenward–Roger inference on covered LMMs |
| `car::Anova` | Type I, II, and III fixed-effects ANOVA |
| `multcomp::glht` | Scoped Tukey/Dunnett contrast families |
| `emmeans` | Equal-weight LMM reference grids and pairwise comparisons |
| `bootMer` | Response simulation/resampling and refitting with `boot_lmer` / `boot_glmer` |

These are workflow mappings, not guarantees of identical options or numerical
output. The guides explain the supported subset.

## Current status

Release changes are recorded in [CHANGELOG.md](CHANGELOG.md). Documentation on
`master` follows the development checkout; use a release tag for documentation
matching an exact published version.

Tests include numerical identities, R reference fixtures, Python binding tests,
and portable consumer examples. [The comparison guide](comparisons/COMPARISONS.md)
identifies the covered models and quantities. [USABILITY.md](USABILITY.md)
explains how to assess a new dataset and distinguishes repository validation
from field experience.

> **Repository completion (evidence-weighted): 91% (235/258 scope units).** This is a generated score for the locked scope in [completion_manifest.json](completion_manifest.json). It does not measure production maturity or suitability for an individual analysis. See [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md).

### Performance evidence

The [2026-07-22 reference run](benchmarks/fair-rust-julia-reference-2026-07-22-full-tier-a.json)
recorded lower Rust cold-fit medians than Julia on all **10 LMM cases** in its
12-case suite. Its two GLMM measurements predate later fitting changes.
Treat timings as measurements of a named revision and machine.
See [methodology](BENCHMARKS.md) and [coverage](BENCHMARK_COVERAGE.md).

## Limitations and compatibility notes

- **Formula support is scoped.** Common Wilkinson syntax, nested/crossed effects,
  transforms, polynomials, and splines are covered; arbitrary R expressions are not.
- **GLMM prediction has two scales.** `predict` returns the linear predictor;
  `predict_response` returns probabilities or means. Reported likelihood constants
  can differ from R, so absolute AIC/BIC values are not interchangeable.
- **Quadrature is bounded by model size.** `n_agq=1` is Laplace. Higher orders
  use adaptive quadrature where the grid fits; large crossed structures can
  retain Laplace optimization.
- **Inference has explicit limits.** Estimated marginal means use an equal-weight
  reference grid; full R package behavior and GLMM response-scale
  marginalization are not implemented.
- **Nonlinear models use one grouping factor.** Built-in means, custom means,
  and parameter bounds are available; this is not a general `nlme` replacement.

See [workflow guidance](USABILITY.md) and [troubleshooting](docs/TROUBLESHOOTING.md)
for practical next steps.

## Documentation

Start at the **[documentation index](docs/README.md)** for learning paths and a
complete reference map.

| Learn or do | Read |
|:------------|:-----|
| Use Rust | [Rust guide](GUIDE.md) · [API reference](https://docs.rs/lme-rs/latest/lme_rs/) |
| Use Python | [Package setup](python/README.md) · [Python guide](python/PYTHON_GUIDE.md) |
| Run an example | [Example catalog](docs/EXAMPLES.md) |
| Choose a workflow | [Workflow assessment](USABILITY.md) |
| Understand numerical evidence | [Comparisons](comparisons/COMPARISONS.md) |
| Measure performance | [Benchmarks](BENCHMARKS.md) · [Engineering notes](OPTIMIZATION.md) |
| Contribute or release | [Contributing](CONTRIBUTING.md) · [Releasing](RELEASING.md) |

## Examples

[The example catalog](docs/EXAMPLES.md) maps sleepstudy, dyestuff, pastes,
penicillin, cbpp, grouseticks, and calibration examples to their source,
dependencies, and commands. It distinguishes `lme_python` examples from
independent `statsmodels` comparisons.

## Development

```bash
mise install
task setup
task preflight
```

Run `task ci` for the full local core validation flow.
[CONTRIBUTING.md](CONTRIBUTING.md) explains required checks and hosted coverage.
Support development through [GitHub Sponsors](https://github.com/sponsors/x4g4p3x).
