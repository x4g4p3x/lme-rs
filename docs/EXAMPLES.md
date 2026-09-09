# Runnable examples

[Documentation](README.md) · [Rust guide](../GUIDE.md) · [Python guide](../python/PYTHON_GUIDE.md)

For a model that needs no checkout or fixture files, use
[the README quick start](../README.md#quick-start).

The examples below need a checkout:

```bash
git clone https://github.com/x4g4p3x/lme-rs.git
cd lme-rs
```

## Rust models

Run from the **repository root**, in release mode.
Cargo example names are declared in [Cargo.toml](../Cargo.toml);
fixture data lives in [tests/data](../tests/data/).

| Workflow | Source | Command |
|:---------|:-------|:--------|
| Random-slope LMM, REML | [sleepstudy.rs](../comparisons/sleepstudy.rs) | `cargo run --release --locked --example sleepstudy` |
| Random-slope LMM, ML | [sleepstudy_ml.rs](../comparisons/sleepstudy_ml.rs) | `cargo run --release --locked --example sleepstudy_ml` |
| Random-intercept LMM | [lmm_dyestuff.rs](../comparisons/lmm_dyestuff.rs) | `cargo run --release --locked --example lmm_dyestuff` |
| Nested random effects | [lmm_pastes.rs](../comparisons/lmm_pastes.rs) | `cargo run --release --locked --example lmm_pastes` |
| Crossed random effects | [lmm_penicillin.rs](../comparisons/lmm_penicillin.rs) | `cargo run --release --locked --example lmm_penicillin` |
| Binomial GLMM | [glmm_cbpp.rs](../comparisons/glmm_cbpp.rs) | `cargo run --release --locked --example glmm_cbpp` |
| Poisson GLMM | [glmm_grouseticks.rs](../comparisons/glmm_grouseticks.rs) | `cargo run --release --locked --example glmm_grouseticks` |
| Weighted LMM | [lmer_weighted.rs](../comparisons/lmer_weighted.rs) | `cargo run --release --locked --example lmer_weighted` |
| Categorical ANOVA | [categorical_anova.rs](../comparisons/categorical_anova.rs) | `cargo run --release --locked --example categorical_anova` |
| Independent versus pooled calibration | [batch_sspower_cpu.rs](../examples/batch_sspower_cpu.rs) | `cargo run --release --locked --example batch_sspower_cpu` |

[Calibration guidance](CALO_CALIBRATION.md) explains why independent curves and
one pooled mixed model answer different questions.

## Python bindings

Build the extension using [the contributor setup](../CONTRIBUTING.md#python-bindings).
Then run from **`python/`**:

```bash
uv run --no-sync python examples/lmer_sleepstudy.py
```

These scripts call `lme_python`, the bindings to the Rust engine, and resolve
fixture paths in their own code.

| Need | Entry point |
|:-----|:------------|
| Ordinary least squares | [lm_basics.py](../python/examples/lm_basics.py) |
| Random-slope LMM | [lmer_sleepstudy.py](../python/examples/lmer_sleepstudy.py) |
| Binomial GLMM | [glmer_cbpp.py](../python/examples/glmer_cbpp.py) |
| Poisson GLMM | [glmer_grouseticks.py](../python/examples/glmer_grouseticks.py) |
| Nested model comparison | [model_comparison.py](../python/examples/model_comparison.py) |
| Reference assertions, including nested and crossed models | [Verification project](../python/examples/verification_project/README.md) |
| Diagnostic plots and optional R overlays | [Plotting demo](../python/examples/plotting_demo/README.md) |

Keep `--no-sync` after building the extension: synchronization can uninstall
the editable module.

## Independent numerical comparisons

Scripts under [comparisons](../comparisons/) use R, Julia, and often Python
`statsmodels` as independent references. A `statsmodels` comparison is not an
example of the `lme_python` API. Some reference scripts intentionally simplify
the random-effects structure.

Read [COMPARISONS.md](../comparisons/COMPARISONS.md) for commands, reference
packages, expected outputs, and known differences. Read
[BENCHMARKS.md](../BENCHMARKS.md) for timing methodology.

## Validation and exploration

Run from the repository root:

| Command | Purpose |
|:--------|:--------|
| `task docs:check` | Local document targets, dashboard data, Rust example compilation, doctests, API docs |
| `task consumer:smoke` | Rust sleepstudy execution and Python wheel tests/portable workflows in isolated environments |
| `task explorations` | Native formula AST, variance-parameter grid, multiple-comparison probes |
| [Fuzz targets](../fuzz/README.md) | Parser and formula pipeline under generated inputs |

The documentation check does not execute every Markdown snippet, validate
external websites or heading anchors, or run R/Julia comparisons. Execute the
exact snippet as well when changing a copied example.
