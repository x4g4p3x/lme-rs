# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Python CI and [`scripts/local_ci.sh`](scripts/local_ci.sh) / [`scripts/local_ci.ps1`](scripts/local_ci.ps1) install dev dependencies from a pinned [`python/requirements-ci.txt`](python/requirements-ci.txt) (regenerated with `pip-compile`). [`.github/workflows/audit.yml`](.github/workflows/audit.yml) runs `pip-audit` on that file.

## [0.1.6] - 2026-03-27

### Added

- Python `examples/plotting_demo`: matplotlib figures from `lme_python`, optional **lme4** parity (`plot_r.R`), side-by-side PNGs (`figures_compare/`), numeric overlay on shared axes (`figures_overlay/` via `figures_data/` JSON/CSV), `run_all.py`, Windows `Rscript` discovery (`find_rscript.py`), and raster fallback (`figures_overlay_raster/`).
- `scripts/local_ci.sh` and `scripts/local_ci.ps1` to run the same checks as [`.github/workflows/ci.yml`](.github/workflows/ci.yml) locally (`fmt`, `clippy`, build, test, `doc`).

## [0.1.5] - 2026-03-22

### Added

- Added automatic categorical dummy encoding (dropping collinear intercepts) directly from categorical/string Polars dataframe columns natively.
- Implemented Multi-DoF Joint Wald F-Tests tracking dummy groups in `anova()` (Type III Satterthwaite representations) for valid categorical fixed-effect testing.
- Added support for generating `Gaussian` families in Python's `glmer` mapping, alongside exposing an equivalent `lmer_weighted` function. 
- Plumbed explicit Adaptive Gauss-Hermite Quadrature (`nAGQ`) scaling configuration arrays throughout the optimizer targeting upcoming structural convergence capabilities.

## [0.1.4] - 2026-03-22

### Added

- Formula-string entry point `lm()` and corresponding Python binding for fitting ordinary least squares models using DataFrames.
- New Python examples directory `python/examples` with five runnable demonstration scripts mapping to common mixed-modeling workflows.

### Changed

- Renamed `examples/` directory to `comparisons/` to clarify that it contains cross-language parity scripts, updating all build and documentation references accordingly.
- Promoted Kenward-Roger degrees of freedom from "provisional" status after validating numerical parity (within 0.01 df) against R's `pbkrtest`.

## [0.1.3] - 2026-03-12

### Added

- Expanded the benchmark suite to cover parsing, matrix construction, prediction paths, inference helpers, weighted fits, Kenward-Roger timing, and size sweeps.
- Added automated cross-language benchmark scripting and a GitHub workflow that saves benchmark artifacts in CI and on release tags.
- Added `CONTRIBUTING.md`, `RELEASING.md`, and `BENCHMARKS.md` to document contributor setup, release flow, and benchmark methodology.

### Changed

- Reworked the README, Rust guide, Python guide, and comparison documentation to better reflect current capabilities, limitations, and release workflow.
- Switched the crate homepage to the docs.rs site and expanded repository metadata sync for GitHub description, topics, and website fields.
- Hardened the benchmark automation and example scripts so cross-language benchmark runs complete consistently on GitHub Actions.

## [0.1.2] - 2026-03-09

### Fixed

- Fixed a NaN deviance issue in the fitting path.
- Expanded Python test coverage for the 0.1.2 release.

### Changed

- Aligned `Cargo.lock` with the 0.1.2 version bump.

## [0.1.1] - 2026-03-09

### Changed

- Bumped the crate and Python extension versions for the 0.1.1 release.
- Updated lockfiles to remove yanked dependencies.
- Fixed missing documentation warnings needed for publishing.

### Fixed

- Hardened the GitHub Actions and wheel build pipeline across Linux, macOS, and Windows.
- Restored missing dependencies and build flags needed for OpenBLAS-based wheel builds.
- Fixed several maturin and PyO3 packaging workflow issues for the Python release path.

## [0.1.0] - 2026-03-08

### Added

- **Core LMM** - `lmer()` with REML and ML estimation, matching R's `lme4` to 4 decimal places
- **GLMM** - `glmer()` with Poisson, Binomial, Gaussian, and Gamma families via Laplace approximation
- **Predictions** - `predict()`, `predict_conditional()`, `predict_response()`, `predict_conditional_response()` with `allow_new_levels` support
- **Inference** - Satterthwaite and Kenward-Roger degrees of freedom, robust sandwich standard errors
- **Model comparison** - `anova()` for likelihood ratio tests between nested models
- **Diagnostics** - `confint()` for Wald confidence intervals, `simulate()` for parametric bootstrap
- **Formula parser** - Wilkinson notation with nested (`a/b`), crossed (`a + b`), and correlated slope+intercept (`x | group`) random effects
- **Sparse matrix architecture** - Compressed sparse column format with sparse Cholesky for large datasets
- **Python bindings** - `lme_python` package via PyO3/maturin with `lmer()`, `glmer()`, `predict()`
- **Documentation** - `GUIDE.md`, `PYTHON_GUIDE.md`, and cross-language `COMPARISONS.md`
