# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
