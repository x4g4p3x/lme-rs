# Documentation

[Repository home](../README.md) · [Rust guide](../GUIDE.md) · [Python guide](../python/PYTHON_GUIDE.md)

Documentation on `master` describes the development checkout. For a published
package, consult the matching release tag and [changelog](../CHANGELOG.md).
The docs.rs reference describes published Rust releases.

## Start here

1. Install the [Rust dependencies](../GUIDE.md#getting-started) or
   [Python package](../python/README.md#install).
2. Run the self-contained [first model](../README.md#quick-start).
3. Learn [formula and data requirements](../GUIDE.md#data-requirements),
   then choose a [prediction method](../GUIDE.md#predictions).
4. Check [workflow scope](../USABILITY.md) before applying the example to your data.
5. Use [the example catalog](EXAMPLES.md) for realistic fixtures and advanced operations.

## Choose a workflow

| Goal | Rust | Python |
|:-----|:-----|:-------|
| Fit OLS, LMM, GLMM, or NLMM | [Modeling](../GUIDE.md#modeling-workflows) | [Modeling](../python/PYTHON_GUIDE.md#fitting-models) |
| Predict for new data or groups | [Predictions](../GUIDE.md#predictions) | [Predictions](../python/PYTHON_GUIDE.md#predictions) |
| Intervals, tests, group comparisons | [Inference](../GUIDE.md#inference-and-diagnostics) | [Inference](../python/PYTHON_GUIDE.md#confidence-intervals-and-summary-data) |
| Refit, cross-validate, bootstrap | [Repeated fits](../GUIDE.md#repeated-fits-and-cross-validation) | [Repeated fits](../python/PYTHON_GUIDE.md#repeated-fits-and-cross-validation) |
| Calibrate pooled sensors | [Calibration](CALO_CALIBRATION.md) | [Nonlinear models](../python/PYTHON_GUIDE.md#nonlinear-mixed-models) |
| Resolve a problem | [Troubleshooting](TROUBLESHOOTING.md) | [Troubleshooting](TROUBLESHOOTING.md) |

## API and examples

- [Rust API reference](https://docs.rs/lme-rs/latest/lme_rs/): signatures, fields, modules.
- [Basin optimizer](BASIN.md): optional backend, numerical controls, and comparisons.
- [Python type reference](../python/lme_python.pyi): signatures and structured results.
- [Runnable examples](EXAMPLES.md): commands, fixtures, dependencies.
- [Python verification project](../python/examples/verification_project/README.md):
  assertions against reference fixtures.
- [Plotting demo](../python/examples/plotting_demo/README.md): diagnostics and optional R overlays.
- [Explorations](../scripts/explorations/README.md): parser, variance-parameter grid, multiple comparisons.
- [Companion MCP server](https://github.com/x4g4p3x/lme-rs-mcp): a separate repository for tool-based CSV analysis.

## Evidence and performance

| Document | Question it answers |
|:---------|:--------------------|
| [Supported workflows](../USABILITY.md) | Does the scope fit my analysis? |
| [Numerical comparisons](../comparisons/COMPARISONS.md) | What agrees with a reference, and what differs? |
| [Benchmarks](../BENCHMARKS.md) | Which harness should I run and how do I interpret it? |
| [Benchmark coverage](../BENCHMARK_COVERAGE.md) | Which models have external timing evidence? |
| [Dashboard](https://x4g4p3x.github.io/lme-rs/benchmarks/) | How can I explore published timing artifacts? |
| [Optimization notes](../OPTIMIZATION.md) | Which implementation choices were measured and why? |
| [Completion report](../REPO_COMPLETION_BY_AREA.md) | Which locked commitments are complete or open? |

A numerical test, a benchmark, and a completion percentage answer different
questions. Follow each claim to its fixture, revision, or manifest.

## Contribute and maintain

| Document | Purpose |
|:---------|:--------|
| [Contributing](../CONTRIBUTING.md) | Setup, checks, pull requests, documentation maintenance |
| [Agent pre-flights](../AGENTS.md) | Required validation and completion-score rules |
| [CI runner](../scripts/ci/README.md) | Commands shared by Task, hooks, and Actions |
| [CI timing history](../CI_PERFORMANCE.md) | Dated hosted measurements and cache design |
| [Fuzzing](../fuzz/README.md) | Parser and formula-pipeline fuzz targets |
| [Releasing](../RELEASING.md) | Versions, tag validation, publication, recovery |
| [Changelog](../CHANGELOG.md) | Versioned history |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Dependency licenses and fixture provenance |
| [Relinking](../RELINKING.md) | Source and rebuild information |
| [License](../LICENSE) | Project license |
