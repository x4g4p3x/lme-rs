# Troubleshooting

[Documentation](README.md) · [Rust guide](../GUIDE.md) · [Python guide](../python/PYTHON_GUIDE.md)

## Installation and builds

| Symptom | Check and next step |
|:--------|:--------------------|
| Rust rejects a Polars DataFrame type | Use Polars 0.46 as in [Rust setup](../GUIDE.md#getting-started); inspect `cargo tree -d` for incompatible versions |
| A Rust fit is very slow | Run numerical examples in release mode: `cargo run --release --example sleepstudy` |
| First build is slow | Native dependencies may need compilation or downloading; check free disk space and the actual error before retrying |
| `pip` attempts a source build | Check [release wheel files](https://pypi.org/project/lme-python/#files) for the interpreter/platform; follow [source setup](../CONTRIBUTING.md#python-bindings) if needed |
| `ModuleNotFoundError: lme_python` | Use `python -m pip show lme-python` with the interpreter that runs your script |
| Import fails after `uv sync` | From `python/`, rebuild with `uv run --no-sync maturin develop --release`; use `uv run --no-sync` afterward |
| `task` is missing | Run `mise install`; use `mise exec -- task docs:check` if mise is available but shims are not active |
| Windows `LNK1318` / `LNK1106` | Check disk space as well as the linker diagnostic; clean only inactive generated build artifacts |

The Python CI matrix exercises 3.10–3.13. Minimum-version metadata, binary wheel
availability, and tested interpreters are separate facts.

## Data and formulas

| Symptom | Check |
|:--------|:------|
| Missing-column error | Formula names must match DataFrame names exactly |
| Numeric conversion error | Inspect dtypes, nulls, non-finite values, and transformation domains |
| Unexpected categorical coefficients | Inspect `fixed_names`, categorical levels, and the intended reference category |
| Rank or dimension error | Check redundant predictors, intercept/dummy coding, and model identifiability |
| Binomial proportions rejected | Use positive integer trial weights with near-integer successes `y * trials` |
| `SSpower` fails | Inputs require `x > 0`; inspect starting values and [bounds](CALO_CALIBRATION.md) |

Precompute unsupported transformations as DataFrame columns.
See [formula expectations](../GUIDE.md#formula-expectations).

## Convergence and inference

A convergence flag alone does not establish whether a result is useful.

1. Verify response domains, labels, missing values, and predictor scaling.
2. Fit a simpler random-effects structure and check identifiability.
3. Inspect separation for binomial models or starting values for nonlinear models.
4. Compare a reference fit using the same rows, formula, family, link, weights,
   and estimation mode.
5. Record estimates and diagnostics when reporting the issue.

For likelihood-ratio comparisons involving different fixed effects, fit both
LMMs with ML on the same observations. `anova(fit_a, fit_b)` compares models;
`fit.anova(...)` produces a fixed-effects table for one model.

Profile intervals repeatedly refit the model. Select a coefficient subset
with Rust `confint_profile_parms` or Python `parms=` when appropriate.
See [inference scope](../GUIDE.md#inference-and-diagnostics).

## Predictions

| Need | Method |
|:-----|:-------|
| Fixed effects only | `predict` |
| Include fitted group effects | `predict_conditional` |
| GLMM probabilities or response means | `predict_response` |
| GLMM response means including fitted group effects | `predict_conditional_response` |

Conditional prediction rejects unseen groups by default.
`allow_new_levels=true` (Rust) or `True` (Python) gives those groups a zero
random-effect contribution; it does not estimate their effects.
Match training column names and encodings, including grouping columns required
by the chosen path. See [prediction details](../GUIDE.md#predictions).

## Repeated fitting

Prepared objects cache a specific design. Rebuild when rows, predictors, groups,
or the formula changes. Cross-validation must prepare each training fold
separately; use `cv_grouped` or `cv_grouped_glmer`.

Rust exposes `fit_prepared_with_response` and
`fit_prepared_glmer_with_response` for new responses on a fixed design.
Python's prepared-fit functions reuse the stored response; use the bootstrap
helpers for response-resampling workflows.

Avoid oversubscribing BLAS when parallelizing outside the library.
See [performance notes](../GUIDE.md#performance-notes).

## Report a reproducible issue

Include the package version, OS/architecture, Rust or Python version, formula,
small synthetic dataset, exact error, and reproducing command. For a reference
mismatch, include its package version and matching fit options. Remove private
data and credentials.

[Open an issue](https://github.com/x4g4p3x/lme-rs/issues) or read
[the contributor guide](../CONTRIBUTING.md).
