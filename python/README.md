# lme-python

Python bindings for [lme-rs](https://github.com/x4g4p3x/lme-rs):
linear, generalized linear, and nonlinear mixed-effects models with
`lme4`-style formulas. Fitting runs in the same Rust engine used by the crate.

[Python guide](https://github.com/x4g4p3x/lme-rs/blob/master/python/PYTHON_GUIDE.md) ·
[Examples](https://github.com/x4g4p3x/lme-rs/blob/master/docs/EXAMPLES.md) ·
[Workflow scope](https://github.com/x4g4p3x/lme-rs/blob/master/USABILITY.md)

## Install

```bash
python -m pip install lme-python
```

Install **`lme-python`**, import **`lme_python`**. Polars is a package dependency.
A compatible binary wheel does not require a Rust toolchain or an R installation.

Check the selected version's [PyPI files](https://pypi.org/project/lme-python/#files)
for your Python interpreter, OS, and architecture. The release workflow builds
CPython 3.10 wheels; source-build CI tests Python 3.10–3.13. Those tests do not
imply that a wheel is published for each tested interpreter. If pip falls back
to a source build, follow [the source setup](https://github.com/x4g4p3x/lme-rs/blob/master/CONTRIBUTING.md#python-bindings).

## Quick start

This example needs no repository files:

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

The formula estimates a shared intercept and slope plus a random intercept
for each group. The small synthetic dataset is an API demonstration.
`predict` returns fixed-effects predictions; use `predict_conditional`
when you want the fitted group effects.

## Continue your analysis

The bindings expose `lm`, `lmer`, `glmer`, and `nlmer`, along with prediction,
intervals, simulation, grouped cross-validation, bootstrap refits, ANOVA,
contrasts, and LMM estimated marginal means.

- [Python guide](https://github.com/x4g4p3x/lme-rs/blob/master/python/PYTHON_GUIDE.md):
  data requirements, fitting, inference, and structured results.
- [Type reference](https://github.com/x4g4p3x/lme-rs/blob/master/python/lme_python.pyi):
  function signatures and result fields.
- [Supported workflows](https://github.com/x4g4p3x/lme-rs/blob/master/USABILITY.md):
  model-specific scope and limitations.
- [Troubleshooting](https://github.com/x4g4p3x/lme-rs/blob/master/docs/TROUBLESHOOTING.md):
  installation, data, convergence, and prediction problems.
- [Changelog](https://github.com/x4g4p3x/lme-rs/blob/master/CHANGELOG.md):
  release history. Use the matching repository tag for version-specific docs.

## Development

See [Contributing](https://github.com/x4g4p3x/lme-rs/blob/master/CONTRIBUTING.md)
for the locked development environment and extension build. After building,
use `uv run --no-sync` for examples and tests.

`task consumer:smoke` builds a wheel, verifies it in isolated environments,
and runs portable examples against the installed artifact.

## Repeated responses and fit diagnostics

Prepare once when fitting several response vectors against the same design:

```python
prepared = lme_python.prepare_lmer("Reaction ~ Days + (Days | Subject)", data)
control = lme_python.FitControl(max_iterations=2000, require_convergence=True)
fit = prepared.fit(y=data["Reaction"].to_list(), reml=True, control=control)
print(fit.diagnostics)
```

`prepare_lmer(..., weights=...)` retains observation precision weights. Responses
must contain finite values and match the original row count. Offsets are applied
exactly once. `prepare_glmer` also supports `prepared.fit(y=..., control=...)`.
The existing `fit_prepared(..., y=..., control=...)` and `fit_prepared_glmer` functions
remain available.

`FitControl` accepts `max_iterations`, `tolerance`, `max_inner_iterations`, starting
covariance parameters `start`, and `require_convergence`. The iteration limit is
per outer search stage. Nonlinear fits use `max_inner` and `max_outer_iters` instead.
Inspect `converged` and `diagnostics` before interpreting a fit; diagnostics report
termination, iteration counts, objective value, and requested/effective quadrature.

Conversion serializes columns needed by an explicit formula. Dot formulas retain
all columns for expansion. Native fitting releases the interpreter lock; custom
Python nonlinear mean callbacks reacquire it while evaluating Python code.
Bootstrap generates each response as its worker needs it, retaining only replicate
summaries. Thread counts no longer change BLAS/OpenMP process settings.
