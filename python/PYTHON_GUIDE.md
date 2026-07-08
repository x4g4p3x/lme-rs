# lme-rs Python Guide

This guide covers the Python bindings shipped in `python/`. The Python package is intentionally narrower than the Rust crate, but it already provides a practical workflow for fitting and using mixed-effects models from Python.

## Installation

### Prerequisites

- Python 3.8 or newer
- Rust toolchain via [rustup](https://rustup.rs)
- `maturin`

### Build and install

```bash
cd python
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install maturin polars pytest

maturin develop --release
```

If you are building with CPython 3.14, set the PyO3 forward-compatibility flag first:

```powershell
# PowerShell
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"

# CMD
set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# macOS / Linux
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

## What the Python package exposes

The Python module mirrors the Rust crate: formula fits, matrix OLS, contrasts, inference, prediction, and simulation. Type hints live in [`lme_python.pyi`](lme_python.pyi).

Top-level functions:

- `lme_python.lm(formula, data)`
- `lme_python.lm_matrix(y, x)` — numeric design matrix (Rust `lm(y, x)`)
- `lme_python.lmer(formula, data, reml=True)`
- `lme_python.lmer_weighted(formula, data, reml=True, weights=None)`
- `lme_python.glmer(formula, data, family_name, n_agq=1)`
- `lme_python.glmer_weighted(formula, data, family_name, n_agq=1, weights=None)`
- `lme_python.nlmer(formula, data, start=None, reml=False, n_agq=1)` — built-in nonlinear means (`SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`)
- `lme_python.nlmer_with_mean(formula, data, mean_fn, param_names, ...)` — user-defined nonlinear means
- `lme_python.contrast_matrix(p, rows)` — **L** from `(column_index, weight)` rows
- `lme_python.contrast_matrix_from_names(fixed_names, rows)` — **L** from coefficient names
- `lme_python.anova(fit_a, fit_b)` → `PyLikelihoodRatioAnova` (nested LRT)

Structured result types: `PyConfintResult`, `PySimulateResult`, `PyFixedEffectsAnova`, `PyContrastTest`, `PyLikelihoodRatioAnova`, `PyFamily`.

Available `PyLmeFit` methods:

- `summary()`
- `predict(newdata)`
- `predict_conditional(newdata, allow_new_levels=False)`
- `predict_conditional_response(newdata, allow_new_levels=False)`
- `predict_response(newdata)`
- `confint(level=0.95)` → `PyConfintResult` (indexable as `(lower, upper)` tuples via `ci[i]`)
- `simulate(nsim)` → `PySimulateResult` (use `.simulations` for the draw list)
- `with_robust_se(data, cluster_col=None)`  # sandwich standard errors
- `with_satterthwaite(data)`  # denominator df and p-values
- `with_kenward_roger(data)`  # Kenward-Roger denominator df and p-values
- `anova(ddf_method="satterthwaite", anova_type="III")` → `PyFixedEffectsAnova`
- `test_contrast(l_matrix, ddf_method="satterthwaite")`  # H₀: Lβ = 0
- `test_contrast_vs(l_matrix, beta_h, ddf_method="satterthwaite")`  # H₀: Lβ = β_h

Selected properties:

- `coefficients`, `b`, `fixed_names`, `formula`, `u`, `beta_se`, `fixed_term_assign`, `categorical_levels`, `v_beta_unscaled`
- `family_name`, `link_name`, `family` (GLMM / NLMM)
- `sigma2`, `theta`
- `aic`, `bic`, `log_likelihood`, `deviance`, `reml_criterion`
- `converged`, `iterations`, `num_obs`
- `std_errors`, `beta_t` (LMM t-values; GLMM z-values)
- `residuals`, `fitted`
- `ranef`  # random effects modes
- `var_corr`  # random-effects variance/covariance summary
- `robust_se`, `robust_t`, `robust_p_values`
- `satterthwaite_dfs`, `satterthwaite_p_values`
- `kenward_roger_dfs`, `kenward_roger_p_values`

## Quick start

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/sleepstudy.csv")
model = lme_python.lmer(
    "Reaction ~ Days + (Days | Subject)",
    data=df,
    reml=True,
)

print(model)
print(model.coefficients)
```

## Fitting models

### Fixed-effects-only linear models

`lm()` fits an ordinary least squares model — no random effects — using the same
Wilkinson formula syntax and Polars `DataFrame` input as `lmer()`:

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/sleepstudy.csv")
fit = lme_python.lm("Reaction ~ Days", data=df)

print(fit)                 # R-style summary
print(fit.coefficients)    # [intercept, slope]
print(fit.std_errors)      # standard errors
print(fit.aic, fit.bic)    # information criteria
```

Formula shorthand for intercept-only and no-intercept variants:

```python
# Intercept only
fit0 = lme_python.lm("Reaction ~ 1", data=df)

# Suppress intercept
fit_no_int = lme_python.lm("Reaction ~ 0 + Days", data=df)
```

### Linear mixed models

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/sleepstudy.csv")
model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df)
```

Set `reml=False` when you want ML instead of REML:

```python
model_ml = lme_python.lmer(
    "Reaction ~ Days + (Days | Subject)",
    data=df,
    reml=False,
)
```

### Generalized linear mixed models

`glmer()` in the Python binding currently accepts these family names:

- `"binomial"`
- `"poisson"`
- `"gamma"`
- `"gaussian"`

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/grouseticks.csv")
model = lme_python.glmer(
    "TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
    data=df,
    family_name="poisson",
)

# Optional non-canonical link (logit, probit, cloglog, log, identity, inverse, sqrt)
probit_fit = lme_python.glmer(
    "y ~ period2 + period3 + period4 + (1 | herd)",
    data=pl.read_csv("tests/data/cbpp_binary.csv"),
    family_name="binomial",
    link_name="probit",
)
```

Supported links per family match Rust [`family::Link`](../../src/family.rs): binomial — logit (default), probit, cloglog; poisson — log, identity, sqrt; gaussian — identity, log, inverse; gamma — inverse, identity, log.

Prior weights use `glmer_weighted(..., weights=[...])` (same validation as `lmer_weighted`).

### Nonlinear mixed models

Built-in means match R `stats::SS*` functions: `SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`.

When `start=None` (or an empty dict), the fitter uses R-style **`selfStart`** heuristics on `(covariate, response)` with multistart fallback to static defaults — the same path as omitting `start` in R `nlmer()`.

```python
df = pl.read_csv("tests/data/orange.csv")
fit = lme_python.nlmer(
    "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
    data=df,
    start={"Asym": 200.0, "xmid": 725.0, "scal": 350.0},
    reml=False,
)

# selfStart when starts are unknown:
fit_auto = lme_python.nlmer(
    "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
    data=df,
    start=None,
)
```

Explicit `start` overrides `selfStart`; partial dicts merge with defaults for missing parameter names.

Set `n_agq` to a value `≥ 2` for adaptive Gauss–Hermite quadrature on scalar random effects (`k = 1`); default `1` is Laplace only (same convention as `glmer`).

`predict()` and `predict_conditional()` work on NLMM fits: population predictions use fixed nonlinear parameters only; conditional predictions add stored random effects, including multivariate nonlinear-parameter effects such as `Asym + xmid | Tree`.

### Custom nonlinear means

Use `nlmer_with_mean` when the built-in `SS*` catalog does not cover your mean function. The formula layout is `response ~ covariate ~ re | group` (the middle segment is the covariate column name, not an `SS*` call). The callable `mean_fn(x, params)` must return `(mu, grad)` where `grad` has one partial derivative per name in `param_names`.

```python
import math

import polars as pl

import lme_python


def exp_mean(x: float, params: list[float]) -> tuple[float, list[float]]:
    a, b = params
    mu = a * math.exp(-b * x)
    return mu, [math.exp(-b * x), -x * a * math.exp(-b * x)]


df = pl.DataFrame({"y": [...], "x": [...], "g": [...]})
fit = lme_python.nlmer_with_mean(
    "y ~ x ~ a | g",
    data=df,
    mean_fn=exp_mean,
    param_names=["a", "b"],
    start={"a": 2.0, "b": 0.4},
)
```

When `start=None`, defaults use `1.0` for each parameter name (custom means do not run R `selfStart` heuristics).

## Predictions

### Population-level prediction

```python
newdata = pl.DataFrame({
    "Days": [0.0, 1.0, 5.0, 10.0],
    "Subject": ["308", "308", "308", "308"],
})

preds = model.predict(newdata)
```

### Conditional prediction

```python
preds_cond = model.predict_conditional(newdata, allow_new_levels=True)
```

Use `allow_new_levels=True` when the prediction frame may contain groups not seen during fitting.

### Response-scale prediction for GLMMs

```python
rates = model.predict_response(newdata)
```

For binomial models this returns probabilities. For poisson models it returns expected counts or rates on the response scale.

## Confidence intervals and summary data

```python
print(model.summary())
print(model.std_errors)
print(model.confint(level=0.95))
```

## Data expectations

The Python layer currently expects a `polars.DataFrame` or another object that can write itself to IPC via a `write_ipc` method. In practice, using `polars.DataFrame` directly is the supported path.

If your data is in pandas, convert it first:

```python
import pandas as pd
import polars as pl

pdf = pd.read_csv("data.csv")
df = pl.from_pandas(pdf)
```

## Current limitations

- The Python binding is not yet a full mirror of the Rust crate.
- Some advanced inference helpers are still not exposed in the Python binding (for example, detailed simulation helpers and some additional model comparison wrappers).
- `glmer()` currently exposes only the string family selector described above.
- The most reliable path for advanced inference remains the Rust API.

## Troubleshooting

### `maturin develop` fails

Make sure the Rust toolchain is installed and available in the active shell.

```bash
rustc --version
cargo --version
```

If those commands fail, install Rust via [rustup](https://rustup.rs).

If you are building with CPython 3.14, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before running `maturin develop` with the current pinned PyO3 version.

### `ModuleNotFoundError: lme_python`

Usually this means one of the following:

- the virtual environment is not activated
- `maturin develop --release` did not complete successfully
- a different Python interpreter is active than the one used for the build

### Prediction or fit errors

If fitting or prediction fails:

- confirm the formula column names match the DataFrame exactly
- ensure numeric predictors are actually numeric
- ensure grouping columns contain the expected identifiers
- try the same case through the Rust API or R `lme4` when debugging a difficult edge case

## Related documents

- Rust crate docs: [../GUIDE.md](../GUIDE.md)
- Cross-language comparisons: [../comparisons/COMPARISONS.md](../comparisons/COMPARISONS.md)
- Python examples: [examples/](examples/)
- End-to-end Python parity checks (fixtures + assertions): [examples/verification_project/README.md](examples/verification_project/README.md)
- Contributor setup: [../CONTRIBUTING.md](../CONTRIBUTING.md)
