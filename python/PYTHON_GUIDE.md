# lme-rs Python Guide

This guide covers the Python bindings shipped in `python/`. The Python package is intentionally narrower than the Rust crate, but it already provides a practical workflow for fitting and using mixed-effects models from Python.

## Installation

### Prerequisites

- Python 3.8 or newer
- Rust toolchain via [rustup](https://rustup.rs)
- [uv](https://docs.astral.sh/uv/) (recommended; also via [`mise.toml`](../mise.toml))

### Build and install

```bash
cd python
uv sync --extra dev --no-install-project
uv run maturin develop --release
```

Without uv, create a venv manually and `pip install` the `[project.optional-dependencies] dev` packages from `pyproject.toml`, then run `maturin develop --release`.

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
- `lme_python.prepare_lmer(formula, data)` → `PyLmerPrepared`
- `lme_python.fit_prepared(prepared, reml=True)`
- `lme_python.refit_lmer(formula, data, reml=True)`
- `lme_python.cv_grouped(formula, data, group, n_splits=5, reml=True, seed=None)` → `PyCvGroupedResult`
- `lme_python.lmer_weighted(formula, data, reml=True, weights=None)`
- `lme_python.glmer(formula, data, family_name, n_agq=1)`
- `lme_python.glmer_weighted(formula, data, family_name, n_agq=1, weights=None)`
- `lme_python.nlmer(formula, data, start=None, reml=False, n_agq=1)` — built-in nonlinear means (`SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`, `SSpower`)
- `lme_python.nlmer_with_mean(formula, data, mean_fn, param_names, ...)` — user-defined nonlinear means
- `lme_python.contrast_matrix(p, rows)` — **L** from `(column_index, weight)` rows
- `lme_python.contrast_matrix_from_names(fixed_names, rows)` — **L** from coefficient names
- `lme_python.anova(fit_a, fit_b)` → `PyLikelihoodRatioAnova` (nested LRT)

Structured result types: `PyConfintResult`, `PySimulateResult`, `PyFixedEffectsAnova`, `PyContrastTest`, `PyLikelihoodRatioAnova`, `PyFamily`, `PyLmerPrepared`, `PyCvFoldMetric`, `PyCvGroupedResult`.

Available `PyLmeFit` methods:

- `summary()`
- `predict(newdata)`
- `predict_conditional(newdata, allow_new_levels=False)`
- `predict_conditional_response(newdata, allow_new_levels=False)`
- `predict_response(newdata)`
- `confint(level=0.95)` → `PyConfintResult` (indexable as `(lower, upper)` tuples via `ci[i]`); uses **t** with Kenward–Roger or Satterthwaite dfs when those are on the fit
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

Built-in means match R `stats::SS*` where available, plus `SSpower` (`a * x^b + c`, MATLAB Curve Fitter `power2`): `SSlogis`, `SSasymp`, `SSfol`, `SSmicmen`, `SSgompertz`, `SSpower`.

When `start=None` (or an empty dict), the fitter uses R-style **`selfStart`** heuristics on `(covariate, response)` with multistart fallback to static defaults; validate against R `nlmer()` when your workflow depends on exact starting behavior.

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

# Grouped calibration (MATLAB power2): random effect on offset c per unit
cal = pl.read_csv("tests/data/sspower_synthetic.csv")
fit_power = lme_python.nlmer(
    "y ~ SSpower(x, a, b, c) ~ c|id",
    data=cal,
    start=None,
    reml=False,
)
```

Requires **x > 0**. `SSpower` is not in R `stats::SS*`; lme4 parity uses a custom R `selfStart` (see [`comparisons/nlmm_sspower.R`](../comparisons/nlmm_sspower.R)). Not a substitute for lmfit/MATLAB bounded single-curve fitting.

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

Call `with_satterthwaite(data)` or `with_kenward_roger(data)` before `confint()` to use t-based intervals with the corresponding denominator degrees of freedom.

## Repeated fits and cross-validation

### Amortized fitting

When fitting the same formula and data repeatedly (REML vs ML, grid search, bootstrap on fixed data), prepare once and refit:

```python
prep = lme_python.prepare_lmer("Reaction ~ Days + (1 | Subject)", data=df)
fit_reml = lme_python.fit_prepared(prep, reml=True)
fit_ml = lme_python.fit_prepared(prep, reml=False)
```

`refit_lmer(formula, data, reml=True)` combines prepare + fit when you do not need to hold the prepared object.

### Group-structure-preserving CV

`cv_grouped` performs k-fold cross-validation that keeps grouping units intact (e.g. all rows for one `Subject` stay in train or test):

```python
cv = lme_python.cv_grouped(
    "Reaction ~ Days + (1 | Subject)",
    data=df,
    group="Subject",
    n_splits=5,
    reml=True,
    seed=42,  # optional, for reproducible group shuffling
)
print(cv.rmse, cv.mae, cv.all_converged)
# cv.oof_predictions, cv.test_fold, cv.folds
```

Held-out groups are predicted with population-level fixed effects only (no subject-specific random effect). **LMM only**; `n_splits` must be between 2 and the number of unique groups. See [GUIDE.md § Repeated fits and cross-validation](../GUIDE.md#repeated-fits-and-cross-validation).

## Data expectations

Formula entry points accept **Polars**, **pandas**, or **PyArrow** tabular data. Internally everything is converted to a Polars `DataFrame` via IPC before the Rust engine runs — Polars remains the canonical representation.

| Input type | Notes |
|:-----------|:------|
| `polars.DataFrame` | Preferred; no conversion overhead beyond IPC |
| `pandas.DataFrame` | Converted with `polars.from_pandas` (requires `pandas` installed) |
| `pyarrow.Table` | Converted with `polars.from_arrow` (requires `pyarrow` installed) |

```python
import polars as pl
import lme_python

df = pl.read_csv("sleepstudy.csv")
fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
```

pandas users can pass a `DataFrame` directly:

```python
import pandas as pd
import lme_python

pdf = pd.read_csv("sleepstudy.csv")
fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=pdf, reml=True)
```

PyArrow pipelines can pass a `Table`:

```python
import pyarrow.csv as pacsv
import lme_python

table = pacsv.read_csv("sleepstudy.csv")
fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=table, reml=True)
```

**Categorical / string columns:** use string columns for grouping factors (same as Polars). With pandas, prefer explicit `str` dtypes or `category` columns whose levels match the formula; nullable integer columns may need casting before fitting.

## Current limitations

- Matrix-only `lm(y, x)` without a DataFrame is Rust-only.
- `cv_grouped` supports LMMs only (not GLMM/NLMM).
- Some advanced inference helpers are still not exposed in the Python binding (for example, detailed simulation helpers and some additional model comparison wrappers).
- `glmer()` currently exposes only the string family selector described above.

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
