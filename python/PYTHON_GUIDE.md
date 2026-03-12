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

## What the Python package currently exposes

Top-level functions:

- `lme_python.lmer(formula, data, reml=True)`
- `lme_python.glmer(formula, data, family_name)`

Available `PyLmeFit` methods:

- `summary()`
- `predict(newdata)`
- `predict_conditional(newdata, allow_new_levels=False)`
- `predict_response(newdata)`
- `confint(level=0.95)`

Selected properties:

- `coefficients`
- `fixed_names`
- `sigma2`
- `aic`, `bic`, `log_likelihood`, `deviance`
- `converged`, `num_obs`
- `std_errors`
- `residuals`, `fitted`

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

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/grouseticks.csv")
model = lme_python.glmer(
    "TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
    data=df,
    family_name="poisson",
)
```

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
- There is no Python API today for `predict_conditional_response()`, robust standard errors, Satterthwaite, Kenward-Roger, simulation, or ANOVA helpers.
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
- Cross-language comparisons: [../examples/COMPARISONS.md](../examples/COMPARISONS.md)
- Contributor setup: [../CONTRIBUTING.md](../CONTRIBUTING.md)
