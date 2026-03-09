# lme-rs Python Guide

Use `lme-rs` directly from Python via the `lme_python` bindings, built with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs). Get native Rust performance for mixed-effects models with a familiar Python/Polars workflow.

## Installation

### Prerequisites

- Python ≥ 3.8
- Rust toolchain (via [rustup](https://rustup.rs))
- [maturin](https://www.maturin.rs): `pip install maturin`

### Build & Install

```bash
cd python/
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install maturin polars
maturin develop --release
```

This compiles the Rust code and installs `lme_python` into your virtual environment.

---

## Quick Start

```python
import polars as pl
import lme_python

# Load data
df = pl.read_csv("tests/data/sleepstudy.csv")

# Fit a linear mixed-effects model
model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)

# Print R-style summary
print(model)
```

**Output:**

```text
Linear mixed model fit by REML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)

     AIC      BIC   logLik deviance
  1755.6   1774.8   -871.8   1743.6
REML criterion at convergence: 1743.6283

Random effects:
 Groups   Name        Variance Std.Dev.
 Subject  (Intercept) 611.9033 24.7367
          Days        35.0801  5.9228
 Corr:
  Days  0.066
 Residual             654.9417 25.5918
Number of obs: 180, groups: Subject, 18

Fixed effects:
            Estimate Std. Error t value
(Intercept) 251.4051     6.8238   36.84
Days         10.4673     1.5459    6.77
```

---

## API Reference

### `lme_python.lmer(formula, data, reml=True)`

Fit a linear mixed-effects model.

| Parameter | Type | Description |
| :-------- | :--- | :---------- |
| `formula` | `str` | Wilkinson formula (e.g. `"y ~ x + (x \| group)"`) |
| `data` | `polars.DataFrame` | Input data |
| `reml` | `bool` | `True` for REML (default), `False` for ML |

**Returns:** `PyLmeFit` object.

```python
# REML (default) — better variance estimates
model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df)

# ML — needed for model comparison
model_ml = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=False)
```

### `lme_python.glmer(formula, data, family_name)`

Fit a generalized linear mixed-effects model.

| Parameter | Type | Description |
| :-------- | :--- | :---------- |
| `formula` | `str` | Wilkinson formula |
| `data` | `polars.DataFrame` | Input data |
| `family_name` | `str` | `"binomial"`, `"poisson"`, or `"gamma"` |

```python
# Poisson GLMM (count data)
model = lme_python.glmer(
    "TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
    data=df,
    family_name="poisson"
)

# Binomial GLMM (binary outcomes)
model = lme_python.glmer(
    "y ~ x1 + x2 + (1 | group)",
    data=df,
    family_name="binomial"
)
```

### `PyLmeFit` Properties & Methods

**Properties** (accessed as `model.property`):

| Property | Type | Description |
| :------- | :--- | :---------- |
| `coefficients` | `list[float]` | Fixed-effects coefficients (β) |
| `fixed_names` | `list[str]` or `None` | Names of fixed effects |
| `sigma2` | `float` or `None` | Residual variance (σ²) |
| `aic` | `float` or `None` | Akaike Information Criterion |
| `bic` | `float` or `None` | Bayesian Information Criterion |
| `log_likelihood` | `float` or `None` | Log-likelihood |
| `deviance` | `float` or `None` | Model deviance |
| `converged` | `bool` or `None` | Whether the optimizer converged |
| `num_obs` | `int` | Number of observations |
| `std_errors` | `list[float]` or `None` | Standard errors of fixed effects |
| `residuals` | `list[float]` | Residuals (y − ŷ) |
| `fitted` | `list[float]` | Fitted values (ŷ) |

**Methods:**

| Method | Returns | Description |
| :----- | :------ | :---------- |
| `summary()` | `str` | R-style model summary |
| `predict(newdata)` | `list[float]` | Population-level predictions (Xβ) |
| `predict_conditional(newdata, allow_new_levels=False)` | `list[float]` | Conditional predictions (Xβ + Zb) |
| `predict_response(newdata)` | `list[float]` | Response-scale predictions (GLMM) |
| `confint(level=0.95)` | `list[tuple]` | Wald confidence intervals `[(lower, upper), ...]` |

---

## Examples

### Linear Mixed Model

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/sleepstudy.csv")

# Fit model
model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df)
print(model)

# Population-level predictions (fixed effects only)
newdata = pl.DataFrame({
    "Days": [0.0, 1.0, 5.0, 10.0],
    "Subject": ["308", "308", "308", "308"],
})
preds_pop = model.predict(newdata)
print(f"Pop Predictions: {preds_pop}")
# [251.405, 261.872, 303.742, 356.078]

# Conditional predictions (fixed + random effects)
# Set allow_new_levels=True if predicting for unseen groups
preds_cond = model.predict_conditional(newdata, allow_new_levels=True)
print(f"Cond Predictions: {preds_cond}")
```

### Confidence Intervals and Standard Errors

```python
import polars as pl
import lme_python

df = pl.read_csv("tests/data/sleepstudy.csv")
model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df)

# Standard errors for fixed effects
print(f"Standard Errors: {model.std_errors}")

# 95% Wald confidence intervals for fixed effects
# Returns list of tuples: [(lower1, upper1), (lower2, upper2), ...]
ci = model.confint(level=0.95)
print(f"95% CI: {ci}")
```

### Intercept-Only Model

```python
df = pl.read_csv("tests/data/dyestuff.csv")
model = lme_python.lmer("Yield ~ 1 + (1 | Batch)", data=df)
print(model)
```

### Poisson GLMM

```python
df = pl.read_csv("tests/data/grouseticks.csv")
model = lme_python.glmer(
    "TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
    data=df,
    family_name="poisson"
)
print(model)
```

### Binomial GLMM

```python
df = pl.read_csv("tests/data/cbpp_binary.csv")
model = lme_python.glmer(
    "y ~ period2 + period3 + period4 + (1 | herd)",
    data=df,
    family_name="binomial"
)
print(model)

# For GLMMs, predict_response() returns probabilities (Binomial) or rates (Poisson)
newdata = pl.DataFrame({
    "period2": [1.0, 0.0],
    "period3": [0.0, 1.0],
    "period4": [0.0, 0.0],
    "herd": ["1", "1"]
})
# Returns predictions on the response scale [0, 1]
probs = model.predict_response(newdata)
print(f"Probabilities: {probs}")
```

### Nested Random Effects

```python
df = pl.read_csv("tests/data/pastes.csv")
model = lme_python.lmer("strength ~ 1 + (1 | batch/cask)", data=df)
print(model)
```

### Crossed Random Effects

```python
df = pl.read_csv("tests/data/penicillin.csv")
model = lme_python.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data=df)
print(model)
```

---

## Integration with pandas

If your data is in a pandas DataFrame, convert it to Polars first:

```python
import pandas as pd
import polars as pl
import lme_python

# Load with pandas
pdf = pd.read_csv("data.csv")

# Convert to Polars
df = pl.from_pandas(pdf)

# Fit model
model = lme_python.lmer("y ~ x + (1 | group)", data=df)
```

---

## Troubleshooting

### `maturin develop` fails

Make sure you have the Rust toolchain installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

On Windows, install via [rustup-init.exe](https://rustup.rs).

### `ImportError: DLL load failed`

If you see DLL errors on Windows, ensure:

1. You're using the same Python in the venv that maturin targeted
2. Intel MKL libraries are accessible (they're statically linked, so this is rare)

### `ModuleNotFoundError: lme_python`

Ensure `maturin develop` completed successfully and you're in the activated virtual environment.

---

## Comparison with R and statsmodels

`lme_python` produces the **same numerical results** as R's `lme4`:

| Feature | R (`lme4`) | Python (`statsmodels`) | Python (`lme_python`) |
| :------ | :--------- | :--------------------- | :-------------------- |
| Algorithm | BOBYQA + PLS | Powell + EM | Nelder-Mead + PLS |
| Random slopes | ✅ | Limited | ✅ |
| Nested effects | ✅ | ✅ | ✅ |
| Crossed effects | ✅ | ✅ | ✅ |
| GLMM (Laplace) | ✅ | ❌ | ✅ |
| Speed | Fast (C++) | Slow (Python) | **Fastest** (Rust) |

See [examples/COMPARISONS.md](../examples/COMPARISONS.md) for full numerical parity results.
