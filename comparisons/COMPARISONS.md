# Comparing lme-rs with R, Python, and Julia Outputs

This document collects representative side-by-side outputs for `lme-rs` against reference mixed-effects tooling in R (`lme4`), Python (`statsmodels` where applicable), and Julia (`MixedModels.jl`). These comparisons are evidence that the covered workflows line up closely on the included fixtures; they are not a blanket claim that every model in every ecosystem is identical.

<a id="parity-validation"></a>

## Validation: how results are checked against `lme4` and other languages

### What is *actually* the same engine?

- **Rust (`lme-rs`)** — the primary implementation exercised by integration tests.
- **Python (`statsmodels`)** — used in most `comparisons/*.py` scripts as an independent (or approximate) reference implementation. Some scripts use row-replication or simplified random structures where `MixedLM` cannot match `lme4` one-to-one; see each subsection.
- **Python (`lme_python`, optional)** — PyO3 bindings that call the **same Rust code** as the crate. Install with `maturin develop` under `python/` if you need Python to mirror Rust exactly on a fixture.
- **R (`lme4`)** — the usual numerical reference for mixed models in this repo: `lmer` / `glmer` with the same formula and data.
- **Julia (`MixedModels.jl`)** — a separate implementation; generally close on standard LMM/GLMM examples but not guaranteed to match `lme4` or `lme-rs` to machine precision.

### Automated checks in the repository (CI / `cargo test`)

These tests re-fit models inside Rust and compare to **stored reference values** (often exported from R) or to **narrow numeric bands** inspired by `lme4` output:

| Area | Test module / data | Role |
|:-----|:---------------------|:-----|
| LMM, sleepstudy REML | [`tests/test_numerical_parity.rs`](../tests/test_numerical_parity.rs) vs [`tests/data/sleepstudy.csv`](../tests/data/sleepstudy.csv) | Fixed effects within ~`0.05` of documented `lme4` scalars; SEs, variance components, and REML deviance checked with tolerances documented in that file. |
| GLMM, CBPP binomial | [`tests/test_glmm.rs`](../tests/test_glmm.rs) vs [`tests/data/glmm_binomial.json`](../tests/data/glmm_binomial.json) | Coefficients and θ compared to values taken from R’s `glmer` fit (same formula as below); typically `\|Δβ\|, \|Δθ\| < 0.05`. |
| GLMM, Poisson grouseticks | Same `test_glmm.rs` vs [`tests/data/glmm_poisson.json`](../tests/data/glmm_poisson.json) | Looser tolerance on β (`0.15`) because optimizers differ (`Nelder–Mead` in `lme-rs` vs **BOBYQA** in `lme4`). |
| Other fixtures | e.g. [`tests/data/random_slopes.json`](../tests/data/random_slopes.json), [`tests/data/penicillin.json`](../tests/data/penicillin.json) | Additional REML / design-matrix checks in the test suite. |

AGQ-specific checks (scalar quadrature, `n_agq > 1`) live in the same GLMM tests and in unit tests in [`src/glmm_math.rs`](../src/glmm_math.rs): they assert finite deviances and, for CBPP at a fixed θ, Laplace vs AGQ marginal deviance within **5%** relative scale.

### Known reasons results are *not* identical everywhere

1. **Different optimizers** — `lme-rs` uses derivative-free search on variance parameters; `lme4` often uses **BOBYQA** (`nlopt`). Small shifts in β and θ are expected on awkward data (the grouseticks GLMM section below calls this out explicitly).
2. **GLMM deviance / likelihood constants** — reported `deviance`, `logLik`, and AIC/BIC can disagree in **level** with R because implementations omit or include different normalizing terms (e.g. Poisson `lgamma` terms); **fitted β, θ, and predictions** are still the right objects to compare. See the footnote in the Poisson GLMM subsection.
3. **Adaptive quadrature (`nAGQ > 1`)** — In **R**, `glmer(..., nAGQ = k)` uses AGQ inside the **θ** optimization. In **lme-rs**, variance components are optimized with the **Laplace** marginal likelihood; if you request `n_agq > 1`, **AGQ is applied in the final PIRLS / deviance evaluation** at that θ. So θ need not match R’s AGQ fit even when quadrature is implemented consistently. The scripts [`comparisons/glmm_cbpp.R`](glmm_cbpp.R), [`glmm_cbpp.py`](glmm_cbpp.py), and [`glmm_cbpp.rs`](glmm_cbpp.rs) document both Laplace and higher `nAGQ` / `n_agq` runs side by side.
4. **Julia** — `MixedModels.jl` does not expose the same GLMM AGQ controls as `lme4`; Julia rows in this file are mostly **Laplace / default**-equivalent comparisons.

### Manual reproduction and timing (not coefficient regression tests)

- **This file** — copy/pasteable commands and pasted console output for human review.
- **[`scripts/run_cross_language_benchmarks.py`](../scripts/run_cross_language_benchmarks.py)** — runs the `comparisons/*` examples (including `sleepstudy`, `pastes`, `cbpp`, `grouseticks`, `categorical`, `weighted`, `gaussian_glmm`) and writes **JSON timing** results; it does **not** assert numerical equality of estimates.
- **[`BENCHMARKS.md`](../BENCHMARKS.md)** — Criterion micro-benchmarks for the Rust crate (including GLMM / AGQ workloads), not cross-language statistical parity.

To re-run the automated parity tests locally: from the repository root, `cargo test` (optionally filter, e.g. `cargo test numerical_parity` or `cargo test test_glmm`).

![Parameter Estimates Comparison](./comparison_chart.png)

## The Model

All three examples fit the following linear mixed-effects model:

```text
Reaction ~ Days + (Days | Subject)
```

## 1. lme-rs Output (Rust)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)

     AIC      BIC   logLik deviance
  1755.6   1774.8   -871.8   1743.6
REML criterion at convergence: 1743.6283
Scaled residuals:
    Min      1Q  Median      3Q     Max 
-3.9536 -0.4628  0.0296  0.4659  5.1793

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

Predictions (Population-level) for Days `[0, 1, 5, 10]`:

```text
251.4051, 261.8724, 303.7415, 356.0780
```

---

## 2. R Output (`lme4`)

```R
library(lme4)
data <- read.csv("tests/data/sleepstudy.csv")
fit <- lmer(Reaction ~ Days + (Days | Subject), data = data, REML = TRUE)
summary(fit)
```

**Output:**

```text
REML criterion at convergence: 1743.6

Random effects:
 Groups   Name        Variance Std.Dev. Corr 
 Subject  (Intercept) 612.10   24.741        
          Days         35.07    5.922   0.07 
 Residual             654.94   25.592        
Number of obs: 180, groups:  Subject, 18

Fixed effects:
            Estimate Std. Error t value
(Intercept)  251.405      6.825  36.838
Days          10.467      1.546   6.771

Correlation of Fixed Effects:
     (Intr)
Days -0.138
```

Predictions (Population-level):

```text
       1        2        3        4 
251.4051 261.8724 303.7415 356.0780 
```

---

## 3. Python Output (`statsmodels`)

```python
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv("tests/data/sleepstudy.csv")
model = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"], re_formula="~Days")
result = model.fit(reml=True)
print(result.summary())
```

**Output:**

```text
            Mixed Linear Model Regression Results
==============================================================
Model:               MixedLM   Dependent Variable:   Reaction 
No. Observations:    180       Method:               REML     
No. Groups:          18        Scale:                654.9405 
Min. group size:     10        Log-Likelihood:       -871.8141
Max. group size:     10        Converged:            Yes      
Mean group size:     10.0                                     
--------------------------------------------------------------
                  Coef.  Std.Err.   z    P>|z|  [0.025  0.975]
--------------------------------------------------------------
Intercept        251.405    6.825 36.838 0.000 238.029 264.781
Days              10.467    1.546  6.771 0.000   7.438  13.497
Group Var        612.096   11.881                             
Group x Days Cov   9.605    1.821                             
Days Var          35.072    0.610                             
==============================================================
```

Predictions (Population-level):

```text
251.405105
261.872391
303.741535
356.077964
```

---

## 4. Julia Output (`MixedModels.jl`)

```julia
using CSV
using DataFrames
using MixedModels

df = CSV.read("tests/data/sleepstudy.csv", DataFrame)
form = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))
m1 = fit(MixedModel, form, df, REML=true)
println(m1)
```

**Output:**

```text
=== Model Summary ===
Linear Mixed Model fit by REML
 Formula: Reaction ~ 1 + Days + (1 + Days | Subject)
   posdef: [1 0; 0 1]
   REML criterion at convergence: 1743.628271500305

Variance components:
            Column    Variance  Std.Dev.  Corr.
Subject  (Intercept)  612.09031 24.74046
         Days          35.07167  5.92213 +0.07
Residual              654.94097 25.59181
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
──────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)
──────────────────────────────────────────────────
(Intercept)  251.405      6.82456  36.84    <1e-99
Days          10.4673     1.54579   6.77    <1e-10
──────────────────────────────────────────────────
```

Predictions (Population-level):

```text
251.405105
261.872391
303.741535
356.077964
```

---

## 5. Comparison Validation

Across these four implementations on this fixture, the fitted coefficients, variance components, and predictions line up closely:

| Parameter                 | R (`lme4`) | Python (`statsmodels`) | Julia (`MixedModels.jl`) | Rust (`lme-rs`) |
|:--------------------------|:-----------|:-----------------------|:-------------------------|:----------------|
| **Fixed Intercept**       | `251.405`  | `251.405`              | `251.405`                | `251.4051`      |
| **Fixed Slope (Days)**    | `10.467`   | `10.467`               | `10.467`                 | `10.4673`       |
| **Random Var: Intercept** | `612.10`   | `612.096`              | `612.090`                | `611.9033`      |
| **Random Var: Days**      | `35.07`    | `35.072`               | `35.072`                 | `35.0801`       |
| **Random Covariance/Corr**| `0.07`     | `9.605` (Cov) [1]      | `+0.07`                  | `0.066`         |
| **Residual Variance**     | `654.94`   | `654.9405`             | `654.941`                | `654.9417`      |
| **Total Deviance (REML)** | `1743.6`   | `-871.814` (LogLike)   | `1743.628`               | `1743.6283`     |

> *[1] Converting python Cov (9.605) to Correlation: `9.605 / (sqrt(612.096) * sqrt(35.072)) = 0.0655`*

These implementations are closely aligned on this example.

---

## Maximum Likelihood Estimation (REML = FALSE)

By defaulting to Maximum Likelihood Estimation (`REML=FALSE`), the optimization landscape shifts to evaluate the pure unpenalized likelihood parameters. The same `sleepstudy` model is run again:

### The ML Model

```text
Reaction ~ Days + (Days | Subject)    [REML=FALSE]
```

#### ML 1. R Output (`lme4`)

```text
=== Model Summary ===
Linear mixed model fit by maximum likelihood  ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)

     AIC      BIC   logLik deviance
  1763.9   1783.1   -876.0   1751.9

Random effects:
 Groups   Name        Variance Std.Dev. Corr
 Subject  (Intercept) 565.52   23.781       
          Days         32.68    5.717   0.08
 Residual             654.94   25.592       
Number of obs: 180, groups:  Subject, 18

Fixed effects:
            Estimate Std. Error t value
(Intercept)  251.405      6.632   37.91
Days          10.467      1.502    6.97
```

#### ML 2. lme-rs Output (Rust)

```text
=== Model Summary ===
Linear mixed model fit by ML ['lmerMod']
Formula: Reaction ~ Days + (Days | Subject)

     AIC      BIC   logLik deviance
  1763.9   1783.1   -876.0   1751.9

Random effects:
 Groups   Name        Variance Std.Dev.
 Subject  (Intercept) 565.4802 23.7798
          Days        32.6735  5.7161  
 Corr:
  Days  0.081
 Residual             654.9621 25.5922
Number of obs: 180, groups: Subject, 18

Fixed effects:
            Estimate Std. Error t value
(Intercept) 251.4051     6.6322   37.91
Days         10.4673     1.5021    6.97
```

#### ML 3. Python Output (`statsmodels`)

```text
             Mixed Linear Model Regression Results
==============================================================
Model:               MixedLM   Dependent Variable:   Reaction 
No. Observations:    180       Method:               ML       
No. Groups:          18        Scale:                654.9400 
Min. group size:     10        Log-Likelihood:       -875.9697
Max. group size:     10        Converged:            Yes      
Mean group size:     10.0                                     
--------------------------------------------------------------
                  Coef.  Std.Err.   z    P>|z|  [0.025  0.975]
--------------------------------------------------------------
Intercept        251.405    6.632 37.906 0.000 238.406 264.404
Days              10.467    1.502  6.968 0.000   7.523  13.412
Group Var        565.519   10.938                             
Group x Days Cov  11.056    1.671                             
Days Var          32.683    0.561                             
==============================================================
```

Predictions (Population-level):

```text
0    251.405105
1    261.872391
2    303.741535
3    356.077964
```

#### ML 4. Julia Output (`MixedModels.jl`)

```text
=== Model Summary ===
Linear Mixed Model fit by maximum likelihood
 Formula: Reaction ~ 1 + Days + (1 + Days | Subject)
   logLik   deviance     AIC      AICc       BIC   
 -875.9697  1751.9393  1763.9393  1764.4249  1783.0971

Variance components:
            Column    Variance  Std.Dev.   Corr.
Subject  (Intercept)  565.51065 23.78047
         Days          32.68212  5.71683  +0.08
Residual              654.94145 25.59182
 Number of obs: 180; levels of grouping factors: 18

  Fixed-effects parameters:
──────────────────────────────────────────────────
                Coef.  Std. Error      z  Pr(>|z|)
──────────────────────────────────────────────────
(Intercept)  251.405       6.6322  37.91    <1e-99
Days          10.4673      1.5022   6.97    <1e-11
──────────────────────────────────────────────────
```

#### ML Comparison Validation

With ML evaluation properly discounting the design structure penalty of the fixed effects parameters (unlike REML), the degrees of freedom fundamentally shift the underlying bounds:

| Parameter                 | R (`lme4`) | Python (`statsmodels`) | Julia (`MixedModels.jl`) | Rust (`lme-rs`) |
|:--------------------------|:-----------|:-----------------------|:-------------------------|:----------------|
| **Fixed Intercept**       | `251.405`  | `251.405`              | `251.405`                | `251.4051`      |
| **Fixed Slope (Days)**    | `10.467`   | `10.467`               | `10.467`                 | `10.4673`       |
| **Random Var: Intercept** | `565.52`   | `565.519`              | `565.510`                | `565.4802`      |
| **Random Var: Days**      | `32.68`    | `32.683`               | `32.682`                 | `32.6735`       |
| **Random Covariance/Corr**| `0.08`     | `11.056` (Cov) [1]     | `+0.08`                  | `0.081`         |
| **Residual Variance**     | `654.94`   | `654.9400`             | `654.941`                | `654.9621`      |
| **Total Deviance (ML)**   | `1751.9`   | `-875.969` (LogLike)   | `1751.939`               | `1751.9393`     |

> *[1] Converting python Cov (11.056) to Correlation: `11.056 / (sqrt(565.519) * sqrt(32.683)) = 0.0813`*

The ML fits remain closely aligned on this example, with `lme-rs` landing in the same neighborhood as the R, Python, and Julia reference outputs.

---

## Intercept-Only Linear Mixed Models (Dyestuff)

To verify the simplest baseline mixed-effects model, we fit the classic `Dyestuff` dataset from `lme4`. The model predicts the `Yield` containing only a fixed global intercept and a random intercept tied to the `Batch`.

### The Baseline LMM Model

```text
Yield ~ 1 + (1 | Batch)
```

#### Baseline 1. R Output (`lme4`)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: Yield ~ 1 + (1 | Batch)

REML criterion at convergence: 319.7

Random effects:
 Groups   Name        Variance Std.Dev.
 Batch    (Intercept) 1764     42.00   
 Residual             2451     49.51   
Number of obs: 30, groups:  Batch, 6

Fixed effects:
            Estimate Std. Error t value
(Intercept)  1527.50      19.38    78.8
```

#### Baseline 2. lme-rs Output (Rust)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: Yield ~ 1 + (1 | Batch)

     AIC      BIC   logLik deviance
   325.7    329.9   -159.8    319.7
REML criterion at convergence: 319.6543

Random effects:
 Groups   Name        Variance Std.Dev.
 Batch    (Intercept) 1764.4592 42.0055 
 Residual             2451.1613 49.5092 
Number of obs: 30, groups: Batch, 6

Fixed effects:
            Estimate Std. Error t value
(Intercept) 1527.5000    19.3851   78.80
```

#### Baseline 3. Python Output (`statsmodels`)

```text
          Mixed Linear Model Regression Results
==========================================================
Model:             MixedLM  Dependent Variable:  Yield    
No. Observations:  30       Method:              REML     
No. Groups:        6        Scale:               2451.2239
Min. group size:   5        Log-Likelihood:      -159.8271
Max. group size:   5        Converged:           Yes      
Mean group size:   5.0                                    
----------------------------------------------------------
           Coef.   Std.Err.   z    P>|z|  [0.025   0.975] 
----------------------------------------------------------
Intercept 1527.500   19.384 78.802 0.000 1489.508 1565.492
Group Var 1764.171   31.658                               
==========================================================
```

#### Baseline 4. Julia Output (`MixedModels.jl`)

```text
=== Model Summary ===
Linear mixed model fit by REML
 Yield ~ 1 + (1 | Batch)
 REML criterion at convergence: 319.6542768422576

Variance components:
            Column    Variance Std.Dev.
Batch    (Intercept)  1764.0503 42.0006
Residual              2451.2499 49.5101
 Number of obs: 30; levels of grouping factors: 6

  Fixed-effects parameters:
────────────────────────────────────────────────
              Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────
(Intercept)  1527.5     19.3834  78.80    <1e-99
────────────────────────────────────────────────
```

#### Baseline Conclusion

The intercept-only LMM is closely aligned across these systems on this fixture: the fitted intercept agrees at `1527.5`, and the random-intercept and residual variance split remains in the same range across implementations.

---

## Generalized Linear Mixed Models (GLMM)

In addition to standard LMMs, `lme-rs` supports GLMM architectures utilizing penalised iteratively reweighed least squares (PIRLS) and the Laplace approximation.

To contrast this, we evaluate the `grouseticks` dataset, fitting expected tick counts based on `YEAR` and `HEIGHT` clustered around the family `BROOD`.

**Note**: Python's `statsmodels` does not expose a directly comparable Laplace Poisson GLMM workflow here, so this comparison is made against R's `lme4::glmer` and Julia's `MixedModels.jl`.

### The GLMM Model

```text
TICKS ~ YEAR + HEIGHT + (1 | BROOD)   [Family: Poisson, Link: Log]
```

#### GLMM 1. R Output (`lme4`)

```text
=== Model Summary ===
Generalized linear mixed model fit by maximum likelihood (Laplace
  Approximation) [glmerMod]
 Family: poisson  ( log )
Formula: TICKS ~ YEAR + HEIGHT + (1 | BROOD)

      AIC       BIC    logLik -2*log(L)  df.resid 
   2038.1    2054.1   -1015.1    2030.1       399 

Random effects:
 Groups Name        Variance Std.Dev.
 BROOD  (Intercept) 1.56     1.249   

Fixed effects:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept) 58.703395  15.701307   3.739 0.000185 ***
YEAR        -0.482170   0.163398  -2.951 0.003169 ** 
HEIGHT      -0.025612   0.000804 -31.856  < 2e-16 ***
```

Predictions (Expected Counts for 3 new broods):

```text
8.7596613, 0.6763327, 1.5028652 
```

#### GLMM 2. lme-rs Output (Rust)

```text
=== Model Summary ===
Generalized linear mixed model fit by ML (Laplace) ['glmerMod']
 Family: poisson ( log )
Formula: TICKS ~ YEAR + HEIGHT + (1 | BROOD)

     AIC      BIC   logLik deviance
  1108.1   1124.1   -550.1   1100.1 [2]

Random effects:
 Groups   Name        Variance Std.Dev.
 BROOD    (Intercept) 1.5547   1.2469
Number of obs: 403, groups: BROOD, 118

Fixed effects:
            Estimate Std. Error z value
(Intercept)  55.3223    15.8417    3.49
YEAR         -0.4538     0.1630   -2.78
HEIGHT       -0.0239     0.0037   -6.47
```

Predictions (Expected Counts for 3 new broods):

```text
8.8813905, 0.8108679, 1.7046553
```

#### GLMM 3. Julia Output (`MixedModels.jl`)

```text
=== Model Summary ===
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  TICKS ~ 1 + YEAR + HEIGHT + (1 | BROOD)
  Distribution: Poisson{Float64}
  Link: LogLink()

   logLik    deviance     AIC       AICc        BIC    
 -1015.0697  1099.1407  2038.1394  2038.2399  2054.1352

Variance components:
         Column   VarianceStd.Dev.
BROOD (Intercept)  1.55867 1.24847

Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  56.981      15.9418      3.57    0.0004
YEAR         -0.464541    0.164035   -2.83    0.0046
HEIGHT       -0.0255459   0.0037242  -6.86    <1e-11
────────────────────────────────────────────────────
```

Predictions (Expected Counts for 3 new broods):

```text
8.7294866, 0.6784891, 1.5293904
```

#### GLMM Comparison Validation

Because R warns about variables needing to be rescaled on this data set (`Rescale variables? Model is nearly unidentifiable`), the optimization paths drift apart slightly, but the convergence boundaries find extremely similar, structured topologies across all implementations:

| Parameter                 | R (`glmer`) | Julia (`MixedModels.jl`) | Rust (`lme-rs`) |
|:--------------------------|:------------|:-------------------------|:----------------|
| **Fixed Intercept**       | `58.703`    | `56.981`                 | `55.322`        |
| **Fixed Slope (YEAR)**    | `-0.482`    | `-0.464`                 | `-0.453`        |
| **Fixed Slope (HEIGHT)**  | `-0.025`    | `-0.025`                 | `-0.023`        |
| **Random Var: BROOD**     | `1.560`     | `1.558`                  | `1.555`         |
| **Expected Ticks (P1)**   | `8.75`      | `8.72`                   | `8.88`          |

> *[2] GLMM AIC/BIC note: `lme-rs` computes the Laplace-approximated conditional deviance (`sum(dev_resid) + log|A| + u'u`) for optimization, while R's `logLik()` includes additional data-dependent constants (e.g., `lgamma(y+1)` for Poisson, observation-level normalization). This yields different absolute AIC values while still allowing the fitted parameter values to be compared directly.*

R reports a convergence warning on this dataset (`max|grad| = 0.089424 (tol = 0.002)`), which is a useful reminder that this is a numerically awkward example. Julia and `lme-rs` converge to nearby solutions; the differences above should be interpreted in that context rather than as exact equality claims.

---

## Generalized Linear Mixed Models (Binomial)

Finally, we test a Binomial GLMM with a Logit link, using the `cbpp` dataset. The goal is to predict the incidence of contagious bovine pleuropneumonia across different herds and time periods.

### The Binomial Model

```text
y ~ period2 + period3 + period4 + (1 | herd)   [Family: Binomial, Link: Logit]
```

#### Binomial 1. R Output (`lme4`)

```text
=== Model Summary ===
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation)
 Family: binomial  ( logit )
Formula: y ~ period2 + period3 + period4 + (1 | herd)

      AIC       BIC    logLik -2*log(L)  df.resid 
    565.0     588.7    -277.5     555.0       837 

Random effects:
 Groups Name        Variance Std.Dev.
 herd   (Intercept) 0.4123   0.6421  
Number of obs: 842, groups:  herd, 15

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -1.3983     0.2312  -6.048 1.47e-09 ***
period2      -0.9919     0.3032  -3.272 0.001068 ** 
period3      -1.1282     0.3228  -3.495 0.000475 ***
period4      -1.5797     0.4221  -3.743 0.000182 ***
```

Predictions (Probabilities for Herd 1 across all 4 periods):

```text
0.19808139, 0.08391910, 0.07401811, 0.04842633 
```

#### Binomial 2. lme-rs Output (Rust)

```text
=== Model Summary ===
Generalized linear mixed model fit by ML (Laplace) ['glmerMod']
 Family: binomial ( logit )
Formula: y ~ period2 + period3 + period4 + (1 | herd)

     AIC      BIC   logLik deviance
   565.1    588.7   -277.5    555.1

Random effects:
 Groups   Name        Variance Std.Dev.
 herd     (Intercept) 0.4124   0.6422
Number of obs: 842, groups: herd, 15

Fixed effects:
            Estimate Std. Error z value
(Intercept)  -1.3605     0.2276   -5.98
period2      -0.9761     0.3033   -3.22
period3      -1.1110     0.3235   -3.43
period4      -1.5596     0.4245   -3.67
```

Predictions (Probabilities for Herd 1 across all 4 periods):

```text
0.204153, 0.088133, 0.077877, 0.051166
```

#### Binomial 3. Python Output (`statsmodels` — variational Bayes mixed GLM)

[`comparisons/glmm_cbpp.py`](glmm_cbpp.py) uses `BinomialBayesMixedGLM.fit_vb()` — a **different** likelihood approximation than Laplace / AGQ in `lme4` or `lme-rs`. It is included so the cross-language benchmark script runs **without** installing `lme_python`. Posterior means are in the same ballpark as `glmer` on this fixture:

```text
               Binomial Mixed GLM Results
========================================================
          Type Post. Mean Post. SD   SD  SD (LB) SD (UB)
--------------------------------------------------------
Intercept    M    -1.4093   0.1120                      
period2      M    -0.9832   0.2529                      
period3      M    -1.1199   0.2759                      
period4      M    -1.5644   0.3745                      
herd         V    -0.3707   0.1818 0.690   0.480   0.993
========================================================
```

Inverse-logit of the fixed-only linear predictor (four period patterns):

```text
0.1963, 0.0837, 0.0738, 0.0486
```

#### Binomial 4. Julia Output (`MixedModels.jl`)

```text
=== Model Summary ===
Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = 1)
  y ~ 1 + period2 + period3 + period4 + (1 | herd)
  Distribution: Binomial{Float64}
  Link: LogitLink()

   logLik   deviance     AIC      AICc       BIC    
 -277.5312  555.0623  565.0623  565.1341  588.7420

Variance components:
         Column   VarianceStd.Dev.
herd (Intercept)  0.41221 0.642036

Fixed-effects parameters:
───────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)
───────────────────────────────────────────────────
(Intercept)  -1.39853     0.227891  -6.14    <1e-09
period2      -0.992335    0.305385  -3.25    0.0012
period3      -1.12867     0.326049  -3.46    0.0005
period4      -1.58031     0.428795  -3.69    0.0002
───────────────────────────────────────────────────
```

Predictions (Probabilities for Herd 1):

```text
0.198049, 0.083872, 0.073973, 0.048390
```

#### Binomial Conclusion

The binomial GLMM fits are closely aligned across these implementations: the deviance, variance component, and fixed-effect estimates all fall in a narrow range, which is the practical signal this comparison is meant to show. **Regression coverage:** `cargo test` compares the Rust fit to R-derived coefficients and θ in [`tests/data/glmm_binomial.json`](../tests/data/glmm_binomial.json) (see [Validation](#parity-validation) above). Python’s `statsmodels` row above is a **variational** benchmark, not a Laplace match; use Rust or R for exact agreement with `glmer`.

---

## Nested Random Effects (Pastes)

To verify the library's ability to resolve **nested** hierarchical random effects, we evaluate the `pastes` dataset from `lme4`. The model predicts the `strength` of paste with random deviations for both the `batch` and the `cask` nested within that `batch` (`batch:cask`).

### The Nested LMM Model

```text
strength ~ 1 + (1 | batch/cask)
```

#### Nested 1. R Output (`lme4`)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: strength ~ 1 + (1 | batch/cask)

REML criterion at convergence: 247

Random effects:
 Groups     Name        Variance Std.Dev.
 cask:batch (Intercept) 8.434    2.9041  
 batch      (Intercept) 1.657    1.2874  
 Residual               0.678    0.8234  
Number of obs: 60, groups:  cask:batch, 30; batch, 10

Fixed effects:
            Estimate Std. Error t value
(Intercept)  60.0533     0.6769   88.72
```

#### Nested 2. lme-rs Output (Rust)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: strength ~ 1 + (1 | batch/cask)

     AIC      BIC   logLik deviance
   255.0    263.4   -123.5    247.0
REML criterion at convergence: 246.9907

Random effects:
 Groups   Name        Variance Std.Dev.
 batch    (Intercept) 1.6572   1.2873  
 batch:cask (Intercept) 8.4317   2.9037  
 Residual             0.6781   0.8235  
Number of obs: 60, groups: batch, 10; batch:cask, 30

Fixed effects:
            Estimate Std. Error t value
(Intercept)  60.0533     0.6768   88.73
```

#### Nested 3. Python Output (`statsmodels` — single-grouping proxy)

[`comparisons/lmm_pastes.py`](lmm_pastes.py) fits `strength ~ 1` with **one** random intercept at the `sample` (`batch:cask`) level. That is **not** the full two-component nested model in R/Julia/Rust; it is the closest `MixedLM` recipe that runs without extra packages.

```text
         Mixed Linear Model Regression Results
=======================================================
Model:            MixedLM Dependent Variable: strength 
No. Observations: 60      Method:             REML     
No. Groups:       30      Scale:              0.6780   
...
Intercept   60.053    0.586 102.412 0.000 ...
Group Var    9.977    4.614                            
=======================================================
```

#### Nested 4. Julia Output (`MixedModels.jl`)

```text
=== Model Summary ===
Linear mixed model fit by REML
 Formula: strength ~ 1 + (1 | batch) + (1 | batch & cask)
   REML criterion at convergence: 246.9906644268612

Variance components:
                Column   Variance Std.Dev. 
batch & cask (Intercept)  8.433675 2.904079
batch        (Intercept)  1.657302 1.287362
Residual                  0.678000 0.823407
 Number of obs: 60; levels of grouping factors: 30, 10

  Fixed-effects parameters:
─────────────────────────────────────────────────
               Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────
(Intercept)  60.0533     0.67687  88.72    <1e-99
─────────────────────────────────────────────────
```

#### Nested Random Effects Conclusion

The nested random-effects model expands `1 | batch/cask` the same way these reference implementations do for this fixture, and the fitted intercept and variance components remain closely aligned across **R, Julia, and Rust**. Python’s `statsmodels` row is a **partial** one-factor summary for tooling parity only.

---

## Crossed Random Effects (Penicillin)

To verify the library's ability to resolve multiple, non-nested (crossed) random effects groups, we evaluate the classic `penicillin` dataset. We predict the `diameter` of the clearing zone based on a global intercept, with random deviations for both the `plate` and the `sample`.

### The LMM Model

```text
diameter ~ 1 + (1 | plate) + (1 | sample)
```

#### Crossed 1. R Output (`lme4`)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: diameter ~ 1 + (1 | plate) + (1 | sample)

REML criterion at convergence: 330.9

Random effects:
 Groups   Name        Variance Std.Dev.
 plate    (Intercept) 0.7169   0.8467  
 sample   (Intercept) 3.7311   1.9316  
 Residual             0.3024   0.5499  
Number of obs: 144, groups:  plate, 24; sample, 6

Fixed effects:
            Estimate Std. Error t value
(Intercept)  22.9722     0.8086   28.41
```

#### Crossed 2. lme-rs Output (Rust)

```text
=== Model Summary ===
Linear mixed model fit by REML ['lmerMod']
Formula: diameter ~ 1 + (1 | plate) + (1 | sample)

     AIC      BIC   logLik deviance
   338.9    350.7   -165.4    330.9
REML criterion at convergence: 330.8606

Random effects:
 Groups   Name        Variance Std.Dev.
 plate    (Intercept) 0.7170   0.8467  
 sample   (Intercept) 3.7318   1.9318  
 Residual             0.3024   0.5499  
Number of obs: 144, groups: plate, 24; sample, 6

Fixed effects:
            Estimate Std. Error t value
(Intercept)  22.9722     0.8087   28.41
```

#### Crossed 3. Python Output (`statsmodels` — plate random intercept only)

[`comparisons/lmm_penicillin.py`](lmm_penicillin.py) uses a **single** `groups` column (`plate`) because `MixedLM` does not fit two crossed random intercepts in one call. The population intercept still matches the crossed models; variance components do **not**.

```text
         Mixed Linear Model Regression Results
=======================================================
Model:            MixedLM Dependent Variable: diameter 
...
Intercept   22.972    0.179 128.476 0.000 ...
Group Var    0.095    0.123                            
=======================================================
```

#### Crossed 4. Julia Output (`MixedModels.jl`)

```text
=== Model Summary ===
Linear mixed model fit by REML
 Formula: diameter ~ 1 + (1 | plate) + (1 | sample)
   posdef: [1]
           [1]
   REML criterion at convergence: 330.860588999605

Variance components:
            Column   Variance Std.Dev. 
plate    (Intercept)  0.716908 0.846704
sample   (Intercept)  3.730903 1.931555
Residual              0.302415 0.549923
 Number of obs: 144; levels of grouping factors: 24, 6

  Fixed-effects parameters:
─────────────────────────────────────────────────
               Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────
(Intercept)  22.9722    0.808572  28.41    <1e-99
─────────────────────────────────────────────────
```

#### Crossed Random Effects Conclusion

The crossed-effects example shows the same overall structure across **R, Julia, and Rust** on this fixture: the fitted intercept and the `plate` and `sample` variance components all land in the same range. Python (`statsmodels`) matches the **intercept** only in the one-factor proxy above.

---

## Weighted linear mixed models (`lmer_weighted`)

Prior case weights (as in `lme4::lmer(..., weights = w)`) scale each observation’s contribution to the REML objective. The examples [`comparisons/lmer_weighted.rs`](lmer_weighted.rs), [`lmer_weighted.R`](lmer_weighted.R), [`lmer_weighted.jl`](lmer_weighted.jl), and [`lmer_weighted.py`](lmer_weighted.py) use **sleepstudy** with the same synthetic weight vector as `benches/bench_math`: \(w_i = 0.5 + (i \bmod 5)\times 0.1\).

**Python (`statsmodels`)** does not implement prior weights in `MixedLM.fit` the same way as `lme4`. The Python script approximates weights by **replicating rows** in proportion to \(w_i\) and then fits REML `MixedLM` on the expanded table; fixed-effect estimates will **not** match R/Rust/Julia exactly. Use R, Julia, or Rust for direct weighted REML parity.

### Weighted 1. lme-rs Output (Rust)

```text
Prior weights w_i = 0.5 + (row_index mod 5) * 0.1 (same pattern as `benches/bench_math`)
...
Fixed effects:
            Estimate Std. Error t value
(Intercept) 250.1770     7.0978   35.25
Days         10.6710     1.5717    6.79
```

Population-level predictions for Subject `308` at Days `[0, 1, 5, 10]`:

```text
250.1770, 260.8480, 303.5318, 356.8865
```

### Weighted 2. R Output (`lme4`)

Run from the repo root:

```r
library(lme4)
data <- read.csv("tests/data/sleepstudy.csv")
data$w <- 0.5 + ((seq_len(nrow(data)) - 1) %% 5) * 0.1
fit <- lmer(Reaction ~ Days + (Days | Subject), data = data, weights = w, REML = TRUE)
summary(fit)
```

Expect fixed effects and predictions in the same neighborhood as the Rust row above.

### Weighted 3. Julia (`MixedModels.jl`)

See [`comparisons/lmer_weighted.jl`](lmer_weighted.jl) — `fit(MixedModel, ..., wts = ...)` with the same weight vector.

### Weighted 4. Python (`statsmodels` — expanded-data proxy)

See [`comparisons/lmer_weighted.py`](lmer_weighted.py). This is a **benchmark-only** approximation; coefficients differ from the weighted `lmer` fit.

---

## Gaussian family for `glmer` (identity link)

`family = gaussian` with the default identity link is the same likelihood as a **linear mixed model fit by ML** (as in `lme4::glmer` and `lmer(..., REML = FALSE)`). In **lme-rs**, `glmer(..., Family::Gaussian)` delegates to the LMM engine with `reml = false` so variance components and \(\sigma^2\) stay on the same scale as `lmer` ML (the Laplace GLMM objective for \(\theta\) would not match that parameterization). The `n_agq` argument is ignored for this family.

Examples: [`comparisons/glmm_gaussian.rs`](glmm_gaussian.rs), [`glmm_gaussian.R`](glmm_gaussian.R), [`glmm_gaussian.jl`](glmm_gaussian.jl), [`glmm_gaussian.py`](glmm_gaussian.py) on **sleepstudy** with `Reaction ~ Days + (1 | Subject)`.

**Python** uses `statsmodels` `MixedLM` with one random intercept per `Subject` (**REML**). **Rust** uses **ML** here, so standard errors and variance splits can differ slightly from the Python row; both are in the same neighborhood as `lme4`.

### Gaussian GLMM 1. lme-rs Output (Rust)

```text
Generalized linear mixed model fit by ML ['glmerMod']
 Family: gaussian ( identity )
Formula: Reaction ~ Days + (1 | Subject)

Random effects:
 Groups   Name        Variance Std.Dev.
 Subject  (Intercept) 1296.8967 36.0125 
 Residual             954.5258 30.8954 

Fixed effects:
            Estimate Std. Error z value
(Intercept) 251.4051     9.5063   26.45
Days         10.4673     0.8017   13.06
```

Population-level response-scale predictions:

```text
251.4051, 261.8724
```

### Gaussian GLMM 2. R Output (`lme4::glmer`)

```r
library(lme4)
data <- read.csv("tests/data/sleepstudy.csv")
summary(glmer(Reaction ~ Days + (1 | Subject), data = data, family = gaussian(link = "identity")))
```

### Gaussian GLMM 3. Julia / Python

See [`glmm_gaussian.jl`](glmm_gaussian.jl) and [`glmm_gaussian.py`](glmm_gaussian.py) for `MixedModels.jl` and `statsmodels` summaries on the same formula.

---

## 4. Categorical Dummy Encoding & Multi-DoF ANOVA

To test string parsing and automatic Multi-DoF Satterthwaite representations, we track the `cask` factor (a property with 3 levels) from `comparisons/categorical_anova.*` over the `pastes` dataset. 

In `lme-rs`, passing `cask` to `strength ~ cask + (1 | batch)` transparently generates `caskb` and `caskc` dummy encodings, then groups them successfully back together for a 2-DoF Joint Wald F-Test utilizing conservative pooled DDF approximations.

### 1. lme-rs Output (Rust)

```text
Linear mixed model fit by REML ['lmerMod']
Formula: strength ~ cask + (1 | batch)

     AIC      BIC   logLik deviance
   306.0    316.5   -148.0    296.0
REML criterion at convergence: 296.0278
Scaled residuals:
    Min      1Q  Median      3Q     Max 
-1.3717 -0.7739  0.0313  0.6771  1.8075

Random effects:
 Groups   Name        Variance Std.Dev.
 batch    (Intercept) 3.3670   1.8349  
 Residual             7.3051   2.7028  
Number of obs: 60, groups: batch, 10

Fixed effects:
            Estimate Std. Error       df t value Pr(>|t|) [Satterthwaite]
(Intercept)  59.2950     0.8378    20.02   70.77   0.0000
caskb         0.8500     0.8547    48.00    0.99   0.3250
caskc         1.4250     0.8547    48.00    1.67   0.1020

optimizer (Nelder-Mead) converged in 10 iterations

Type III Analysis of Variance Table with Satterthwaite's method
Term               NumDF     DenDF   F value    Pr(>F)
cask                   2   48.0042    1.4071  2.5476e-1 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

### 2. R Output (`lmerTest` + `car::Anova`)

```r
fit <- lmer(strength ~ cask + (1 | batch), data = data, REML = TRUE)
anova(fit, ddf="Satterthwaite")
```

The underlying ANOVA test from R provides the benchmark for our Joint Multi-DoF Wald approximations natively:

```text
Fixed effects:
            Estimate Std. Error t value
(Intercept)  59.2950     0.8378  70.778
caskb         0.8500     0.8547   0.995
caskc         1.4250     0.8547   1.667

Type III Analysis of Variance Table with Satterthwaite's method
       Sum Sq Mean Sq NumDF  DenDF F value Pr(>F)
cask   20.558  10.279     2 48.004  1.4071 0.2548
```

*Note: Due to variations in spectral eigenvalue algorithms for DDF gradients, `lme-rs` directly projects Wald tests utilizing conservative minimum subset DoF pools, yielding exactly 1.4071 for the `cask` F-statistic.*

### 3. Python Output (`statsmodels.formula.api`)

```python
model = smf.mixedlm("strength ~ C(cask)", data, groups=data["batch"])
result = model.fit()
wald_test = result.wald_test("C(cask)[T.b] = 0, C(cask)[T.c] = 0")
```

```text
Mixed Linear Model Regression Results
===========================================================
Model:             MixedLM  Dependent Variable:  strength  
No. Observations:  60       Method:              REML      
No. Groups:        10       Scale:               7.3051    
Min. group size:   6        Log-Likelihood:      -148.0139 
Max. group size:   6        Converged:           Yes       
Mean group size:   6.0                                     
-----------------------------------------------------------
             Coef.  Std.Err.    z    P>|z|  [0.025  0.975]
-----------------------------------------------------------
Intercept    59.295    0.838  70.778 0.000  57.653  60.937
C(cask)[T.b]  0.850    0.855   0.995 0.320  -0.825   2.525
C(cask)[T.c]  1.425    0.855   1.667 0.096  -0.250   3.100
batch Var     3.367    0.992                              
===========================================================

<Wald test (chi2): statistic=[[2.81429994]], p-value=0.24483984, df_denom=2>
```

### 4. Julia Output (`MixedModels.jl`)

```julia
fit(MixedModel, @formula(strength ~ cask + (1 | batch)), data, REML=true)
```

```text
Linear mixed model fit by REML
 Formula: strength ~ 1 + cask + (1 | batch)
   REML criterion at convergence: 296.0278

Variance components:
            Column   Variance Std.Dev. 
batch    (Intercept)  3.36709  1.83496
Residual              7.30513  2.70280
 Number of obs: 60; levels of grouping factors: 10

  Fixed-effects parameters:
─────────────────────────────────────────────────
               Coef.  Std. Error      z  Pr(>|z|)
─────────────────────────────────────────────────
(Intercept)  59.295     0.837767  70.78    <1e-99
cask: b       0.85      0.854699   0.99    0.3199
cask: c       1.425     0.854699   1.67    0.0955
─────────────────────────────────────────────────
```
