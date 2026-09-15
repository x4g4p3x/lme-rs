# Model-based analysis in Rust and Python

These development APIs extend the linear and mixed-model engine. They do not
reproduce an entire general-purpose statistical package. They become available
in the next published release after 0.2.5.

Run [the self-contained Python example](../python/examples/model_based_analysis.py)
for an incomplete-subject mixed-model analysis.

## Ordered factors and contrasts

Pass explicit specifications to Python `lm`, `lmer`, or `prepare_lmer`:

```python
import lme_python as lme

factors = {
    "treatment": lme.FactorSpec(["control", "low", "high"], coding="sum"),
    "visit": lme.FactorSpec(["baseline", "week2", "week10"], coding="sum"),
}
fit = lme.lm("response ~ treatment*visit + baseline_value", data, factors=factors)
```

`FactorSpec` declares both order and coding. Treatment coding omits the first
level; sum coding assigns the last level minus one in each contrast column.
Without an intercept, the first factor uses a full indicator basis. Declared
levels must be unique and observed in the training data; null or unknown levels
raise errors. A prediction grid may contain a subset of the training levels.
Numeric columns may be explicitly declared as factors using their string labels.
These specifications concern fixed factors, not categorical random slopes.

Encodings are saved on the fit and reused for prediction, marginal means,
Satterthwaite/Kenward–Roger inference, and bootstrap/profile refits. Multi-column
main effects and interactions remain grouped as model terms. Unspecified factors
retain the existing sorted treatment coding. `C(factor, Sum)` formula syntax is
not required or implemented. Interaction parsing retains the existing two-factor
scope; this change does not add arbitrary higher-order formula expressions.

Rust uses `FactorSpec { levels, coding: FactorCoding::Sum }` in a `HashMap` with
`lm_df_with_factors` or `prepare_lmer_with_factors`. The latter accepts optional
observation precision weights; call `.fit(None, reml, &control)` on the result.

## Classical OLS ANOVA and uncertainty

```python
table = fit.anova(ddf_method="residual", anova_type="III")
intervals = fit.confint(level=0.95)
```

`residual` uses exact residual degrees of freedom, `n - rank(X)`, for a full-rank
Gaussian OLS model with classical covariance. It rejects mixed/generalized models,
robust covariance, and saturated fits. Rank-deficient designs are rejected by the
OLS solver rather than silently assigning arbitrary coefficients.

ANOVA reports term F tests, numerator/denominator degrees of freedom, p-values,
`sum_sq`, `mean_sq`, `residual_sum_sq`, `residual_df`, and `partial_eta_sq`.
Partial eta squared is `SS_effect / (SS_effect + SS_error)`; it is not generalized
eta squared. Mixed-model tables leave the OLS-only fields absent. Coefficient
confidence intervals for classical OLS use Student's t, including matrix OLS.
The same residual method is available for explicit contrasts and comparisons.

Choose the hypothesis deliberately:

- Type I tests terms sequentially in the stored formula order.
- Type II tests each term after other terms except its containing interactions.
- Type III tests each term conditional on all other terms. With interactions,
  use sum coding for the usual equal-weight factorial interpretation; numeric
  covariate zero points also affect the hypothesis.

Rust uses `fit.anova_typed(AnovaType::Type3, DdfMethod::Residual)`.
The existing default degrees-of-freedom arguments remain unchanged; request
`residual` explicitly for classical OLS tests and marginal means.

## Reference grids and adjusted comparisons

```python
means = fit.emmeans_grid(
    ["treatment"], data,
    by=["visit"], at={"baseline_value": 12.0},
    weights="equal", ddf_method="residual",
)
pairs = fit.emmeans_grid_pairs(
    ["treatment"], data,
    by=["visit"], at={"baseline_value": 12.0},
    adjust="holm", ddf_method="residual",
)
```

Both operations derive estimates and uncertainty from the fitted model. In an
ANCOVA, comparisons therefore retain its covariate adjustment. `terms` can contain
several factors to request their interaction cells. `by` holds conditioning
factors fixed; comparisons and multiplicity corrections form separate families
within each `by` cell. To compare all interaction cells in one family, put all
factors in `terms` and leave `by` empty.

`at` supplies one finite reference value per numeric covariate. Other numeric
covariates use the supplied data's means. Nuisance factors are averaged using:

- `equal`: equal weight for every nuisance-factor combination.
- `proportional`: pooled observed joint frequencies of nuisance-factor
  combinations, shared across every target and conditioning cell. This is not
  target-specific cell weighting or weighting by observation precision.

The result includes named `cells` and the contrast/design rows in `linfct`.
Pairwise results include `groups` identifying each adjustment family. Supported
adjustments are none, Holm, Bonferroni, and Tukey–Kramer. No simultaneous confidence
interval claim is made for the pointwise marginal-mean intervals. Use the original
analysis data when its covariate means or nuisance frequencies define the target.

Rust uses `ReferenceGrid { terms, by, at, weights }` with
`fit.emmeans_with_grid` and `fit.emmeans_pairs_with_grid`.

Automatic reference grids currently cover linear models and LMMs, up to 4,096
factor combinations. Offset models require an explicitly constructed prediction
design instead. GLMM response-scale averaging is outside this API's scope.

## Unconditional null bootstrap

```python
full = lme.prepare_lmer("response ~ treatment*visit + (1|subject)", data, factors=factors)
null = lme.prepare_lmer("response ~ treatment + visit + (1|subject)", data, factors=factors)
result = lme.bootstrap_lrt(full, null, 999, seed=127, n_jobs=2)
```

This tests the additional fixed-effect dimensions in `full`. Both observed models
and every replicate use ML. The models must share the same ordered observations,
response, offsets, precision weights, and random-effect structure. The null's
fixed-effect column space must be nested in the full model's space. This API does
not test changes to the random-effect structure.

Each replicate draws fresh Gaussian random effects and residual noise under the
fitted null. Correlated random slopes, multiple grouping blocks, offsets, and
precision weights follow the prepared design. Immutable design data is shared;
workers reuse independent response workspaces. Seeds and output order are stable
across worker counts, but do not reproduce R's random-number stream.

`statistics` and `errors` retain an entry for every requested replicate. Only
converged finite pairs with a valid nested likelihood ordering count as `valid`.
The reported p-value is `(1 + exceedances) / (1 + valid)`; `mc_se` is its plug-in
Monte Carlo standard error. Both are absent if all replicates fail. Inspect
failure counts and reasons before interpreting a result: excluding failures can
bias inference, and a small valid sample does not establish a reliable p-value.
Choose the replicate count for the precision needed; 999 is illustrative.

Rust exposes `bootstrap_lrt(&full, &null, nsim, seed, n_jobs, &control)`.
The existing `fit.boot()` retains its conditional simulation contract.

## Validation and missing observations

[Reference generation](../scripts/validation/generate_modeling_reference.R) uses
project-owned deterministic synthetic observations and black-box calls to R
`stats`, `car`, `emmeans`, `lmerTest`, and `lme4`. Package versions are recorded in
[the numerical fixture](../tests/data/modeling_reference.json). The Rust and
Python tests cover unbalanced factorial ANCOVA, chosen covariate reference values,
ordered sum coding, separate comparison families, and incomplete-subject LMMs.
R-generated unconditional null responses also exercise matching ML refits.
A simulation-moment test checks fresh group covariance, offsets, and weights.

Prepare analysis rows deliberately before fitting. The library rejects missing
model values; it does not impute them or automatically discard complete subjects.
Retaining observed rows from incomplete subjects is an explicit analysis choice
whose validity depends on the model and missingness assumptions. An LMM is not a
numerically identical substitute for sphericity-corrected repeated-measures ANOVA,
and a random intercept is not a general temporal residual covariance model.

The implementation uses the documented linear-model quadratic-form tests,
[reference-grid conventions](https://rvlenth.github.io/emmeans/reference/emmeans.html),
and [unconditional parametric simulation contract](https://lme4.github.io/lme4/reference/bootMer.html).
No external implementation source is copied or translated. See
[the provenance policy](../PROVENANCE.md).
