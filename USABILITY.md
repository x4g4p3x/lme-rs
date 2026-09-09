# Supported workflows and adoption

[Documentation](docs/README.md) · [Rust guide](GUIDE.md) · [Python guide](python/PYTHON_GUIDE.md)

Use this page to decide whether the implemented workflow fits your analysis.
The [completion report](REPO_COMPLETION_BY_AREA.md) measures locked project
commitments; it is not a measure of statistical suitability.

The assessment below is grounded in the repository's tests and documented
scope. A documentation review does not establish new production experience
or refresh old benchmark measurements.

## Choose a workflow

### Covered workflows

These paths have repository tests or examples. Match your model shape to the
evidence before relying on the result.

| Workflow | Rust and Python support | Main scope |
|:---------|:------------------------|:-----------|
| OLS | Formula and matrix interfaces | Caller supplies intercept for matrix OLS |
| Gaussian LMM | `lmer`, weights, offsets, REML/ML | Random intercepts/slopes, nested and crossed structures |
| GLMM | Binomial, Poisson, Gaussian, Gamma | Family/link domains and quadrature limits apply |
| Prediction | Population and conditional methods | GLMM link scale versus response scale must be chosen explicitly |
| LMM inference | Satterthwaite/Kenward–Roger, ANOVA I–III, contrasts | Covered model shapes; inspect degrees-of-freedom method |
| Multiple comparisons | Tukey/Dunnett `glht`, LMM `emmeans` and pairs | Equal-weight EMM reference grid; no full R package replacement |
| Model comparison | Nested likelihood-ratio tests | Same observations; ML when fixed effects differ |
| Intervals | Wald and profile paths | Profile scope is LMM/GLMM, not NLMM; variance-component scales differ by structure |
| Simulation and bootstrap | LMM/GLMM response draws and refits | Conditional draws; LMM residual bootstrap, GLMM parametric only |
| Grouped cross-validation | `cv_grouped`, `cv_grouped_glmer` | Entire levels of one selected grouping column stay together |

### Workflows needing additional care

| Workflow | Check before use |
|:---------|:-----------------|
| Large crossed or random-slope models | Memory, scaling, convergence, and timings for your own design |
| Less common GLMM links or combined weights/offsets | Which exact combinations have golden fixtures |
| Higher-order quadrature | Grid size grows with random-effect dimension; large structures can retain Laplace optimization |
| Built-in nonlinear means | One grouping factor; mean domain, starts, and identifiability |
| Custom nonlinear means | Supply sensible starts and verify predictions against a reference |
| Sensor calibration | Choose independent curves or a pooled model; [calibration guide](docs/CALO_CALIBRATION.md) |
| Prepared fits | Reuse only the same design; rebuild when rows or predictors change |
| Python pandas/PyArrow inputs | Conversion to Polars, dtypes, and optional dependencies |

### Outside the implemented scope

- Full drop-in compatibility with `lme4`, `lmerTest`, `car`, `nlme),
  `multcomp`, or `emmeans`.
- Arbitrary R expressions, general multivariate `cbind()` responses, or all spline options.
- GLMM response-scale estimated marginal means and compact-letter displays.
- Every R bootstrap option, BCa intervals, or NLMM bootstrap/CV helpers.
- Independent per-sensor fitting inside the pooled `nlmer` API.
- A blanket speed guarantee or established production suitability for untested models.

## Understand the evidence

| Evidence | What it establishes | What it does not establish |
|:---------|:--------------------|:---------------------------|
| [Golden fixtures](tests/data/golden_parity_manifest.json) | Agreement within stated tolerances on named cases | Parity for every formula or dataset |
| [Integration tests](tests/) | Behavior and identities on exercised paths | Independent verification of every statistic |
| [Python tests](python/tests/) and clean-wheel examples | Binding behavior and package installation | A second numerical engine |
| [Cross-language comparisons](comparisons/COMPARISONS.md) | Reference outputs and known differences | Equivalence where a reference uses a simplified model |
| [Benchmarks](BENCHMARKS.md) | Timing on a specific revision, machine, and workload | Universal throughput |
| [Completion manifest](completion_manifest.json) | Fulfilled locked project commitments | Adoption maturity or statistical validity |

Both language interfaces use the same Rust implementation.
Repository validation is distinct from a long record of diverse independent
deployments. The project remains in the 0.2.x series; review release changes
when upgrading.

## Validate an analysis

1. Identify the response distribution, grouping structure, and fixed effects.
2. Run a reference fit with the same observations, formula, link, weights,
   and ML/REML or quadrature settings.
3. Compare coefficients, variance components, predictions, and relevant tests.
   Raw GLMM AIC/BIC can differ because likelihood conventions differ.
4. Inspect convergence, residual behavior, and sensitivity to starts or scaling.
5. Measure the whole workflow if runtime matters, including setup and inference.
6. Record package versions and options with the analysis.

[The guides](docs/README.md#choose-a-workflow) explain the methods;
[troubleshooting](docs/TROUBLESHOOTING.md) helps investigate failures.

## Match the API to the workload

| Call pattern | Starting point |
|:-------------|:---------------|
| One fit | `lmer`, `glmer`, or `nlmer`; inspect diagnostics |
| Same design, repeated LMM fits | `prepare_lmer` and `fit_prepared` |
| Same design, repeated GLMM fits | `prepare_glmer` and `fit_prepared_glmer` |
| Response-resampling inference | `boot_lmer` / `boot_glmer`; inspect replicate convergence |
| Held-out groups | `cv_grouped` / `cv_grouped_glmer`; train each fold independently |
| Many simulations | Seeded parallel draws or batched simulation |

For crossed structures, holding out levels of one grouping factor does not
necessarily hold out every other factor. Choose a split that matches the
intended generalization question.

The July 22 reference contains **10 LMM** and **2 GLMM** timing cases. LMM cold
and prepared timings passed that run's gate; later GLMM fitting changes make
its GLMM rows historical. See [benchmark coverage](BENCHMARK_COVERAGE.md).

## Maintenance

Update scope statements with implementation and test evidence. Keep benchmark
dates tied to the artifacts actually measured. Do not change the
[completion score](REPO_COMPLETION_BY_AREA.md) merely because wording or
navigation improves.
