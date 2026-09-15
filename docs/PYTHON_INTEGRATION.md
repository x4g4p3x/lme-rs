# Integrating lme-python into Python applications

This guide covers packaging, data conversion, and statistical contracts when
adding lme-python to an existing analysis application.

## Python and distribution

A regular `cp314` wheel targets the CPython 3.14 ABI across patch releases;
applications do not need the exact patch version used to build it.
The release workflow now builds CPython 3.10–3.14 wheels for Windows x64,
Linux x86_64/aarch64, and macOS x86_64/aarch64. These changes take effect in the
next published release; 0.2.5 only has CPython 3.10 wheels.

For a local source build, select the application's interpreter explicitly:

```powershell
uv run --no-project --with maturin maturin build --release --locked --manifest-path python/Cargo.toml --interpreter C:/path/to/app/.venv/Scripts/python.exe --out python/dist
```

Install the resulting wheel with the `pandas` extra in an isolated test environment
before adopting a released version in the application. The extra includes pandas and PyArrow;
the latter is needed for nullable/categorical pandas columns. Polars remains a
runtime dependency. The wheel must also be exercised in the frozen/PyInstaller
application before a backend rollout.

## Data and statistical contracts

Preserve the analysis-cell aggregation, missing-cell handling, subject identity,
covariate decomposition, and declared factor order. The library does not choose
those scientific conventions for the caller. Use safe column identifiers.

- Explicit formulas project relevant columns **before** pandas/Arrow conversion,
  so unused application metadata does not need to be Arrow-compatible.
- Nullable numeric columns are supported when the model values are complete.
  Missing/non-finite model values must be handled deliberately by the caller.
- Categorical/Enum labels are sent as strings over IPC, avoiding a Rust dictionary
  decoding panic. Automatic factor coding still sorts labels and uses treatment
  coding; pandas' declared category order does not change that behavior.

### Preserve sum coding explicitly

The formula parser does not implement Patsy's `C(factor, Sum)` syntax. Use the
[portable example](../python/examples/repeated_measures.py) to encode declared levels
into numeric sum columns before fitting. For `[Day 1, Day 2, Day 10]` the rows are
`[1, 0]`, `[0, 1]`, and `[-1, -1]`. Reuse that mapping for prediction grids, even
when a grid contains only a subset of levels. Numeric `:` interactions then retain
the original design. Alternatively, keep Patsy for design construction
and assign safe numeric column names when passing the encoded columns to lme-rs.

Preserve the chosen covariate adjustment: none, within-subject deviations,
separate between- and within-subject components, or a baseline value. For a
prediction grid evaluated at the original centering point, hold centered
covariates at **zero**. Recomputing a mean after dropping baseline rows can change
the requested reference point.

Use `fit.design_matrix(grid)` for fixed-effect rows in `fit.fixed_names` order.
It reuses training encodings, excludes offsets and random effects, and rejects
unknown/missing categorical levels. A between-group contrast is the mean of one
group's rows minus the mean of the other's rows. This supports ordered cell
means and equal-weight factor averages without reconstructing lme-rs internals.
For pre-encoded multi-column factors, group their contrast rows explicitly with
`fit.test_contrast(L)`; automatic ANOVA sees each numeric column as a separate term.

### Preserve the null bootstrap

An unconditional null bootstrap simulates fresh subject random effects under the
null model, then fits full and reduced models by ML on identical responses. `fit.boot()`
conditions on existing fitted random effects and is not a replacement for that
test. The example shows the appropriate integration:

1. Prepare each full/reduced design once.
2. Fit the observed models with `reml=False`.
3. Draw new subject effects and residual noise from the null variance components.
4. Call each prepared object's `fit(y=response, reml=False)`.
5. Count only converged finite pairs and report the valid replicate count and seed.

The example's 19 replicates make it quick to execute; production inference needs
a replicate count chosen for the required Monte Carlo precision. It preserves the
procedure, not NumPy's exact random-number stream. Applications can retain their existing RNG
and simulation function unchanged when using the prepared fitting backend.

## Result mapping and validation

| Existing analysis operation | lme-python interface |
|---|---|
| Fixed coefficients | `coefficients`, `fixed_names` |
| Fixed coefficient standard errors | `std_errors` |
| Population versus conditional predictions | `predict` / `predict_conditional` |
| Conditional fitted values and residuals | `fitted`, `residuals` |
| Subject variance and residual variance | `var_corr` rows and `sigma2` |
| Random-effect estimates | `ranef` rows |
| Design reconstruction for contrasts | `design_matrix` |
| ML log likelihood | `log_likelihood` |
| Convergence | `converged`, `diagnostics`, `FitControl(require_convergence=True)` |

Small-sample Satterthwaite/Kenward–Roger tests are optional inference changes;
they will not reproduce asymptotic Wald p-values. Keep changes to inference
separate from coefficient/prediction parity checks and label the chosen method.
Likewise, preserve the chosen per-factor Holm families and report calculations (ICC,
R-squared, standardized differences, influence checks).

The Python suite checks pandas conversion, prediction/design consistency,
balanced cell means, declared sum coding, and new-animal null simulation. Run:

```text
python scripts/ci/lme_ci.py python --python-version 3.14 --examples
```

Before changing an application's default backend, compare its covariate modes and
unbalanced/missing-cell cases against existing numeric tests and reference fits,
then benchmark complete analyses. A backend migration alone does not establish
a general speedup or replace unrelated functionality from other dependencies.
