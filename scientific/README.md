# Frozen numerical baseline

This suite checks the published **0.2.7** engine against eight declared contracts.
It provides deterministic numerical verification and selected worked-example
agreement. Statistical calibration, independent scientific review, GLMM and NLMM
validation remain separate, unfinished work. See the
[programme](../docs/SCIENTIFIC_VALIDATION_PLAN.md) and
[AI-reviewed pilot protocol](../docs/LMM_CALIBRATION_PROTOCOL.md).
The [initial results](RESULTS_0.2.7.md) report the actual published-wheel run.

The [completed 0.2.7 exploratory pilot](CALIBRATION_PILOT_0.2.7.md) reports
8,000 paired datasets, and the [completed 0.2.7 confirmatory study](CALIBRATION_CONFIRMATORY_0.2.7.md)
reports 1,600,000 candidate and known-variance GLS datasets (20,000 replicates
across all 80 designs) plus 80,000 paired R reference fits under
[protocol v3](../docs/LMM_CALIBRATION_PROTOCOL.md) and the
[confirmatory harness review](../docs/LMM_CALIBRATION_CONFIRMATORY_REVIEW.md).
`task scientific:calibration` runs the 100-replicate exploratory pilot
(`--replicates 1` runs a smoke/cost assessment), and
`task scientific:calibration:confirmatory` runs the frozen 20,000-replicate
confirmatory study across bounded workers under
`benchmark-results/scientific/calibration-confirmatory/0.2.7/`. See the
[AI critique](../docs/LMM_CALIBRATION_AI_REVIEW.md) for resolved objections;
AI review does not constitute human peer review.
For an already archived original pilot, `python scripts/lmm_calibration.py
reanalyse RUN_DIRECTORY` performs only the documented diagnostic correction,
checks the original evidence hashes, and writes a new report without refitting.
The scientific R 4.6.1 reference environment is locked in [renv.lock](renv.lock)
and verified or restored with
`Rscript --vanilla comparisons/scientific/restore_r_environment.R --verify`.

## Contracts and independent evidence

| Case | Evidence | Quantities |
|:---|:---|:---|
| Rational OLS | Exact integer/Fraction normal equations | Coefficients, residual variance, SE and a covariance contrast |
| Longley | [NIST certified regression](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat) | Seven coefficients/SE and RSS |
| Wampler1 | [NIST exact polynomial](https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Wampler1.dat) | Six coefficients and zero residual variance |
| Balanced LMM, ML and REML | Closed-form variance optima, exact dense GLS and 70-digit Decimal likelihood | Fixed effects, both variances, SE, criterion and covariance contrast; REML scalar Satterthwaite df |
| Sleepstudy | [Bates et al. (2015)](https://doi.org/10.18637/jss.v067.i01), author script line 19 | REML fit, variances, SE, criterion and conditional fitted values |
| TVbo | [Kuznetsova et al. (2017)](https://doi.org/10.18637/jss.v082.i13), lines 90–106 | Fit, Type III Satterthwaite and Type II KR tables |
| Beets | [Halekoh and Højsgaard (2014)](https://doi.org/10.18637/jss.v059.i09), lines 26–30 and 79 | REML fit and scalar harvest KR comparison |

The oracle in [scientific_validation.py](../scripts/scientific_validation.py)
uses standard-library rational arithmetic, Gauss–Jordan inversion and Decimal;
it does not call the candidate's matrix routines or fit a reference model.
The paper adapter runs selected statements from hash-verified author scripts
using modern R packages. This is not historical-binary reproduction or a
replication of every analysis or conclusion in those papers. Bootstrap examples
and the original simulation campaigns are excluded.

[cases.json](cases.json) freezes source hashes, case IDs, package/namespace
versions, fit controls and tolerances. OLS is tight (1e-10); Longley requires
eight-digit relative agreement; optimized balanced variances allow two parts
per million. Initial paper gates allow 1e-3 relative plus 3e-4 absolute error,
with tighter criterion and probability gates. These are explicit initial
engineering contracts, not universally accepted scientific thresholds. Numerator
degrees of freedom require exact equality. The balanced Satterthwaite df check
has a separately declared absolute gate of 0.05 for its numerical derivatives.
R convergence messages fail the corresponding comparison. Contract changes
need a visible registry revision. TVbo uses a declared tighter optimization
control; the other paper cases retain the default controls.

## Reproduce

From the repository root, using the pinned tools:

Use a checkout containing this validation harness: its frozen initial commit is
`0343652f6b11e1fd6244864cf7afe5728b2bb0c2`. The engine's earlier `v0.2.7` tag
predates the harness. A newer development checkout still installs the exact
published 0.2.7 engine for this baseline.

```bash
task scientific:oracle
task scientific:baseline
```

Alternatively use `python scripts/ci/lme_ci.py scientific-baseline` (or `python3`).
Pass `--rscript PATH` and optionally a new `--output-dir PATH` through the runner
or `task scientific:baseline -- ...`. Existing output directories are refused.
The full baseline needs internet access, R 4.6.1 and the exact R package and
loaded-namespace versions recorded in the registry, including lme4 2.0.1,
lmerTest 3.2.1, pbkrtest 0.5.5 and jsonlite 2.0.0. It fails if versions drift.
It deliberately runs R with `--vanilla`; the older general comparison renv
environment is not its reference environment. An observed version manifest is
not a complete portable R dependency lock; exporting a restorable R environment
is a remaining reproducibility task.

The runner downloads source files and checks SHA-256 before evaluating author
code. It exports the three datasets into an ignored, unique run directory,
creates a fresh Python environment from [pyproject.toml](pyproject.toml) and
the frozen uv lock, and installs the registry's exact published engine. The
native module must reside inside that environment. No local editable engine or
sibling path override is accepted. Small analytic cases also run in the usual
Python editable and isolated-wheel CI tests.

Runs are retained under `benchmark-results/scientific/0.2.7/`. Preserve the
whole run directory: downloaded sources, data exports, R output/session,
environment and `results.json`. Reports include per-metric values and limits,
all case failures, binary/source/adapter/registry/lock hashes, package versions
and the release Git SHA. Nonfinite results and empty comparison sets fail.
The number of scalar comparisons is descriptive; many fitted values are
correlated and it is not a count of independent scientific experiments.

## Provenance and reuse

NIST data and publicly downloadable JSS author scripts are retrieved from their
original URLs. The project-owned adapter does not translate reference-library
implementation into lme-rs. R package datasets and downloaded author code stay
outside Git and the distributed crate/wheel; their upstream licenses still
govern reuse and redistribution. Do not publish run-directory exports without
checking those terms. The public report can state model specifications, scoped
aggregate outcomes and hashes without redistributing source datasets.
