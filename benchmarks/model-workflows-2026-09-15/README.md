# Mixed-model workflow comparison — 15 September 2026

## Decision

**Promising performance; do not switch a production backend yet.** The tighter-tolerance profile passes all recorded checks on five of six synthetic designs. The default profile passes the full gate on only the large fit-scaling case. Both complete benchmark runs exit 2 because at least one compatibility gate fails.

These are model-level comparisons with statsmodels/Patsy and Pingouin Holm correction. They do not measure a complete application, replace its data preparation, or establish that a general-purpose statistics dependency can be removed.

## Numerical agreement

| Design | Rows | Default controls | Tolerance `1e-10` |
|:---|---:|:---|:---|
| `balanced_none` | 48 | Fail | Pass |
| `balanced_within` | 96 | Fail | Pass |
| `incomplete_decomposed` | 157 | Fail | Pass |
| `incomplete_baseline` | 91 | Fail | Pass |
| `large_within` | 1226 | Pass | Pass |
| `near_boundary` | 96 | Fail | Fail |

Passing means agreement on the recorded model and inference quantities, independent-likelihood checks on every measured fit/refit, and bootstrap worker-count checks where applicable. The large case excludes bootstrap and subject-deletion workloads. It does not prove correctness for other designs.

**Default precision:** several ordinary fits or refits miss the predeclared variance/likelihood tolerances even when their estimates are close. This is a failed equivalence check, not by itself proof that a statistical method is invalid. The JSON records each failed quantity and its error. Tolerances were not loosened to obtain a pass.

**Boundary blocker:** with tolerance `1e-10`, the random-intercept search can stop at a poorer likelihood on a near-zero-variance null-response refit while reporting convergence. Across the 19 shared responses, the largest recorded deviance discrepancy from the independent reference is approximately **0.0491**, with a variance-component discrepancy of approximately **0.00468**. The statsmodels reference also has nonconverged refits in this case. The suite rejects both; a successful convergence flag alone is insufficient.

The tighter tolerance selects a different optimization path from the specialized default search. It improves agreement in the ordinary cases, but is not a general remedy for boundary behavior. No library optimizer or production backend was changed in this work.

## Qualified timing results

Windows release extension; one computational thread for the main metrics. Three independent process pairs with alternating order; one warmup and five measured repetitions per process. The table uses the tighter-tolerance profile and excludes the rejected boundary case.

| Design | statsmodels REML (ms) | Rust REML (ms) | Paired fit ratio [95% block interval] | Subject-deletion ratio |
|:---|---:|---:|:---|---:|
| `balanced_none` | 10.11 | 1.02 | 10.0× [9.4, 10.4] | 9.2× |
| `balanced_within` | 17.78 | 1.23 | 14.5× [14.4, 15.6] | 14.0× |
| `incomplete_decomposed` | 24.31 | 1.53 | 17.3× [15.9, 18.2] | 16.7× |
| `incomplete_baseline` | 17.75 | 1.26 | 14.6× [11.9, 16.3] | 14.5× |
| `large_within` | 134.45 | 2.82 | 50.4× [47.6, 54.1] | Not measured |

Ratios above one favor Rust. Ratios are paired by execution block; dividing the two displayed overall medians need not reproduce the reported ratio. Intervals resample only three blocks and have limited resolution. These are workload-specific measurements on one machine.

### Repeated null-response fits

Each workload fits both the full and additive null model for **19 identical simulated response vectors**. Responses are generated once and shared between engines.

| Design | Formula-refit ratio | Prepared-design ratio | statsmodels prepared batch (ms) | Rust prepared batch (ms) |
|:---|---:|---:|---:|---:|
| `balanced_none` | 7.6× | 134.7× | 244.33 | 1.81 |
| `balanced_within` | 12.4× | 175.8× | 458.43 | 2.61 |
| `incomplete_decomposed` | 13.9× | 124.8× | 611.70 | 4.90 |
| `incomplete_baseline` | 13.0× | 175.3× | 468.02 | 2.67 |

**The prepared-design ratio excludes preparation and simulation.** Both engines reuse fixed-effect designs; statsmodels constructs each response model, while Rust uses the prepared-fit API. This is a reuse benefit, not an end-to-end bootstrap or UI speedup. The formula-refit column includes design construction and frame copies and is the more conservative comparison for existing formula-refit loops. Native parallel-bootstrap samples are retained as diagnostics, not claimed as repeated speed evidence.

Adjusted simple-comparison ratios range from about 1.2× to 2.6× on passing designs. They use matched Holm correction, and refer to conditional within-visit comparisons. Separate regression tests check averaged factor comparisons, intervals, correction families, and standardized effects; do not exchange those hypotheses during integration.

## What this supports

The smallest complete REML fit already takes about 10 ms in the reference
backend and about 1 ms in Rust. That absolute saving alone may have little effect
on a report dominated by data preparation and plotting. Repeated refits are the
more useful performance case; complete report latency still needs measurement.

- A credible performance case for repeated model fitting, especially reusable designs and subject-deletion diagnostics.
- Delegating factor coding, marginal-mean grids, contrasts, and null-model refits to a reusable library can reduce custom modelling code.
- Retaining the existing analysis-frame preparation, covariate interpretation, reporting, and unrelated statistical procedures.

## Requirements before adoption

1. Resolve and regression-test the default precision and boundary-optimization failures without weakening the comparison gate.
2. Preserve model formulas, ML/REML choices, reference grids, contrast families, and the existing inference convention. Any switch to Satterthwaite or Kenward–Roger inference must be an explicit scientific change.
3. Run the actual consumer workflow on representative data, including aggregation, missing rows, warnings, report fields, progress/cancellation, and packaging. These synthetic backend tests do not cover that integration.
4. Use appropriately sized bootstrap runs and calibration simulations before making scientific error-rate claims. Nineteen replicates here test computation, not inferential precision.

## Reproduction and evidence

Validation of the harness and repository: **101 Python tests passed**, including
13 new numerical/failure-gate cases; Ruff and formatting checks passed;
documentation checks and six Rust doctests passed. Benchmark compatibility is
reported separately above and remains failed overall for both profiles.

- Engine source revision: `b917460045c77e06b79b7ebd7bad0aa5bd4095dc`. The working tree contained the new benchmark/tests/docs, with no fitting-engine changes.
- Python: `3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)]`.
- Package versions: `lme_python 0.2.6.dev0`, `numpy 2.5.1`, `pandas 3.0.3`, `scipy 1.18.1`, `statsmodels 0.14.6`, `patsy 1.0.2`, `pingouin 0.6.1`, `polars 1.42.1`, `pyarrow 24.0.0`.
- Extension SHA-256: `649bed8ade3a88773c531cd91c3f0f24b217875c47f2f5ba46810e50272b8b08`.
- Harness SHA-256: `06bd09ed604dfab257309bf9cb55c815790d070e68fb195d08693bbce4f1ca22`.
- [Method, timing boundaries, and commands](../../docs/MODEL_WORKFLOW_BENCHMARK.md).
- [Readable summary and tolerances](summary.json).
- [Complete default-profile report, gzip JSON](default.json.gz).
- [Complete precision-profile report, gzip JSON](precision.json.gz).

The archives retain every warmup/timing record, warnings, errors, per-fit reference checks, simulation/input hashes, package/thread metadata, and model/inference outputs. Default-profile and failed-case timings remain available but are unqualified. Earlier exploratory and interrupted runs are not used in these tables.

The measured runtime is not a claim of compatibility with every Python or dependency patch release. Validate the intended deployed environment separately.
