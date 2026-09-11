# Basin LMM covariance-boundary recovery

11 September 2026 · [Optimizer guide](../docs/BASIN.md) · [Measurement summary](basin-covariance-recovery-2026-09-11.json)

**The change fixes a premature boundary stop, with a measured runtime cost.**
Argmin remains the default. This is a correctness improvement to lme-rs's Basin
adapter, not evidence that Basin is ready to become the default optimizer.

## The fixed case

An 80,000-row, 3,000-group random-slope fit, using production-fixture seed 106,
reported convergence at deviance **226271.4878426**. R/lme4 and alternative Rust
starts found **226267.8237454**, lower by 3.66410. The new default Basin fit
returns **226267.8237450**. The formula is `y ~ x + (x | group)`, using ML.

The collapsed Cholesky column had a zero diagonal and positive off-diagonal.
At that boundary, reversing the off-diagonal column preserves covariance, but
an inward move from its other orientation exposes an improving direction.
Another diagonal was approximately `1e-16`; the old exact-boundary check missed it.

Recovery now uses the actual random-effect block layout, recognizes numerically
near-zero faces, and tests both column orientations. A successful probe starts
a smaller simplex, with its cost reused once. Restarts require improvement beyond
the requested cost tolerance and numerical roundoff, and share the caller's
iteration budget. Generic GLMM/NLMM recovery retains its existing search rules.

The regression covers ML, REML, observation weights, and budget exhaustion.
Unit coverage also checks three-row columns, offsets from preceding covariance
blocks, true boundary minima, and one-use cost reuse. The independent reference
is [retained with its generation recipe](../tests/data/basin_boundary_lme4.json);
the Rust generator is in [the regression test](../tests/test_basin_boundary_recovery.rs).

## Before/after measurements

Both builds use locked **Basin 1.10.0 from crates.io**, the same dependency graph,
Rust 1.98.1, Windows, and one BLAS/Rayon/Polars thread. Before is lme-rs
`251e5b8c46552c290645c757fca34606d68bea3c`; the summary records after-source and executable hashes.
No dependency upgrade is part of this change.

Prebuilt consumers ran ABBA / BAAB / ABBA, six independent processes per variant,
three warmups and eleven measurements per case/metric/process. All **4,224
measured fit checks** report convergence. Each timer excludes data loading and
validation. Complete fits include preparation; prepared fits reuse the design.
Tracing and phase profiling were separate from timing, and no build ran concurrently.

The table reports after/before ratios: below 1 is faster. Ranges are the observed
three paired block ratios, **not confidence intervals**. Three blocks on one
workstation do not establish the default-switch 5% noninferiority gate. Ratios
are paired-block medians, so displayed process-median times in the JSON need
not divide to exactly the same ratio.

| Case | Complete fit ratio [range] | Prepared fit ratio [range] | Numerical agreement |
|:---|:---|:---|:---|
| `sleepstudy_reml` | 0.975 [0.797, 1.025] | 1.056 [1.048, 1.077] | yes |
| `sleepstudy_weighted_reml` | 1.029 [1.015, 1.040] | 1.032 [1.027, 1.046] | yes |
| `penicillin_crossed_reml` | 0.987 [0.971, 0.989] | 0.998 [0.997, 1.002] | yes |
| `pastes_nested_reml` | 1.050 [0.973, 1.075] | 0.985 [0.955, 1.013] | yes |
| `random_intercept_10k` | 1.022 [1.020, 1.051] | 0.972 [0.952, 1.215] | yes |
| `random_intercept_50k` | 0.958 [0.886, 0.983] | 0.975 [0.871, 1.081] | yes |
| `random_intercept_100k` | 1.011 [0.998, 1.070] | 1.111 [0.958, 1.274] | yes |
| `large_random_slopes_100k` | 0.988 [0.978, 1.025] | 1.001 [0.989, 1.045] | yes |
| `crossed_20k` | 1.012 [0.995, 1.030] | 0.999 [0.998, 1.003] | yes |
| `nested_10k` | 0.999 [0.984, 1.008] | 0.994 [0.993, 1.006] | yes |
| `cbpp_binomial_ml` | 0.997 [0.993, 0.999] | — | yes |
| `grouseticks_poisson_ml` | 0.982 [0.969, 1.000] | — | yes |
| `boundary_42` | 1.160 [1.145, 1.178] | 1.244 [1.236, 1.289] | yes |
| `boundary_105` | 1.166 [1.109, 1.174] | 1.238 [1.207, 1.267] | yes |
| `boundary_106` | 1.643 [1.613, 1.708] | 1.888 [1.866, 1.905] | improved objective; different fit |
| `boundary_107` | 1.130 [1.111, 1.181] | 1.157 [1.140, 1.189] | yes |
| `boundary_108` | 0.812 [0.777, 0.845] | 0.794 [0.780, 0.821] | yes |

All twelve standard workloads preserve their numerical results
(bit-identical).
Unchanged scalar searches still show timing variation, and the Sleepstudy prepared
fit is about 5.6% slower in this run. These results do not establish universal
performance noninferiority even outside the newly recovered cases.

### Boundary evaluation cost

| Production-fixture seed | Before evaluations | After evaluations | Interpretation |
|:---|---:|---:|:---|
| 42 | 111 | 135 | Extra checks at a true near-zero face |
| 105 | 127 | 151 | Extra checks; same fit |
| 106 | 144 | 265 | Recovers the R-verified better solution |
| 107 | 150 | 174 | Extra checks; same fit |
| 108 | 224 | 170 | Smaller recovery simplex saves work |

Counts come from the separate phase harness and can include diagnostic work;
they are not optimizer iteration counts. Complete fits for seeds 42/105/107
are roughly **13–17% slower**, while seed 108 is about **19% faster**. Seed 106
takes about **64% longer** than the incorrect earlier fit; those timings compare
different fit quality. Reducing unsuccessful boundary-probe work remains open.

## Validation and limitations

- The new seed-106 regression fails before the change and passes afterward.
- Default Rust validation: 131 unit tests, 307 integration tests, and six doctests
  pass; four pre-existing heavy integration tests remain ignored.
- Basin validation: 146 unit tests, 310 integration tests, and six doctests pass;
  the same four heavy tests remain ignored.
- Both release-golden tests pass with each backend. Required Rust/Python lint,
  all-target compilation, rustdoc, and documentation checks pass.
- Extra strict all-target Clippy reports two pre-existing findings in
  `src/mcp.rs` and `tests/test_bug_hunt_edges.rs`. The new integration target passes
  strict Clippy. The extended check passes when only those two existing lint
  categories are allowed on the command line; no lint suppression was committed.
- Weighted-reference comparisons explicitly remove the existing whitened-data
  likelihood constant `sum(log(weights))` before comparing to R. This does not
  change the optimum or loosen tolerance. Absolute weighted-likelihood
  normalization is a separate existing issue, not fixed by this optimizer change.
- No hosted OS matrix, publishing, or default-optimizer change was performed.

An initial timing attempt was rejected after a sanity check showed that Cargo
had reused the old executable for both variants. The retained valid run rebuilt
both variants in one build directory with refreshed source timestamps, verified
distinct executable hashes, and checked each binary's seed-106 objective before
timing. Rejected measurements are excluded from the table and summary.

## Reproduction and retained evidence

Use the existing `bench_fair_rust_julia` example with the shared CSV, formula,
backend feature, ML/REML flag, and `--warmups 3 --repeats 11 --with-phases`.
Alternate separately built before/after binaries in the order above. Validate
their actual numerical behavior before timing; a source hash alone cannot detect
a stale executable. Generate boundary fixtures with the regression-test recipe,
substituting seeds 42, 105, 106, 107, and 108.

Raw process reports, source snapshots, verified executables, fixture hashes,
generation/validation scripts, and full logs are retained locally under
`artifacts/research/basin-covariance-recovery/`. That ignored directory survives
`cargo clean` but is not a remote backup. The checked-in summary preserves
the numerical results, paired ratios, provenance, and raw-report hash.
