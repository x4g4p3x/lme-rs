# Robust-inference design validation, 10 October 2026

Scanning wider contiguous prediction and training designs by row reduces the
cost of complete robust-inference calls. The candidate collects each column's
maximum absolute value and maximum absolute difference together, replacing
repeated strided column scans. Comparing that maximum difference with the
original `64 * EPSILON * column_scale` threshold preserves the acceptance rule.
Covariance accumulation, weights, residuals, standard errors, statistics, and
p-value calculations are unchanged.

The row path applies to standard-layout designs with at least eight columns.
Smaller and nonstandard layouts retain the original inline column check. The row
helper is deliberately not inlined: an earlier version that moved the fallback
into a helper made small calls slower. Its two temporary vectors require
`16 * p` bytes of element storage, plus allocation overhead, and live only for
the validation call. There is no persistent cache.

## Workloads and attribution

The common harness in `benches/bench_math.rs` generates deterministic OLS data
with seed `261010`. Cases have 180 rows and two coefficients, or 10,000 rows and
16 or 48 coefficients. Formulas include an intercept and independently generated
uniform numeric predictors. Timing covers the complete `compute_robust_se` call
and result destruction, excluding input generation and fitting. Clustered cases
use 100 cyclic string labels. Each measured call must succeed; initial checks
also require finite positive standard errors.

Existing sleepstudy REML LMM controls measure `with_robust_se` with two
coefficients, 180 observations, and either HC0 or 18 Subject clusters. These use
the existing batched timing boundary, which excludes fit cloning. Their absolute
times should not be compared with the direct-call OLS cases as if setup and
destruction boundaries were identical.

Separate temporary timers inside the original routine measured approximately
5.63 ms for reconstruction plus design validation, 1.61 ms for influence
construction, and 3.57 ms for HC0 covariance accumulation in the 48-coefficient
case. These diagnostics identify substantial validation work; they are excluded
from throughput comparisons. A standalone reference-loop probe did not reproduce
the full routine's cost and was rejected as phase attribution.

## Complete-call measurements

Baseline production source is revision
`b1de799fa1da7bd8adea6175ecc552fe83e9fce8`, with the common benchmark additions.
Candidate production changes are confined to `src/robust.rs`. The
[raw report](perf-tune-robust-covariance-2026-10-10.json) retains source and binary
hashes, patches, commands, samples, Criterion estimates and outlier information,
session logs, diagnostics, and rejected experiments.

Eight independent processes ran sequentially in ABBA / BAAB order, with four
processes per version. Each case used two seconds of warmup and ten measured
samples over approximately three seconds. The second block was added to resolve
uncertainty in the small-call control; all first-block samples were retained.
Table times are medians of four process medians of sample-average time per call.

| 10,000-row operation | Baseline | Candidate | Reduction | Candidate/baseline 95% interval |
|---|---:|---:|---:|---:|
| HC0, 16 coefficients | 2.152 ms | 1.644 ms | 23.6% | 0.700–0.778 |
| Clustered, 16 coefficients | 2.205 ms | 1.692 ms | 23.3% | 0.747–0.784 |
| HC0, 48 coefficients | 9.158 ms | 7.434 ms | 18.8% | 0.772–0.851 |
| Clustered, 48 coefficients | 6.718 ms | 4.976 ms | 25.9% | 0.708–0.789 |

| Small control | Baseline | Candidate | Median change | Candidate/baseline 95% interval |
|---|---:|---:|---:|---:|
| OLS HC0, 180 rows, 2 coefficients | 4.758 µs | 4.835 µs | 1.6% slower | 0.973–1.092 |
| OLS clustered, 180 rows, 2 coefficients | 19.405 µs | 18.547 µs | 4.4% faster | 0.924–0.986 |
| Sleepstudy LMM HC0 | 6.800 µs | 7.016 µs | 3.2% slower | 0.957–1.050 |
| Sleepstudy LMM clustered | 18.660 µs | 18.889 µs | 1.2% slower | 1.002–1.041 |

The smaller-call results do not establish a general speedup. One control has a
small slowdown whose interval excludes no change, while two intervals span no
change. The OLS HC0 interval cannot exclude a larger slowdown. The source retains
its original small-design calculation, but code generation and process variation
can still affect latency. No small-call improvement is attributed to the row
algorithm, including the faster clustered OLS control.

Intervals use 2,000 hierarchical bootstrap draws, resampling processes and ten
samples within each selected process, with seed `261010`. Outliers are retained.
Four processes per version on one machine provide limited information about
session variation, and no evidence about other platforms.

Two covariance-triangle attempts were removed: the iterator version was slower,
and the indexed version's large-case gain was inconclusive while a small LMM
control regressed 5.3%. A first row-validation prototype improved wider calls but
regressed small HC0 controls by 6.8% and 14.2%; it was replaced with the inline
fallback and separate row helper. The raw report preserves these measurements.

## Environment and reproduction

Measurements used Windows x86_64, an AMD Ryzen 5 8600G with six cores and twelve
logical processors, Rust/Cargo 1.99.0, the unchanged repository bench profile,
default Argmin features, and static Intel MKL. `MKL_NUM_THREADS`,
`OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, and `POLARS_MAX_THREADS` were one before
each process started. `LME_PERF_DIAG` was unset. Timing runs were serialized, with
no competing builds or tests. Preflight ran between the two measurement blocks.
Gnuplot was unavailable; Criterion used Plotters without plots.

Build the common benchmark harness against the recorded baseline revision and
candidate production patch, retaining each executable:

```sh
cargo bench --locked --bench bench_math --no-run
```

Run the retained executables in the recorded order with distinct labels:

```sh
<executable> --bench 'robust_covariance|inference/with_robust_se' --noplot --warm-up-time 2 --measurement-time 3 --save-baseline robust-<label>
```

The report contains exact labels, the harness and candidate patches, hashes of
the lockfile and sleepstudy input, and the summary-generation code. The retained
executables under the local ignored `benchmark-results/robust-2026-10-10/`
directory provide the local rerun inputs; the checked-in report contains their
identity and raw evidence.

## Correctness and scope

New tests in `tests/test_robust.rs` independently specify HC0 and clustered
covariance for an orthogonal four-coefficient design, including positive and
negative off-diagonal entries. A 16-column Walsh design checks per-column
tolerances with predictor units differing by `1e200`, harmless rounding,
changed-design rejection, and both contiguous and column-major layouts. These
tests passed against the baseline before changing the relevant paths. Existing
coverage checks R fixtures, extreme unit rescaling, singleton clusters, precision
scaling, explicit whitening, missing clusters, reordered rows, and stored bases.

Validation passed `task preflight` and `task test:consolidated`: 132 unit tests,
380 integration tests, and six doctests passed. Four existing heavy production
tests were ignored. Example compilation also passed. Full Python binding tests,
the optional Basin feature suite, and other-platform validation were not run.

These results measure robust post-fit inference in the listed cases. They do not
measure faster fitting, Python overhead, cross-library parity, or performance
for strided designs, and do not change completion commitments.
