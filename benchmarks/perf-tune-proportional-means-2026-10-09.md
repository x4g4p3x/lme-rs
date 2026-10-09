# Proportional marginal-mean weighting, 9 October 2026

Indexing stored nuisance-factor levels once per reference-grid call reduces the
cost of proportional marginal means. The original row loop retrieved each typed
string column, searched the stored level list, and retrieved its cardinality for
every observation. The candidate resolves typed columns and builds borrowed-key
lookup maps before that loop. Joint combination indexing, row-first validation,
floating-point weight accumulation, grid construction, and inference are unchanged.
The extra storage is proportional to the number of stored nuisance levels; it is
local to the call, with no cache or change to the 4,096-row grid limit.

## Workload and attribution

The common harness in `benches/bench_math.rs` generates deterministic data and fits
OLS `y ~ a + b + x` outside the timed region. Factor `a` has two target levels;
factor `b` has two or 32 nuisance levels. Reference frames contain 2,000 or 50,000
observations. Timing includes the complete `emmeans_with_grid` call and result
destruction, with 95% asymptotic intervals. Input construction and fitting are
excluded. The retained benchmark patch specifies the generator completely.

Equal weighting is an attribution control: it shares grid reconstruction and
inference but does not scan nuisance observations. In the initial baseline,
50,000-row equal-weight calls took roughly 57–85 microseconds, compared with
2.29–4.15 milliseconds for proportional calls. Increasing nuisance cardinality
also increased the proportional cost. This identified the observation-to-level
lookup as a useful bounded target; it does not measure fitting performance.

## Measurements

Baseline production source is revision
`293f02b92fa3c7fb1241839ca96f9416cd96461e`, with the common benchmark additions.
Candidate production changes are confined to `src/emmeans.rs`. The
[raw report](perf-tune-proportional-means-2026-10-09.json) records source and binary
hashes, the candidate and harness patches, commands, every sample, per-process
Criterion estimates, outlier classifications, and timing-ratio intervals.

Eight independent processes ran sequentially in ABBA / ABBA order, where A is
baseline and B is candidate. Each case used two seconds of warmup followed by ten
samples over approximately three seconds. Table values are medians of four
process medians of sample-average time per complete call. All samples, including
outliers and control variation, are retained.

| Proportional call | Baseline | Candidate | Time reduction | Candidate/baseline 95% interval |
|---|---:|---:|---:|---:|
| 2,000 observations, 2 nuisance levels | 131.75 µs | 98.68 µs | 25.1% | 0.734–0.820 |
| 2,000 observations, 32 nuisance levels | 218.60 µs | 127.29 µs | 41.8% | 0.557–0.606 |
| 50,000 observations, 2 nuisance levels | 2,279.86 µs | 1,443.14 µs | 36.7% | 0.615–0.646 |
| 50,000 observations, 32 nuisance levels | 4,150.21 µs | 1,462.69 µs | 64.8% | 0.313–0.362 |

| Equal-weight control | Baseline | Candidate | Candidate/baseline 95% interval |
|---|---:|---:|---:|
| 2,000 observations, 2 nuisance levels | 48.83 µs | 48.96 µs | 0.932–1.048 |
| 2,000 observations, 32 nuisance levels | 76.17 µs | 78.91 µs | 0.876–1.246 |
| 50,000 observations, 2 nuisance levels | 56.12 µs | 56.82 µs | 0.987–1.248 |
| 50,000 observations, 32 nuisance levels | 86.19 µs | 86.49 µs | 0.955–1.025 |

Equal-weight medians increased by 0.3–3.6%, with intervals spanning no change.
These control results are inconclusive; the wider intervals cannot exclude larger
changes. The proportional intervals fall below one, and proportional process
medians improved in every baseline/candidate comparison. Intervals use 2,000
hierarchical bootstrap draws with seed `261009`, resampling four processes and
ten samples per process for each version. Four processes per version on one
machine provide limited precision about session variation and no evidence about
other platforms.

The machine was Windows x86_64 with an AMD Ryzen 5 8600G, six cores and twelve
logical processors. Rust and Cargo were 1.99.0. Builds used the unchanged
repository bench profile and default Argmin features. `MKL_NUM_THREADS`,
`OMP_NUM_THREADS`, and `RAYON_NUM_THREADS` were one; `POLARS_MAX_THREADS` retained
its default. `LME_PERF_DIAG` was disabled. Gnuplot was unavailable; measurements
used Criterion's Plotters backend without plots.

## Reproduction and correctness

Build the common harness against the baseline revision and candidate patch,
retaining each executable. For the initial run of each build:

```sh
cargo bench --locked --bench bench_math -- proportional_means --noplot --warm-up-time 2 --measurement-time 3 --save-baseline <session-label>
```

For fresh-process repetitions, invoke the retained executable with `--bench` and
the same filter and options. Use the recorded ABBA / ABBA order and distinct
session labels. Configure thread limits before starting each process and avoid
competing builds or measurements.

Regression coverage in `tests/test_emmeans.rs` compares proportional means and
linear functions against averages of direct predictions and their design rows.
It independently checks variance propagation and pair differences, exercises
joint frequencies with absent combinations, changed reference data, row reversal,
numeric and boolean factors, custom level order, and missing/unknown-level errors.
The regression passed on both baseline and candidate. Existing R fixtures check
uncertainty, conditioning, contrasts, and both weighting policies.

These measurements cover complete asymptotic marginal-mean calls in the listed
workloads. Pairwise comparisons use the same optimized grid path and have
correctness coverage, but their latency was not measured. No cross-library,
complete-fit, or general performance-parity claim is made.
