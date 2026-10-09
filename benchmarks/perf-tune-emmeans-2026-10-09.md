# Marginal-mean reference-grid projection, 9 October 2026

Selecting fixed-effect source columns before replicating reference-grid rows
reduces the cost of complete marginal-mean and Tukey pairwise calls. The largest
measured benefit is for reference data containing unused measurements or metadata.

The implementation change is in `src/emmeans.rs`. It retains source-column order
for dot expansion and uses the original observations for numeric means and
proportional nuisance weights. It preserves categorical reconstruction from
stored levels even when the reference frame contains no model columns.

## Measurement

The common harness in `benches/bench_math.rs` uses the 60-row Pastes fixture and
the fitted REML model `strength ~ cask + (1 | batch)`. The grid has three cask
levels. The narrow frame has four columns. The wide frame adds 64 numeric and
64 text columns, each text value containing 512 bytes. The fitted model is the
same for both frames. Timing includes each complete post-fit API call and its
result destruction; model fitting and benchmark-data construction are outside
the timing boundary.

The baseline production source is revision
`1488defc9da12da9862406f8ec70b914a090a878`, with the common benchmark additions.
The candidate changes only reference-grid projection in the production source.
The [raw report](perf-tune-emmeans-2026-10-09.json) retains the candidate patch,
source and executable hashes, fixture hash, environment, samples, and per-process
Criterion estimates and 95% bootstrap intervals.

Four independent processes ran sequentially in baseline/candidate/candidate/baseline
order. Each case received two seconds of warmup and ten measured samples over
approximately three seconds, with thousands of API calls. All samples, including
outliers, are retained. Table values are medians over the two process medians of
sample-average time per call.

| Complete operation | Baseline | Candidate | Time reduction |
|---|---:|---:|---:|
| Marginal means, narrow frame | 45.47 µs | 31.58 µs | 30.6% |
| Tukey pairs, narrow frame | 52.03 µs | 37.36 µs | 28.2% |
| Marginal means, wide frame | 107.51 µs | 39.72 µs | 63.1% |
| Tukey pairs, wide frame | 113.81 µs | 44.36 µs | 61.0% |

For wide-frame means, the two baseline process medians were 105.22 and 109.80 µs;
the candidate medians were 39.41 and 40.03 µs. Wide-frame pair medians were
109.91 and 117.70 µs versus 44.17 and 44.55 µs. Narrow-frame results also improved
in each process. These results cover four post-fit operations on one Windows
x86_64 system with an AMD Ryzen 5 8600G, six cores and twelve logical processors.
Two processes per version provide limited evidence about variation across
machines or sessions; per-process bootstrap intervals do not measure that variation.

Rust was 1.99.0, with the repository's unchanged bench profile and default
Argmin backend. `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, and `RAYON_NUM_THREADS`
were set to one. `POLARS_MAX_THREADS` retained its default; `LME_PERF_DIAG`
was disabled. Gnuplot was unavailable, so Criterion used its Plotters backend
and ran without plots.

## Reproduction and correctness

Build the common benchmark harness separately against the recorded baseline
source and the candidate patch, retaining each executable. Initial measurements
used this command, with a distinct baseline name for each version:

```sh
cargo bench --locked --bench bench_math -- inference/emmeans --noplot --warm-up-time 2 --measurement-time 3 --save-baseline perf-tune-base-1
```

Fresh-process repetitions used the saved executable with `--bench`, the same
filter and timing arguments, and distinct `--save-baseline` names. The JSON
records their actual chronological order and all four process reports.

The harness asserts identical narrow/wide estimates and standard errors.
Regression coverage in `tests/test_bug_hunt_postfit.rs` checks exact agreement
for transformed covariates, colon/star interactions, nuisance weighting,
conditioning factors, intervals, adjusted pairs, and categorical-only source
reconstruction. Existing R fixtures and dot/basis tests run in the consolidated
suite. Rust/Python lint, all-target compilation, and the consolidated suite
passed: 132 unit tests, 328 integration tests, and six doctests. Four heavy
production-load tests remain ignored. Documentation/tooling checks also passed;
two R-dependent tooling tests were skipped because their R runtime was unavailable.
