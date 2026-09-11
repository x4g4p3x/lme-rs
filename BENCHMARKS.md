# Performance and benchmarks

[Documentation](docs/README.md) · [Live dashboard](https://x4g4p3x.github.io/lme-rs/benchmarks/) · [Coverage](BENCHMARK_COVERAGE.md) · [Optimizer choice](docs/BASIN.md)

Choose a benchmark that matches the question you need to answer. A fast fit is
useful only when it solves the intended model and reports convergence honestly.

> **Reading a result:** check the model, revision, numerical agreement, and timing
> boundary before comparing speed. A lower median alone is not a verified win.

## Start with the results

The [dashboard](https://x4g4p3x.github.io/lme-rs/benchmarks/) presents current
checked-in results, their qualifications, and source evidence. The
[coverage map](BENCHMARK_COVERAGE.md) lists the workloads included in each suite.
The [Basin guide](docs/BASIN.md) explains the optional optimizer and the evidence
needed before changing the default.

The [11 September 2026 refresh](benchmarks/refresh-2026-09-11.md) records the
full suite, its numerical gaps, and the decision to retain Argmin as default.
Use the dated report when citing these measurements.

Historical measurements are preserved in [the measurement archive](BENCHMARK_HISTORY.md).
Their dates and revisions matter: a July result does not describe September's
optimizer or convergence behavior.

## Choose a benchmark

| Your question | Run | What the result means |
|:---|:---|:---|
| How does a complete fit compare with Julia? | Fair Rust/Julia harness | Shared data, warmed runtimes, fit-only samples |
| Is Basin competitive with Argmin? | Repeated backend comparison | Matched fits, alternating execution order, independent process runs |
| Which Rust operation became slower? | Criterion | Microbenchmarks and model operations on one machine |
| Where does fitting time go? | Phase breakdown | Setup, optimization, post-fit work, and objective evaluations |
| What do nonlinear fits, inference, and Python calls cost? | External timings | Selected Rust/R operations and Python binding overhead |
| Do the language examples still run? | Whole-script suite | Process time, including startup and Julia compilation |

The whole-script suite is useful for regressions and compatibility. Its timings
do not establish which statistical fitting engine is faster.

## Run the suites

Run these commands from the repository root. Rust numerical work uses release
builds. Keep benchmark runs sequential and avoid other CPU-intensive work.

### Fair Rust versus Julia

```sh
python scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia --with-phases --warmups 3 --repeats 11 --threads 1 --output benchmark-results/fair-argmin.json
```

This runs all 12 cases. Add `--rust-features basin` to measure Basin. To inspect
one workload, use `--cases sleepstudy_reml` or a comma-separated case list.
Run a second session with `--order julia-first` to check execution-order drift.

The driver checks the binary's backend label and records the revision, source
and binary hashes, input hashes, runtime versions, and thread settings.
If you reuse a binary with `--skip-rust-build`, specify its matching features.

### Argmin versus Basin

```sh
python scripts/run_optimizer_comparison.py --output benchmark-results/optimizer-comparison.json
```

This runs the same 12 workloads in ABBA / BAAB / ABBA order: six independent
processes per backend, three warmups and eleven measurements per process. A is
Argmin and B is Basin. Each process records its compiled backend and every
measured fit's numerical checks. The result retains all source reports.

The reported ratio is the median of three paired block ratios. Each block uses
the median of its two process medians per backend. The accompanying backend
times are medians over six process medians, so dividing those displayed times
need not reproduce the paired ratio. Intervals use 2,000 bootstrap resamples of
the three blocks with seed `260911`; three blocks provide limited precision.

A case meets the predeclared 5% margin only when its fits agree and the upper
ratio interval is at most `1.05`. Passing some cases does not justify a default
change. Specialized scalar searches are unchanged by the feature, so their
timing variation is also a control for build and machine effects.

To reuse separately built executables, pass `--argmin-target <directory>` and
`--basin-target <directory>`. The driver verifies each binary's backend label.
Otherwise it builds each requested configuration before measurement.

### Criterion and production workloads

```sh
cargo bench --locked --bench bench_math -- --noplot
cargo bench --locked --features slow-production-benches --bench bench_load_production -- --noplot
```

Add `--features basin` for the main suite, or
`--features basin,slow-production-benches` for production workloads. Criterion
reports live in `target/criterion/`. Use named baselines when comparing builds;
do not silently reuse a baseline from another machine or compiler.

### Phase breakdown

```sh
python scripts/run_perf_breakdown.py --cases crossed_20k,nested_10k,random_intercept_10k
```

Add `--rust-features basin` for Basin and `--output <file>` to retain a dated
report. Reused binaries honor `CARGO_TARGET_DIR` and must match the feature.

Use this to distinguish setup costs from optimizer work. Diagnostic counters
add overhead, so collect them separately from throughput samples. See the
[engineering guide](OPTIMIZATION.md#performance-diagnostics-lme_perf_diag).

### R, Python, and complete examples

```sh
python scripts/run_external_timings.py --warmups 3 --repeats 11 --output benchmark-results/external.json
python scripts/run_cross_language_benchmarks.py --warmups 2 --repeats 10 --output benchmark-results/cross-language.json
```

The external suite includes four Rust/R cases and a Python binding case.
Use a Python environment containing the locally built `lme_python` extension
when measuring binding overhead. The whole-script suite also needs `statsmodels`
for its independent Python examples.

R operation timings calibrate batches to at least 0.2 seconds before measuring.
Each reported sample is batch elapsed time divided by its call count, including
loop overhead. Raw batch times, counts, and extra calibration calls are retained.
This avoids zero-duration results on coarse clocks; it does not establish
cross-library numerical equivalence or identical setup boundaries.

## Prepare the reference runtimes

| Runtime | Required packages | Check |
|:---|:---|:---|
| Rust | Repository's locked dependencies | `cargo --version` |
| Julia | `CSV`, `DataFrames`, `JSON`, `MixedModels`, `GLM` | `julia --version` |
| R | `lme4`; `lmerTest`, `pbkrtest`, `car`, `rlang` for inference and broader examples | `Rscript --version` |
| Python | Local `lme_python`, `polars`; `pandas`, `statsmodels` for reference examples | Use the repository's Python environment |

On Windows, R may be installed under `C:\Program Files\R\R-<version>\bin`
without `Rscript` being on `PATH`. Check that location before concluding that
R is absent. Formatting additionally needs R's `styler` and Julia's
`JuliaFormatter`; they are not modeling dependencies.

Missing runtimes and packages must appear as skips or failures in the report.
A partially executed suite must not be described as a complete comparison.

## How to interpret results

### Compare the same work

| Metric | Included | Appropriate comparison |
|:---|:---|:---|
| `cold_fit` | Model construction, optimization, and returned fit | Another complete fit with matching data and settings |
| `prepare_lmer` | Reusable model/design preparation | Preparation alone |
| `fit_prepared` | Fit using a prepared Rust design | Repeated-fit costs; comparing with a fresh Julia model is diagnostic |
| Whole-script time | Startup, imports, JIT, loading, and analysis | The same complete script over time |

The fair harness checks the formula, family, ML/REML choice, observations, and
every measured fit. LMM speed claims additionally require converged fits with
matching objectives and fixed effects. Weighted objective conventions and GLMM
likelihood constants can prevent qualification even when coefficients look close.
Those cases stay visible with the reason; they are not counted as wins.

### Read uncertainty with the median

At least two warmups and ten measurements are required for a speed assessment.
Shorter runs are smoke tests. The driver recomputes summaries from samples and
estimates a 95% timing-ratio interval with 2,000 bootstrap resamples.

An interval spanning the target is inconclusive. These intervals describe the
measured process; they do not capture changes between machines or sessions.
For optimizer decisions, alternate the backend order and compare independent
process runs as well as individual samples.

### Keep correctness and completion separate

Golden fixtures test numerical correctness. Benchmarks measure execution cost.
The [completion manifest](completion_manifest.json) defines separate, locked
commitments. A new API, a fast case, or a refreshed dashboard does not by itself
complete one of those commitments.

## Publish and maintain results

1. Save dated raw reports under `benchmarks/`, including unsuccessful or
   unqualified cases and their reasons.
2. Record the revision, optimizer, compiler, runtime versions, thread limits,
   warmups, repeats, and any skipped suite.
3. Update the dashboard's selected inputs and run `task benchmarks:site`.
4. Run `task benchmarks:test`, `task docs:check`, and applicable code checks.
5. Verify the GitHub Pages deployment and the data it serves.

Keep older evidence available. See [optimization guidance](OPTIMIZATION.md) for
numerical guardrails and [historical results](BENCHMARK_HISTORY.md) for past experiments.

<a id="fair-rust-vs-julia-reference-results"></a>
<a id="fair-rust-vs-julia-2026-07-07-wip"></a>
<a id="large-random-slopes-showcase"></a>
<a id="native-formula-parser-optimization-2026-08-10"></a>

## Historical references

Older deep links are retained here. Continue to the archived
[fair-harness results](BENCHMARK_HISTORY.md#fair-rust-vs-julia-reference-results),
[large random-slope experiments](BENCHMARK_HISTORY.md#large-random-slopes-showcase),
or [parser measurements](BENCHMARK_HISTORY.md#native-formula-parser-optimization-2026-08-10).
