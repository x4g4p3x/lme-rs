# Benchmarks

This repository includes Criterion benchmarks for the Rust crate. Performance is an important selling point of `lme-rs`, but the benchmark suite should be described accurately: it is useful and non-trivial, yet it is not comprehensive enough to support blanket speed claims on its own.

## Current benchmark coverage

The existing benchmark target is [benches/bench_math.rs](benches/bench_math.rs).

It currently covers these benchmark families:

- formula parsing for representative random-slope, nested, and crossed formulas
- model-matrix construction for representative random-slope, nested, and crossed datasets
- internal REML deviance evaluation on a fixture-backed random-slopes model
- isolated large-scale REML deviance evaluation on a synthetic sparse model (`50k` observations)
- end-to-end `lmer()` fitting on `sleepstudy`
- end-to-end weighted `lmer_weighted()` fitting on fixture-backed and large synthetic cases
- end-to-end `glmer()` fitting on `grouseticks` and `cbpp`
- population-level and conditional prediction on a larger repeated `sleepstudy` frame
- response-scale and conditional response-scale GLMM prediction on repeated Poisson and Binomial frames
- GLMM post-fit helpers such as Wald confidence intervals
- end-to-end `lmer()` fitting on a large synthetic random-intercept model (`100k` observations)
- end-to-end large crossed-effects and nested-effects fits
- parameterized size and complexity sweeps for random-intercept, crossed-effects, and nested-effects fits
- inference helpers: `with_robust_se()`, `with_satterthwaite()`, `with_kenward_roger()`, and ANOVA paths

That is enough to catch obvious regressions in the parser, design-matrix path, core math path, several end-to-end fits, and a meaningful slice of the inference and post-fit surface.

## What the current suite does well

- Exercises both internal evaluation and public API entry points.
- Covers multiple stages of the modeling pipeline instead of only full-model fits.
- Covers both LMM and GLMM workflows.
- Includes both fixture-backed and synthetic large-scale cases.
- Includes nested and crossed random-effects structures.
- Includes weighted-model coverage.
- Includes parameterized size and complexity sweeps instead of only a few one-off scales.
- Includes several inference-heavy operations that are materially more expensive than fitting alone.
- Uses Criterion, which gives stable repeated measurements and report generation.

## What it does not cover yet

The current benchmark suite is still narrow in a few important ways.

### Missing comparisons

The repo now includes automated cross-language timing for representative example workloads through [scripts/run_cross_language_benchmarks.py](scripts/run_cross_language_benchmarks.py), including Rust, Python, R, and Julia runs where the required runtimes are installed.

What it still does not provide is a fully normalized cross-ecosystem benchmark harness with carefully matched optimizer settings, dataset loading rules, and machine-locked baselines for publication-grade claims.

### Missing workload diversity

It does not yet isolate or compare:

- prediction throughput on large crossed and nested structures
- weighted GLMM-style workflows if those are added later
- GLMM inference and post-fit helper costs beyond the currently benchmarked paths
- confidence interval and simulation throughput across multiple model sizes

### Missing benchmark dimensions

It does not currently sweep over controlled workload dimensions such as:

- number of observations
- number of groups
- random intercept vs random slope structure
- family and link combinations
- dense-like vs sparse-like random-effect structure

## Cross-language benchmark methodology

If you want to make public performance claims against R, Python, or Julia, use a fixed methodology and record it alongside the results. The repo now automates example-level cross-language timing, but the methodology still matters because the ecosystems do not all expose exactly the same optimizers or post-fit APIs.

Recommended rules:

- run all languages on the same machine
- record CPU model, RAM, OS, compiler/interpreter versions, and BLAS backend
- use fixed datasets and fixed formulas
- compare like with like: REML vs REML, ML vs ML, same family, same link, same random-effects structure
- exclude file I/O from timed sections unless I/O is the thing being compared
- warm up the runtime before collecting measurements
- run multiple repetitions and report medians or confidence intervals, not a single best run
- note when optimizers are not directly comparable across ecosystems
- keep output artifacts or raw logs so later releases can be compared to the same baseline

For this repo, the most defensible cross-language benchmark set would start with the same reference datasets already used in `comparisons/COMPARISONS.md`:

- `sleepstudy`
- `dyestuff`
- `pastes`
- `penicillin`
- `cbpp`
- `grouseticks`

## How to run the benchmarks

Run from the repository root:

```bash
cargo bench
```

To compile the benchmarks without executing them:

```bash
cargo bench --no-run
```

Criterion will emit reports under `target/criterion/`.

For automated cross-language timing, run:

```bash
python scripts/run_cross_language_benchmarks.py
```

This script writes JSON output with runtime versions, machine metadata, and per-command timing summaries.

The repository also includes a dedicated workflow in [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml) that:

- runs Criterion benchmarks
- archives `target/criterion`
- runs the cross-language benchmark script
- uploads the resulting artifacts in CI
- attaches them to GitHub Releases on tag pushes

## Latest published results

The latest published benchmark artifacts are attached to the [v0.1.3 release](https://github.com/x4g4p3x/lme-rs/releases/tag/v0.1.3).

That release currently includes:

- `criterion-e8a82e0b97b88c5549bc61e89c22dc12c9060a02.tar.gz`
- `cross-language-e8a82e0b97b88c5549bc61e89c22dc12c9060a02.json`

The cross-language JSON includes representative timings for:

- `sleepstudy`
- `pastes`
- `cbpp`
- `grouseticks`

across:

- Rust
- Python
- R
- Julia

Treat those numbers as versioned release artifacts, not as universal constants. They were produced on GitHub-hosted runners with the workflow's pinned toolchain setup, so they are useful for release-to-release comparison and public transparency, but not as machine-independent proof of absolute speed.

If you want to cite benchmark results in release notes or external docs, prefer linking the release asset directly instead of copying raw numbers into the README. That keeps the landing page stable while still making the measured outputs inspectable.

## How to interpret results

Use the existing suite primarily for:

- detecting regressions after changes to fitting logic
- tracking the cost of core LMM and GLMM paths over time
- sanity-checking large synthetic scaling cases

Do not use the current suite alone as evidence that `lme-rs` is universally faster than `lme4`, `statsmodels`, or `MixedModels.jl`.

## Recommended next extensions

If performance is going to remain a central public claim, the benchmark suite should be extended. The highest-value additions are:

1. Version-to-version regression tracking with saved benchmark outputs in CI or release notes.
2. More controlled cross-language benchmark cases with tighter optimizer comparability notes.
3. Prediction and post-fit sweeps for larger crossed and nested structures than the current medium-scale cases.
4. More explicit GLMM post-fit benchmarks beyond fitting, prediction, and Wald intervals.
5. Benchmarks that isolate optimizer iteration cost separately from formula parsing and matrix construction.

## Recommendation for this repo today

The current benchmarking is meaningful, but not comprehensive.

That means:

- yes, keep performance as a selling point
- no, avoid absolute claims without extending the suite further
- yes, run benchmarks for performance-sensitive changes before release
- yes, extend the benchmark surface if performance is going to stay front-and-center in the README and release notes
