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
- `glmm_agq_cbpp`: same CBPP binomial fit with `n_agq = 1` (Laplace) vs `n_agq = 7` (scalar AGQ), lower sample size because AGQ is expensive
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

The repo includes two cross-language timing paths:

1. **Example-script timing** — [scripts/run_cross_language_benchmarks.py](scripts/run_cross_language_benchmarks.py) (Rust, Python, R, Julia whole scripts; small fixtures favor prebuilt Rust binaries).
2. **Fair fit-only timing** — [scripts/run_fair_rust_julia_benchmark.py](scripts/run_fair_rust_julia_benchmark.py) (`lme-rs::lmer` vs **MixedModels.jl** with shared CSVs, warmup, and fit-only samples). See [Fair Rust vs Julia reference results](#fair-rust-vs-julia-reference-results) below.

What the repo still does not provide is a fully normalized cross-ecosystem harness with matched optimizer iteration counts and publication-grade baselines for every language pair.

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

This script writes JSON output with runtime versions, machine metadata, and per-command timing summaries. It times **whole example scripts** (including Julia process startup and JIT), so small fixtures favor prebuilt Rust binaries.

### Fair Rust vs Julia fit timing

For a more apples-to-apples throughput comparison against **MixedModels.jl**, use the dedicated harness:

```bash
python scripts/run_fair_rust_julia_benchmark.py
# or
task benchmarks:fair-rust-julia
```

[`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) and [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs) / [`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl):

- generate **shared CSV fixtures** from the same RNG recipes as [`benches/bench_math.rs`](benches/bench_math.rs)
- load each dataset **once** before timing
- run **warmup fits** (including Julia JIT) before measured samples
- time **only the model fit** (`lme-rs::lmer` vs `MixedModels.fit`), excluding CSV I/O and process startup
- default cases: `sleepstudy_reml`, `random_intercept_10k/50k/100k`, `crossed_20k`, `nested_10k`
- write JSON to `benchmark-results/fair-rust-julia-benchmarks.json` with per-case medians and `rust_over_julia_median` ratios

Julia is resolved from `--julia`, `JULIA_BIN`, `PATH`, or (on Windows) `%LOCALAPPDATA%\Programs\Julia-*\bin\julia.exe`. Requires Julia packages `CSV`, `DataFrames`, `JSON`, and `MixedModels`.

For **where time goes** inside a fit (Rust phase breakdown + Julia `optsum.feval` on the same CSV), use [`scripts/run_perf_breakdown.py`](scripts/run_perf_breakdown.py) (`task benchmarks:perf-breakdown`). The Rust runner reports **`prepare_wall_seconds`**, **`fit_prepared_wall_seconds`** (hot path), and **`blocked_kernel`** alongside `LME_PERF_DIAG` phases. See [OPTIMIZATION.md § Performance diagnostics](OPTIMIZATION.md#performance-diagnostics-lme_perf_diag).

**Caveats:** optimizers and likelihood paths still differ between `lme-rs` and MixedModels.jl; use this harness for **throughput**, not coefficient identity. ML (`reml=false`) is used on synthetic sweeps to match Criterion; `sleepstudy_reml` uses REML.

<a id="fair-rust-vs-julia-reference-results"></a>

### Fair Rust vs Julia reference results

Checked-in summary: [benchmarks/fair-rust-julia-reference-2026-07-06.json](benchmarks/fair-rust-julia-reference-2026-07-06.json) (current); prior datapoint [benchmarks/fair-rust-julia-reference-2026-07-04.json](benchmarks/fair-rust-julia-reference-2026-07-04.json).  
Full per-sample JSON from the same run can be reproduced locally as `benchmark-results/fair-rust-julia-benchmarks.json`.

**Recorded:** 2026-07-06 on a Windows 10 AMD64 workstation (12 logical CPUs).  
**Toolchain:** `lme-rs` at git `76fdb61`, `rustc 1.96.0`, Julia **1.12.6**, MixedModels.jl **5.7.0** (Julia medians from the 2026-07-04 run on the same machine).  
**Method:** 2 warmup fits + 5 measured fits per case; median wall time of the fit call only.

| Case | Formula | *n* | Rust median | Julia median | Julia faster (median ratio) |
|:-----|:--------|----:|------------:|-------------:|------------------------------:|
| `sleepstudy_reml` | `Reaction ~ Days + (Days \| Subject)` | 180 | 2.71 ms | 0.77 ms | **3.5×** |
| `random_intercept_10k` | `y ~ x + (1 \| group)` | 10 000 | 3.16 ms | 1.14 ms | **2.8×** |
| `random_intercept_50k` | `y ~ x + (1 \| group)` | 50 000 | 12.1 ms | 6.00 ms | **2.0×** |
| `random_intercept_100k` | `y ~ x + (1 \| group)` | 100 000 | 24.1 ms | 11.9 ms | **2.0×** |
| `crossed_20k` | `y ~ x + (1 \| plate) + (1 \| sample)` | 20 000 | 272 ms | 14.3 ms | **19×** |
| `nested_10k` | `y ~ x + (1 \| batch/cask)` | 10 000 | 53.8 ms | 6.88 ms | **7.8×** |

**Takeaway:** after caching [`LmmData`](src/math.rs) in the θ optimizer, precomputing `Z^T X` / `Z^T y`, and an intercept-only diagonal-Λ fast path (git `76fdb61`), **random-intercept cases on this machine are within ~2–3× of MixedModels.jl** (down from ~5–9× on the 2026-07-04 reference). **Nested** improved (~28× → ~8×). **Crossed** remains the outlier (~19×). Synthetic cases used ML (`reml=false`); sleepstudy used REML.

<a id="fair-rust-julia-2026-07-07-wip"></a>

### 2026-07-07 intercept-only optimization pass

Work on [`src/math.rs`](src/math.rs) and [`src/optimizer.rs`](src/optimizer.rs): reused sparse LDL, deviance-only intercept hot path, hand-unrolled **p = 1 / p = 2** profile finishes, precomputed θ block indices, golden-section θ search for |θ| = 1, and a two-stage **2D log-grid** for crossed ML (REML: grid + short Nelder–Mead polish for golden parity). Golden parity passes. Engineering detail (what worked, what was reverted, Julia lessons) lives in **[OPTIMIZATION.md](OPTIMIZATION.md)**.

**Recorded:** 2026-07-07, same Windows AMD64 workstation as the 2026-07-06 reference; `rustc 1.96.0`; 2 warmups + 10 measured fits (`scripts/run_fair_rust_julia_benchmark.py --implementations rust`).

| Case | 2026-07-06 Rust median | After LDL pass | After θ-search pass | Julia median (2026-07-06) | Julia faster |
|:-----|-----------------------:|---------------:|--------------------:|--------------------------:|-------------------:|
| `random_intercept_10k` | 3.16 ms | 2.78 ms | **~2.5 ms** | 1.14 ms | **~2.3×** |
| `crossed_20k` | 272 ms | 109 ms | **~113 ms** | 14.3 ms | **~7.9×** |
| `nested_10k` | 53.8 ms | 17.0 ms | **~15.5 ms** | 6.88 ms | **~2.3×** |

**Takeaway:** the LDL reuse pass delivered most of the crossed / nested speedup (~2.5× and ~3× vs 2026-07-06 Rust). The θ-search pass improved nested and random-intercept slightly and fixed ML eval budget for crossed, but **did not materially beat** the LDL-only crossed median. Julia still leads on all three; crossed remains the main gap (~8×). See [OPTIMIZATION.md § Why MixedModels.jl is faster](OPTIMIZATION.md#why-mixedmodelsjl-is-faster-and-what-to-learn) for structural reasons and next steps (blocked augmented Cholesky).

<a id="fair-rust-julia-2026-07-08-blocked-cholesky"></a>

### 2026-07-08 blocked augmented Cholesky (MixedModels-style)

Implemented [`src/intercept_blocked.rs`](src/intercept_blocked.rs): precomputed per-RE Gram blocks, RE ordering by level count, and per-θ **blocked Cholesky** in the layout of MixedModels.jl [`updateL!`](https://github.com/JuliaStats/MixedModels.jl/blob/main/src/linearmixedmodel.jl) — profile deviance from the factored `Xy` block **without** q-dimensional LDL solves or explicit β on the hot path. Full write-up: **[OPTIMIZATION.md § Blocked augmented Cholesky](OPTIMIZATION.md#blocked-augmented-cholesky--what-delivered-the-crossed-speedup-2026-07-08)**.

**Recorded:** 2026-07-08, same Windows AMD64 workstation; `rustc 1.96.0`; 2 warmups + 10 measured fits (`scripts/run_fair_rust_julia_benchmark.py --implementations rust`).

| Case | 2026-07-06 Rust | After θ-search (07-07) | After blocked Cholesky | Julia median (2026-07-06) | Julia faster |
|:-----|----------------:|-----------------------:|-----------------------:|--------------------------:|-------------------:|
| `random_intercept_10k` | 3.16 ms | ~2.5 ms | **~3.0 ms** | 1.14 ms | **~2.6×** |
| `crossed_20k` | 272 ms | ~113 ms | **~68 ms** | 14.3 ms | **~4.8×** |
| `nested_10k` | 53.8 ms | ~15.5 ms | **~16.5 ms** | 6.88 ms | **~2.4×** |

**Takeaway:** crossed median **~1.7× faster** than the 07-07 LDL path and **~4× faster** than the 2026-07-06 reference — the largest single-step gain so far. Nested and random-intercept unchanged (blocked path gated off: single RE or cross block too large for dense storage). Julia still leads; crossed gap narrowed from ~**8×** to ~**5×** on this machine.

<a id="fair-rust-julia-2026-07-08-blocked-hotpath"></a>

### 2026-07-08 blocked Cholesky hot-path pass

Removed per-θ **clone/alloc** overhead in [`src/intercept_blocked.rs`](src/intercept_blocked.rs) (in-place Schur, fused `assign_scaled`, reused workspaces) and tightened the ML 2D log-grid to **5×5 + 4×4** (~42 evals). Nested sparse crosses remain on reused LDL (`fits_blocked_gate`). Details: **[OPTIMIZATION.md § Closing the crossed_20k gap](OPTIMIZATION.md#closing-the-crossed_20k-gap-vs-julia-14-ms)**.

**Recorded:** 2026-07-08, same Windows AMD64 workstation; `rustc 1.96.0`; 2 warmups + 10 measured fits (`--implementations rust`).

| Case | After blocked Cholesky | After hot-path pass | After Schur scratch + row-dot | Julia median (2026-07-06) | vs Julia |
|:-----|-----------------------:|--------------------:|------------------------------:|--------------------------:|---------:|
| `random_intercept_10k` | ~3.0 ms | **~2.9 ms** | **~2.8 ms** | 1.14 ms | **~2.5×** |
| `crossed_20k` | ~68 ms | **~52 ms** | **~52–60 ms** (bimodal) | 14.3 ms | **~3.6–4.2×** |
| `nested_10k` | ~16.5 ms | **~13 ms** | **~14 ms** | 6.88 ms | **~2.0×** |

**Takeaway:** crossed improved **~1.3×** again (~68 ms → **~52 ms** best cluster); a second pass reused Schur scratch buffers (no cross-block `clone`) and row-dot kernels in rank/Schur updates. Remaining crossed gap vs Julia was still ~**3.6×+** on cold `lmer()` — mostly setup + per-eval constant factor.

<a id="fair-rust-julia-2026-07-08-gemm-prepared"></a>

### 2026-07-08 GEMM, batched trisolve, and prepared fit

Further blocked-kernel tuning in [`src/intercept_blocked.rs`](src/intercept_blocked.rs) plus **`prepare_lmer` / `fit_prepared`** in [`src/lib.rs`](src/lib.rs). Full write-up: **[OPTIMIZATION.md § Blocked kernel tuning](OPTIMIZATION.md#blocked-kernel-tuning--gemm-batched-trisolve-prepared-fit-2026-07-08-continued)**.

**Recorded:** 2026-07-08, same Windows AMD64 workstation; `rustc 1.96.0`; `bench_perf_breakdown` with 1 warmup + measured `fit_prepared`.

| Case | Cold `lmer()` | `fit_prepared` | `prepare_lmer` | Julia `fit` median (2026-07-06) | vs Julia (hot) |
|:-----|--------------:|---------------:|---------------:|--------------------------------:|---------------:|
| `crossed_20k` | ~25–28 ms | **~12–14 ms** | ~10–13 ms | 14.3 ms | **~1.0×** |
| `nested_10k` | ~16 ms | ~14 ms | ~10 ms | 6.88 ms | ~2.0× |

**Takeaway:** on `crossed_20k`, Rust **matches Julia** when setup is amortized (`fit_prepared`). Per-deviance-eval time (~0.23 ms) is at or below Julia (~0.24 ms). Cold `lmer()` is still ~2× Julia because Rust times formula parse + design-matrix build + post-fit assembly explicitly. Tried `ndarray` BLAS for GEMM — **no measurable win** at these block sizes; not enabled. Fixed intermittent sparse-LDL fallback by always densifying cross blocks ≤100k elements.

<a id="fair-rust-julia-2026-07-08-setup-postfit"></a>

### 2026-07-08 setup and post-fit pass

Targets cold-`lmer()` overhead outside θ search: design-matrix construction, intercept-cache setup, and post-fit `evaluate()`. Full write-up: **[OPTIMIZATION.md § Setup and post-fit pass](OPTIMIZATION.md#setup-and-post-fit-pass-2026-07-08-continued)**.

**Recorded:** 2026-07-08, same Windows AMD64 workstation; `rustc 1.96.0`, Julia **1.12.6**, MixedModels.jl **5.7.0**; 2 warmups + 10 measured fits (`--implementations rust,julia`).

| Case | Cold `lmer()` Rust | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **22.4 ms** | 16.1 ms | **1.39×** |
| `nested_10k` | **12.2 ms** | 7.4 ms | 1.65× |
| `random_intercept_10k` | **1.6 ms** | 1.4 ms | **1.14×** |

`bench_perf_breakdown` on `crossed_20k`:

| Phase | Before | After |
|:------|-------:|------:|
| `prepare_lmer` | ~10–13 ms | **~4.8 ms** |
| `fit_prepared` | ~12–14 ms | **~12.1 ms** |
| `lmer_post_fit` | ~4–5 ms | **~1.8 ms** |

**Takeaway:** `prepare_lmer` **~2× faster** (direct CSR `Zᵀ`, skip sparse LDL when blocked active, no unweighted clones). Post-fit **~2.5× faster** (lazy sparse LDL reuse in `evaluate()`). **`fit_prepared` beats Julia** on crossed (~12 ms vs ~16 ms); cold `lmer()` gap narrowed from **~1.7× to ~1.4×** on crossed and **~2.4× to ~1.1×** on random intercept. Fixed Julia fair harness (`ProgressMeter.enable` removed).

<a id="fair-rust-julia-2026-07-08-prepare-ownership"></a>

### 2026-07-08 prepare ownership pass

Eliminates duplicate `x` / `zt` / `y` / `re_blocks` clones in [`prepare_lmer`](src/lib.rs) and speeds categorical dummy encoding in [`build_x_matrix`](src/model_matrix.rs). Full write-up: **[OPTIMIZATION.md § Prepare ownership pass](OPTIMIZATION.md#prepare-ownership-pass-2026-07-08-continued)**.

**Recorded:** 2026-07-08, same Windows AMD64 workstation; `rustc 1.96.0`, Julia **1.12.6**, MixedModels.jl **5.7.0**; 2 warmups + 10 measured fits (`--implementations rust,julia`).

| Case | Cold `lmer()` Rust | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **21.1 ms** | 15.7 ms | **1.34×** |
| `nested_10k` | **12.5 ms** | 6.9 ms | 1.81× |
| `random_intercept_10k` | **1.3 ms** | 1.7 ms | **Rust faster** |

**Takeaway:** `mem::take` into `LmmData` drops a full copy of design matrices at prepare time. **`random_intercept_10k` cold `lmer()` now beats Julia** on this machine; crossed gap narrowed to **~1.3×**. Nested kernel gap unchanged (still sparse LDL).

<a id="fair-rust-julia-2026-07-08-simple-x"></a>

### 2026-07-08 simple fixed-effects fast path

[`try_build_simple_x_matrix`](src/model_matrix.rs) for `y ~ 1`, `y ~ x`, and `y ~ 1 + x` (fair-harness fixtures). See **[OPTIMIZATION.md § Prepare ownership pass](OPTIMIZATION.md#prepare-ownership-pass-2026-07-08-continued)** (simple fixed-effects subsection).

**Recorded:** 3 warmups + 20 repeats, same workstation.

| Case | Cold `lmer()` Rust | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **22.0 ms** | 15.2 ms | 1.45× |
| `nested_10k` | **11.9 ms** | 7.6 ms | 1.57× |
| `random_intercept_10k` | **1.3 ms** | 1.5 ms | **Rust faster** |

`prepare_lmer` on `crossed_20k`: **~4.2 ms**. No regression on `random_intercept_10k` vs the prepare-ownership commit (A/B: `inv_from_chol_lower` post-fit change regressed that case and was **not** shipped).

<a id="fair-rust-julia-2026-07-08-prepare-fast-paths"></a>

### 2026-07-08 prepare fast paths (ZᵀZ + nested interaction groups)

Obs-major `ZᵀZ` accumulation for intercept-only models with `q ≥ 256` and fast nested interaction group indexing (`batch:cask`). Full write-up: **[OPTIMIZATION.md § Prepare fast paths pass](OPTIMIZATION.md#prepare-fast-paths-pass-2026-07-08-continued)**.

**Recorded:** 3 warmups + 20 repeats, same workstation.

| Case | Cold `lmer()` Rust | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **25.1 ms** | 15.6 ms | 1.61× |
| `nested_10k` | **10.7 ms** | 7.0 ms | **1.53×** |
| `random_intercept_10k` | **1.2 ms** | 1.3 ms | **Rust faster** |

`prepare_lmer` on `nested_10k`: **~5.3 ms** (was ~7 ms). `fit_prepared` on nested **~4 ms** beats Julia **~6.6 ms** (`bench_perf_breakdown`). Crossed cold wall time unchanged in shape: hot fit ≈ Julia; gap is post-fit sparse LDL init (~2 ms).

**How to read this:**

- These numbers are **machine- and version-specific**; Linux CI or different BLAS builds may differ. Re-run the harness before citing new hardware.
- `lme-rs` uses derivative-free variance-component search; MixedModels.jl uses its own optimizer stack — iteration counts and convergence paths are not matched.
- A few Julia samples were noisy (e.g. one 271 ms repeat on `crossed_20k` vs 12–14 ms others); **medians** still favored Julia on all cases. Prefer medians over single runs.
- **Fair timing ≠ lme4 parity.** This benchmark answers “how fast is the fit on shared data?”, not “which matches R?” — the project’s parity evidence remains in [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) and golden tests.
- **Implication for Julia bindings:** wrapping `lme-rs` in Julia would not improve fit throughput versus MixedModels.jl on this baseline; the Rust crate’s value is **native Rust workflows and lme4-aligned numerics**, not beating Julia on speed.

Tag/manual CI (`.github/workflows/benchmarks.yml`) also runs this harness on `ubuntu-latest` and uploads `benchmark-results/fair-rust-julia-<sha>.json` alongside the example-script cross-language JSON.

The repository also includes a dedicated workflow in [.github/workflows/benchmarks.yml](.github/workflows/benchmarks.yml) that:

- runs Criterion benchmarks
- archives `target/criterion`
- runs the fair Rust vs Julia fit benchmark
- runs the cross-language example-script benchmark
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

The [fair Rust vs Julia harness](#fair-rust-vs-julia-reference-results) on the 2026-07-06 Windows reference showed **MixedModels.jl still faster on every fit-only case**, but the gap **narrowed sharply** on random-intercept workloads (~**2×** vs ~**5–9×** on the 2026-07-04 baseline). Crossed (~**19×**) and nested (~**8×**) were the main gaps on that reference. An [2026-07-07 pass](#fair-rust-julia-2026-07-07-wip) cuts crossed to ~**8×** and nested to ~**2.5×**; [blocked Cholesky](#fair-rust-julia-2026-07-08-blocked-cholesky) and [hot-path tuning](#fair-rust-julia-2026-07-08-blocked-hotpath) cut cold `lmer()` to ~**52 ms**; a [GEMM + prepared-fit pass](#fair-rust-julia-2026-07-08-gemm-prepared) brings **`fit_prepared` to ~13 ms** (~**1×** Julia on `crossed_20k`); a [setup/post-fit pass](#fair-rust-julia-2026-07-08-setup-postfit) cuts cold `lmer()` to **~22 ms** (~**1.4×** Julia on crossed, **~1.1×** on random intercept); a [prepare ownership pass](#fair-rust-julia-2026-07-08-prepare-ownership) brings **`random_intercept_10k` ahead of Julia** and crossed to **~1.3×**. See [OPTIMIZATION.md](OPTIMIZATION.md) for engineering detail. Treat these as versioned datapoints — re-run the harness on your hardware before citing speed claims.

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

- yes, keep performance as a **completion criterion** for Rust-native workflows, not only regression tracking (Criterion + fair harness)
- yes, prioritize **LMM fit optimization** (especially crossed/nested RE) until fair-harness medians are within ~2× of MixedModels.jl — see [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) row 13
- no, avoid claims that `lme-rs` already beats **MixedModels.jl** on fit throughput
- no, Julia bindings to `lme-rs` are not justified by speed — see [fair Rust vs Julia results](BENCHMARKS.md#fair-rust-vs-julia-reference-results)
- yes, run benchmarks for performance-sensitive changes before release
- yes, extend the benchmark surface (GLMM fit-only, prediction sweeps) as optimization work proceeds
