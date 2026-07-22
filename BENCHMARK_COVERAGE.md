# Benchmark coverage map

This file maps **which parts of `lme-rs` have external performance references** (not just Rust-only Criterion benches). Use it to ground [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) axis (3) and [USABILITY.md](USABILITY.md) performance posture.

**Last assessed:** 2026-07-22

---

## Harness tiers

| Tier | Entry point | External reference | Times | Use for completion % |
|:-----|:------------|:-------------------|:------|:---------------------|
| **A — Fair fit-only** | [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | **MixedModels.jl** (+ GLM.jl for GLMM) | `cold_fit`; optional Rust `prepare_lmer`, `fit_prepared` | **Yes** — axis (3) thresholds |
| **B — Phase breakdown** | [`scripts/run_perf_breakdown.py`](scripts/run_perf_breakdown.py) | Julia `optsum.feval` | Rust phases + Julia eval count | Engineering only |
| **C — Example scripts** | [`scripts/run_cross_language_benchmarks.py`](scripts/run_cross_language_benchmarks.py) | R / Julia / Python / Rust | Whole script (unfair) | Smoke / regression, not axis (3) |
| **D — Criterion** | [`benches/bench_math.rs`](benches/bench_math.rs) | None | Rust-only | Regression guard |

**Numerical parity** ([`tests/test_golden_parity.rs`](tests/test_golden_parity.rs), [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md)) is **correctness**, not throughput.

---

## Axis (3) threshold

Default target: **Rust median &lt; 1.0× Julia median** on `cold_fit` for tier-A LMM cases on the reference workstation (strictly faster than MixedModels.jl).

Prior milestones: **≤ 2×** (through 2026-07-08) while crossed/nested were multi× slower; **≤ 1.5×** (2026-07-09–15); selected crossed/nested strict passes on 2026-07-16. The [2026-07-22 full tier-A reference](benchmarks/fair-rust-julia-reference-2026-07-22-full-tier-a.json) records all 12 cases at the current **1.0×** gate with no failures.

```powershell
python scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia --with-phases --repeats 10
```

Hot-path target (batch / CV): **`fit_prepared` ≤ ~1× Julia `fit`** when `--with-phases` is set (LMM only). Override cold threshold: `--target-ratio 1.5` (legacy bar).

---

## Tier A case catalog

| Case | Model | Fixture | Reference | `cold_fit` vs Julia | `fit_prepared` vs Julia | Notes |
|:-----|:------|:--------|:----------|:--------------------|:------------------------|:------|
| `sleepstudy_reml` | LMM | Real (180 obs, random slopes) | MixedModels.jl | **0.934×** | **0.848×** | Canonical real-world LMM; block LDL fast path |
| `sleepstudy_weighted_reml` | LMM weighted | Real | MixedModels.jl `wts` | **0.923×** | **0.807×** | Same weights as [`benches/bench_math.rs`](benches/bench_math.rs) |
| `penicillin_crossed_reml` | LMM | Real crossed intercept | MixedModels.jl | **0.398×** | **0.305×** | Smaller *n* than `crossed_20k` |
| `pastes_nested_reml` | LMM | Real nested intercept | MixedModels.jl | **0.495×** | **0.390×** | |
| `random_intercept_10k` | LMM | Synthetic | MixedModels.jl | **0.661×** | **0.173×** | |
| `random_intercept_50k` | LMM | Synthetic | MixedModels.jl | **0.569×** | **0.174×** | Single-factor setup fast path |
| `random_intercept_100k` | LMM | Synthetic | MixedModels.jl | **0.509×** | **0.171×** | Single-factor setup fast path |
| `large_random_slopes_100k` | LMM | Synthetic (100k obs; 2k groups) | MixedModels.jl | **0.898×** | **0.646×** | Showcase: correlated intercept/slope, 3 θ; linear cache setup |
| `crossed_20k` | LMM | Synthetic | MixedModels.jl | **0.879×** | **0.677×** | Direct two-factor Gram + allocation-free blocked gate |
| `nested_10k` | LMM | Synthetic | MixedModels.jl | **0.961×** | **0.492×** | Direct slash design + membership Gram |
| `cbpp_binomial_ml` | GLMM | Real binomial | MixedModels.jl GLMM | **0.749×** | N/A | Laplace; not R `nAGQ`-in-θ |
| `grouseticks_poisson_ml` | GLMM | Real Poisson | MixedModels.jl GLMM | **0.033×** | N/A | Laplace |

Cases not in tier A (no fair external fit timing yet):

| Area | Rust bench | Comparable | Status |
|:-----|:-----------|:-----------|:-------|
| `nlmer` / Orange | Golden parity only | `lme4::nlmer` | **No** fair harness |
| GLMM AGQ (`n_agq > 1`) | Criterion | R semantics differ | **No** apples-to-apples |
| Post-fit inference (KR, ANOVA, predict) | Criterion | `lmerTest` / MixedModels | **No** external timing |
| Python `lme_python` FFI | None | N/A | **No** |
| `lm` / `lm_df` | Minimal | R `lm` | **No** (usually negligible) |

---

## What each completion row may claim

| Summary row | Throughput claim valid when |
|:------------|:----------------------------|
| **1 LMM** | Tier-A LMM cases meet `cold_fit` target (or documented exception) |
| **2 GLMM** | Tier-A GLMM cases measured; do not infer from LMM row 13 alone |
| **13 Throughput** | Fraction of tier-A cases at target; see case table above |
| **4 Inference** | Correctness / API only unless tier added |

---

## Running benchmarks

### Full tier A (all cases, with Rust phases)

```powershell
task benchmarks:fair-rust-julia
# or:
python scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia --with-phases --repeats 10
```

### LMM core only (CI smoke when Julia is installed)

```powershell
python scripts/run_fair_rust_julia_benchmark.py --cases sleepstudy_reml,random_intercept_10k --warmups 1 --repeats 2
```

### Julia dependencies

```julia
using Pkg; Pkg.add(["CSV", "DataFrames", "JSON", "MixedModels"])
# GLM only for GLMM tier-A cases (cbpp, grouseticks):
using Pkg; Pkg.add("GLM")
```

[`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl) lazy-loads **GLM** only when `--model` is `glmm_*`, so LMM-only runs do not require GLM installed.

---

## Maintenance

1. After optimization work on a workflow, add or refresh its tier-A case.
2. Record medians in [BENCHMARKS.md](BENCHMARKS.md) and/or commit a dated reference JSON under [`benchmarks/`](benchmarks/).
3. Update **Measured** / **Re-run** cells in the case catalog above.
4. Adjust [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) row **13** only from tier-A evidence.
