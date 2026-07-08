# LMM fit optimization

Engineering notes for **LMM variance-component (θ) search** throughput. For harness methodology and versioned timings, see [BENCHMARKS.md](BENCHMARKS.md). For completion goals, see [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) row 13.

**Read this before changing** [`src/math.rs`](src/math.rs) intercept-only paths or [`src/optimizer.rs`](src/optimizer.rs) θ search.

## Goal

On the [fair Rust vs Julia harness](BENCHMARKS.md#fair-rust-vs-julia-reference-results), reach **within ~2× of MixedModels.jl** on all six reference cases **without breaking** golden parity or numerical tests.

Hardest case: **`crossed_20k`** — `y ~ x + (1 | plate) + (1 | sample)`, ML, 20k obs, q ≈ 350, p = 2, |θ| = 2.

## Architecture (two paths)

θ optimization calls `LmmData::log_reml_deviance` many times; final `evaluate()` runs once at the converged θ.

| Path | Entry | Used for | Must |
|:-----|:------|:---------|:-----|
| **Optimizer hot** | `profile_deviance` → `profile_deviance_diagonal` → `InterceptLdlCache::profile_deviance` | θ search cost | Fast; return `f64::MAX` on infeasible θ (no panic) |
| **Blocked hot (crossed)** | `InterceptBlockedChol::profile_deviance` when cross blocks fit in memory | θ search on intercept-only crossed models | Same deviance as slow path; gated by cross-block size (see below) |
| **Full profile** | `solve_profile` → `solve_profile_diagonal` | `evaluate()`, SEs, fitted values | Correct; fresh LDL per θ |

**Invariant:** do **not** route `solve_profile_diagonal` through `InterceptLdlCache` until parity is proven on all golden intercept-only cases. A cached evaluate path broke `crossed_20k` and golden parity during development.

### Intercept-only fast path (when it applies)

Enabled when every RE block has **k = 1** (`intercept_only_re`).

Key pieces in [`src/math.rs`](src/math.rs):

- **`LmmData`** — precomputed `zt_x`, `zt_y`, `zt_z`, `xt_x`, `xt_y`, `y_norm2`; `Arc` reuse in [`src/optimizer.rs`](src/optimizer.rs).
- **`InterceptLdlCache`** — `Mutex` holding reused sparse LDL symbolic structure; numeric `update()` per θ.
- **`row_block`** — maps each random-effect row to its variance-component index (block scaling instead of full `d` vector on the hot path).
- **`profile_deviance_p2`** — hand-unrolled 2×2 SPD solve for fixed effects when p = 2 (avoids LAPACK on the hot path).
- **`profile_deviance_p1`** — hand-unrolled 1×1 β solve when p = 1 (random intercept, nested).
- **`nz_theta_i` / `nz_theta_j`** — precomputed θ block indices per `A` nonzero in `InterceptSparseLdl::factor_blocks`.

General random-slopes / nested correlated blocks still use `solve_profile_general` (no intercept fast path).

### Intercept-only θ search ([`src/optimizer.rs`](src/optimizer.rs))

When `LmmData::intercept_only_re()` is true, `optimize_theta_lmm` dispatches by |θ|:

| |θ| | Method | ~eval budget |
|:--|:-------|:-------------|
| 1 | Golden-section on a local log bracket | ~25 |
| 2 | 5×5 global log-grid + local 4×4 log-grid; REML adds short Nelder–Mead polish | ML ~42; REML ~41 grid + ≤20 NM iters |
| >2 | Nelder–Mead (unchanged) | variable |

**ML vs REML for |θ| = 2:** grid-only is enough for the fair `crossed_20k` harness (`reml=false`). **REML** golden fixtures (e.g. `penicillin_crossed_reml`) need the NM polish after the grid — grid-only broke parity.


### Sparse vs dense backend

`INTERCEPT_DENSE_MAX_Q` gates an optional **dense Cholesky** backend in `InterceptDenseChol`. It is **`0` (disabled)** today: dense assembly was **wrong** (golden θ failures) and **slow** (~15 s/fit on `crossed_20k` when rebuilding via `TriMat` each θ).

Keep sparse LDL reuse for q ≈ 350 until dense matches `build_a_diagonal_scaled` + LDL log-det exactly and benchmarks win.

## What improved

| Change | Cases helped | Notes |
|:-------|:-------------|:------|
| Cached `LmmData` in θ optimizer | All LMM | git `76fdb61`; documented in [CHANGELOG.md](CHANGELOG.md) |
| Precomputed `Z^T X`, `Z^T y` | All LMM | Avoids repeated sparse×dense products per θ |
| Reused **sparse LDL** symbolic + `update()` | `crossed_20k`, `nested_10k` | 2026-07-07 pass |
| Deviance-only intercept path | Intercept-only θ search | Skips `u`, `b`, `v_cols` allocations |
| `profile_deviance_p2` (p = 2) | `crossed_20k` | 2×2 β solve without LAPACK on hot path |
| Reused `w_y` / `w_col` buffers in cache | Crossed / nested intercept | No per-θ `Vec` clones |
| Precomputed `nz_theta_i/j` in `factor_blocks` | Crossed / nested intercept | Fewer indirect lookups per nnz |
| `profile_deviance_p1` (p = 1) | `random_intercept_*`, nested | No LAPACK on 1×1 β finish |
| Golden-section θ search (|θ| = 1) | Random intercept, nested | Replaces Nelder–Mead simplex overhead |
| 2D log-grid θ search (|θ| = 2, ML) | `crossed_20k` | ~42 fixed evals (5×5 + 4×4 log-grids) |
| **Blocked augmented Cholesky** (`intercept_blocked.rs`) | `crossed_20k` (cross block ≤100k) | MixedModels-style `updateL!`; no q solves per θ |
| **Blocked kernel tuning** (GEMM, batched trisolve, prepared fit) | `crossed_20k` | ~12–14 ms `fit_prepared`; per-eval ≈ Julia; see § below |
| **`prepare_lmer` / `fit_prepared`** | Repeated fits on same data | Amortizes setup (~10–13 ms); hot path ≈ Julia `fit` only |

Fair-harness snapshot (Windows AMD64, 10 repeats) — see [BENCHMARKS.md § 2026-07-07](BENCHMARKS.md#fair-rust-julia-2026-07-07-wip):

| Case | 2026-07-06 Rust | After LDL pass | After θ-search | After blocked Cholesky | After GEMM + prepared fit (2026-07-08) | vs Julia (2026-07-06) |
|:-----|----------------:|---------------:|---------------:|-----------------------:|---------------------------------------:|----------------------:|
| `random_intercept_10k` | 3.16 ms | 2.78 ms | ~2.5 ms | **~3.0 ms** | **~3.0 ms** | ~2.6× slower |
| `crossed_20k` | 272 ms | 109 ms | ~113 ms | **~52 ms** (cold `lmer`) | **~12–14 ms** (`fit_prepared`) | **~1.0×** (hot) |
| `nested_10k` | 53.8 ms | 17.0 ms | ~15.5 ms | **~16.5 ms** | **~14 ms** | ~2.4× slower |

Blocked Cholesky is enabled when intercept-only and every cross block passes `fits_blocked_gate` (≤100k elements). Cross blocks under that cap are **always stored dense** (see § reliability fix below); nested sparse crosses still use LDL.

## Blocked augmented Cholesky — what delivered the crossed speedup (2026-07-08)

The **~1.7×** drop on `crossed_20k` (113 ms → **~68 ms** median) came from adopting the core idea in [MixedModels.jl `updateL!`](https://github.com/JuliaStats/MixedModels.jl/blob/main/src/linearmixedmodel.jl) and [Bates et al. 2025](https://arxiv.org/html/2505.11674v1), implemented in [`src/intercept_blocked.rs`](src/intercept_blocked.rs).

**Old hot path (sparse LDL, still used for nested / large cross blocks):**

1. Build and factor a monolithic **q×q** matrix `A = ΛᵀZᵀZΛ + I` (q ≈ 350 for `crossed_20k`).
2. Run **q-dimensional solves** for `ΛZᵀy` and each column of `ΛZᵀX` every θ eval.
3. Finish with a small **p×p** profile solve for β.

**New hot path (blocked, for crossed intercept-only models):**

1. **Precompute** θ-free Gram blocks per RE term: diagonal `ZᵢᵀZᵢ`, dense cross `ZⱼᵀZᵢ`, and `Xy` couplings — once at `LmmData` setup.
2. **Sort RE blocks** by level count descending (largest factor first), matching MixedModels.
3. **Per θ:** scale blocks by Λ_θ, then **blocked Cholesky** on the augmented system (RE blocks + fixed/response block).
4. Read **profiled deviance** from the bottom-right of the factored `Xy` block (`r_yy`) — **no q solves, no explicit β** on the hot path.

**Why that is faster on crossed:** work scales with **per-term block sizes** (e.g. 250×100 cross + 100×100 filled second diagonal) instead of a single sparse **350×350** LDL plus **p** full-q backsolves per θ eval. On our fair harness that is roughly **~4×** faster than the prior crossed median and **~4×** the 2026-07-06 reference — still **~5×** behind MixedModels.jl on the same machine, but the dominant structural gap is much smaller.

**Gate:** `CrossBlock::fits_blocked_gate()` — crosses with ≤ **100 000** elements use the blocked path (`crossed_20k` qualifies). **Nested** sparse crosses (e.g. `batch/cask`) return `None` from `InterceptBlockedChol::try_new` and keep **reused sparse LDL** until column-disjoint block Cholesky is wired with matching Schur layout.

**Reliability (2026-07-08):** `from_submatrix` used to pick **sparse** storage when cross density ≤ 35%, but `fits_blocked_gate()` rejects sparse crosses — so borderline crossed fixtures could **intermittently** fall back to sparse LDL (~60 ms) instead of blocked (~14 ms). Cross blocks with `nrows × ncols ≤ 100_000` are now **always densified** at build time.

**Parity:** `tests/test_crossed_mock.rs` and `intercept_blocked` unit tests compare blocked deviance to `evaluate().reml_crit`; golden parity passes with the blocked path enabled for crossed fixtures.

## Blocked kernel tuning — GEMM, batched trisolve, prepared fit (2026-07-08, continued)

Second optimization pass on [`src/intercept_blocked.rs`](src/intercept_blocked.rs) plus API/setup changes in [`src/lib.rs`](src/lib.rs). **All of this is in the current tree** (not yet tagged in a release).

### Kernel changes (`intercept_blocked.rs`)

| Change | Effect |
|:-------|:-------|
| `general_mat_mul` into reusable `rank_gram_buf` | Rank updates `cross @ crossᵀ` without per-θ `dot()` allocation |
| GEMM Schur on RE crosses (`li @ ljᵀ`) | Replaces per-element row-dot loops in `schur_sub_re_cross` |
| GEMM Schur on `Xy` blocks (`lij @ ljjᵀ`, `beta=-1`) | `schur_sub_xy_re` ~4× faster (`blocked_schur_xy` ~1.7 ms → ~0.4 ms) |
| `ReLower::solve_multi_rhs` | Batched forward substitution for all cross columns at once |
| Dense `trisolve_full_lower` uses multi-RHS | ~250 column `solve_col` calls → one batched solve per cross block |
| `xy_trisolve_buf` + transpose | `trisolve_xy_re` solves all fixed/response rows in one batch |
| Packed-array `chol()` | Direct indexing on lower-triangular storage (no `get`/`set` in inner loops) |
| Always-dense cross ≤100k elements | Fixes intermittent `sparse_ldl` fallback (see above) |

### Setup amortization (`prepare_lmer` / `fit_prepared`)

Cold `lmer()` pays **setup every time** (~10–13 ms on `crossed_20k`: formula parse, Polars → design matrices, `LmmData::new_weighted`, blocked LDL cache build). Julia’s fair harness times `fit(MixedModel, …)` with model-matrix work **inside** `fit`, but Rust’s `lmer_setup` phase was an apples-to-oranges overhead when comparing wall times.

**API** (same numerics as `lmer` / `lmer_weighted`):

```rust
use lme_rs::{prepare_lmer, fit_prepared, lmer};

// One-shot (includes setup)
let fit = lmer(formula, &df, false)?;

// Amortized (CV, bootstrap, grid search on same data)
let prepared = prepare_lmer(formula, &df)?;
let fit = fit_prepared(&prepared, false)?;  // ~12–14 ms on crossed_20k
```

`LmerPrepared` exposes `blocked_kernel` and `blocked_kernel_detail` (`blocked_active`, `blocked_unavailable_*`) for diagnostics.

**Post-fit fix:** `fit_prepared` reuses `coefs.fitted` and `coefs.reml_crit` from a single `evaluate()` — removed duplicate `log_reml_deviance` and redundant `Z*b` assembly (~5 ms → ~2 ms post-fit).

### BLAS A/B (`ndarray` `blas` feature)

Tried enabling `ndarray = { features = ["blas"] }` so `general_mat_mul` routes to MKL/OpenBLAS (already linked via `ndarray-linalg` on x86_64).

| Config | `fit_prepared` median (`crossed_20k`) |
|:-------|--------------------------------------:|
| BLAS on | ~12–14 ms |
| BLAS off (default) | **~13.9 ms** (8-run avg) |

**Verdict:** no meaningful win at these block sizes (~100×250); **BLAS left disabled** to avoid extra linking surface. Rank/Schur GEMM still uses `ndarray`’s `matrixmultiply` backend.

### Recorded timings (`crossed_20k`, ML, blocked path, Windows AMD64, 2026-07-08)

| Metric | Rust | Julia (same fixture) |
|:-------|-----:|-------------------:|
| `fit_prepared` (hot) | **~12–14 ms** | ~14–16 ms |
| Cold `lmer()` | ~25–28 ms | — |
| `prepare_lmer` | ~10–13 ms | (inside Julia `fit`) |
| Mean deviance eval | **~0.23 ms** | ~0.24–0.25 ms |
| Eval count | 43 | 60 |
| `blocked_rank_chol` share of eval | ~68% | — |

**Takeaway:** once setup is amortized, Rust **matches Julia on crossed fit wall time** and is **at or below Julia per feval**. The remaining ~2× gap on cold `lmer()` is almost entirely **one-time setup + post-fit assembly**, not θ-search algebra.

`bench_perf_breakdown` now reports `prepare_wall_seconds`, `fit_prepared_wall_seconds`, `blocked_kernel`, and `blocked_kernel_detail` alongside the existing `LME_PERF_DIAG` phases.

## What did not work (do not reintroduce blindly)

| Attempt | Outcome | Lesson |
|:--------|:--------|:-------|
| Dense Cholesky for q ≤ 512 on hot path | Reverted | Wrong deviance + O(q³) rebuild cost; use O(nnz) assembly if revisited |
| Brent / golden-section 2D θ optimizer (wide brackets) | Reverted | Slower than Nelder–Mead + fast deviance; unreliable with broken dense path |
| Wide cyclic 2D golden-section (12×48 iters) | Reverted | ~1000+ evals; **~2.4 s/fit** on `crossed_20k` |
| 7×7 + 5×5 grid + coordinate polish (no NM) | Reverted for REML | OK for ML timing; **breaks** `penicillin_crossed_reml` golden parity |
| 6×6 grid + long NM polish (35 iters) | Reverted | **~200–260 ms** crossed — grid evals stack on top of NM |
| LDL cache on **evaluate** path | Reverted | Optimizer cache ≠ correct full profile solve |
| Two-block Schur LDL (smaller crossed block) | Removed | ~10 s/fit; golden failures |
| 2D golden-section with 40×60 cycles | Removed earlier | ~4.8k evals; large regression |
| Blocked path without cross-block size gate | Reverted | `nested_10k` regressed to **~1.8 s** (2000×200 dense cross); gate at 100k elements |
| `ndarray` BLAS for `general_mat_mul` (100×250 blocks) | **No win** | ~12–14 ms vs ~14 ms without BLAS; not enabled in `Cargo.toml` |
| 2D coordinate golden-section polish (2 cycles) | Reverted earlier | ~85 extra evals; crossed regressed to ~94 ms |

**Transient pitfall:** a broken dense hot path briefly showed **`random_intercept_10k` ~480 ms**. Current code is **~2.8 ms** — do not cite the 480 ms figure as a design regression.

## Invariants for contributors

1. **Golden parity** — `cargo test --release --test test_golden_parity` after any `src/math.rs` / optimizer change.
2. **Fast vs slow deviance** — `tests/test_crossed_mock.rs` compares `log_reml_deviance` to `evaluate().reml_crit` on a θ grid for crossed intercept-only models.
3. **Infeasible θ** — hot path returns `f64::MAX` on Cholesky / SPD failure; do not `.expect()` on the optimizer path.
4. **Fair harness** — after meaningful changes:
   ```powershell
   python scripts/run_fair_rust_julia_benchmark.py --implementations rust --cases crossed_20k,nested_10k,random_intercept_10k --repeats 10
   ```
5. **Bimodal `crossed_20k`** — older builds showed two clusters (~70 ms and ~100 ms) from OS scheduling or intermittent sparse-LDL fallback; the dense-cross reliability fix and `blocked_kernel` reporting in `bench_perf_breakdown` reduce confusion. Use medians over ≥10 repeats on a **fresh** `cargo build --release`.

## Performance diagnostics (`LME_PERF_DIAG`)

When optimizing crossed/nested fits, enable phase timing with **`LME_PERF_DIAG=1`** (zero overhead when unset).

**Quick run** (fair `crossed_20k` fixture):

```powershell
task benchmarks:perf-breakdown
# or
python scripts/run_perf_breakdown.py --cases crossed_20k,nested_10k
```

Runs **Rust** (`LME_PERF_DIAG=1` phase breakdown) and **Julia** (`m.optsum.feval`, optimizer, mean seconds/eval) on the same CSV, then writes `benchmark-results/perf-breakdown.json` with a `comparison` block (`rust_over_julia_feval`, `julia_over_rust_mean_eval`, …). Skips Julia when not installed.

**Caveat:** Rust `fit_wall_seconds` in `bench_perf_breakdown` is still end-to-end cold `lmer()` for backward compatibility. Use **`fit_prepared_wall_seconds`** for optimizer-only fairness vs Julia `fit(MixedModel, …)`. Compare eval counts and `mean_*_eval_seconds` for algebra fairness; use `prepare_wall_seconds` / `lmer_post_fit` for overhead.

**Example (`crossed_20k`, blocked path, 2026-07-08 workstation):**

| | Rust (`fit_prepared`) | Rust (cold `lmer`) | Julia (`fit` only) |
|:--|----------------------:|-------------------:|-------------------:|
| Eval count | 43 | 43 | 60 |
| Mean s/eval | **~0.23 ms** | ~0.23 ms | ~0.24 ms |
| Fit wall | **~13 ms** | ~26 ms | ~14 ms |
| Setup (`prepare` / `lmer_setup`) | (prior call ~11 ms) | ~11 ms | inside `fit` |
| Optimizer | 5×5 + 4×4 grid | same | NLopt `LN_NEWUOA` |

Rust matches Julia on **hot fit wall** and **per-eval** time; cold `lmer()` is ~2× Julia because of explicit setup + post-fit phases.

**What you get** — JSON with:

| Field / phase | Meaning |
|:--------------|:--------|
| `kernel` | `blocked`, `sparse_ldl`, or `dense_chol` deviance path |
| `kernel_detail` | `blocked_active` vs `blocked_unavailable` (sparse cross gate, etc.) |
| `deviance_eval_count` | `log_reml_deviance` calls (compare to Julia `optsum.feval`) |
| `mean_deviance_eval_seconds` | Per-eval wall time |
| `lmer_setup` | Formula parse + model-matrix build |
| `lmer_optimize` | θ grid / Nelder–Mead |
| `lmer_post_fit` | Full `evaluate()` + DataFrame assembly |
| `blocked_reset` | Scale / copy Gram blocks each θ |
| `blocked_rank_chol` | RE rank updates + Cholesky |
| `blocked_schur_re` / `blocked_schur_xy` | Schur complements |
| `blocked_trisolve` | Column triangular solves on cross / `Xy` |
| `deviance_sparse_ldl` | Nested path (when blocked gate is off) |

Phases include `fraction_of_deviance` (share of total eval time) and top-level `optimize_fraction_of_fit` / `deviance_fraction_of_fit` when `fit_wall_seconds` is set.

Implementation: [`src/perf_diag.rs`](src/perf_diag.rs); Rust runner [`comparisons/bench_perf_breakdown.rs`](comparisons/bench_perf_breakdown.rs); Julia runner [`comparisons/bench_fair_julia_perf.jl`](comparisons/bench_fair_julia_perf.jl).

## Closing the `crossed_20k` gap vs Julia (~14 ms)

**Target:** fair-harness `crossed_20k` ML median should approach MixedModels.jl (~14 ms on the 2026-07-06 reference).

**Status (2026-07-08, continued pass):** with `prepare_lmer` + `fit_prepared`, Rust is **~12–14 ms** — **parity with Julia** on hot fit wall time. Cold `lmer()` remains **~25–28 ms** (~1.8–2× Julia) because of one-time setup and post-fit work outside the optimizer.

| Milestone | `crossed_20k` median | vs Julia |
|:----------|---------------------:|---------:|
| 2026-07-06 reference | 272 ms | ~19× |
| Sparse LDL reuse | 109 ms | ~8× |
| Blocked Cholesky | ~68 ms | ~5× |
| Hot-path tuning (in-place Schur, scratch) | ~52 ms (cold `lmer`) | ~3.6× |
| GEMM + batched trisolve + prepared fit | **~13 ms** (`fit_prepared`) | **~1.0×** |

**Where time goes (Rust, blocked path, hot fit):**

| Component | Approx. share | Notes |
|:----------|:--------------|:------|
| θ optimizer evals | ~42 ML grid points | Julia uses more fevals (60) but similar per-feval cost |
| Per-θ `updateL!` | **~0.23 ms/eval** | `blocked_rank_chol` ~68%; Schur/trisolve reduced by GEMM + multi-RHS |
| One-time setup | ~10–13 ms | `prepare_lmer` only; not in `fit_prepared` |
| Post-fit | ~2 ms | Single `evaluate()`; DataFrame assembly |

**2026-07-08 hot-path passes** (`intercept_blocked.rs`):

*First pass (in-place Schur, scratch buffers, ML grid 5×5+4×4):*

- In-place Schur on `l_xy_re` / `l_xy_xy` (no per-θ `clone()` of cross / XY blocks).
- Fused `assign_scaled` for dense cross blocks (one pass vs `assign` + `scale`).
- Reused `diag_buf` / `cross_rhs_buf` / `schur_li_scratch` / `schur_lj_scratch`.
- ML grid tightened to **5×5 + 4×4** (~42 evals) after parity checks.

*Second pass (GEMM, batched trisolve, prepared fit):*

- `general_mat_mul` rank updates and Schur complements (no `dot()` alloc).
- `solve_multi_rhs` for dense cross trisolve and `trisolve_xy_re`.
- `prepare_lmer` / `fit_prepared` API; post-fit deduplication.
- Always-dense cross blocks ≤100k elements (reliability).
- BLAS feature tried and **not** enabled (no measurable win).

**Still to do for nested / general workflows:**

1. ~~Instrument `optsum.feval` (Julia) vs Rust eval count on the same fixture.~~ → `scripts/run_perf_breakdown.py`
2. Nested blocked path (sparse + column-disjoint Cholesky) — still on sparse LDL (~14–16 ms vs Julia ~7 ms).
3. Optional: speed up cold `lmer()` setup (design-matrix build, `zt_z` assembly) if single-shot API must match Julia.
4. Optional: deviance-only Criterion bench separate from full `lmer()` to isolate algebra from optimizer.

## Why MixedModels.jl is faster (and what to learn)

The fair harness times only `fit(MixedModel, ...)` after JIT warmup ([`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl)). On the 2026-07-06 reference, Julia led on all cases; **`crossed_20k` was ~5×** at the blocked-Cholesky median (~68 ms vs ~14 ms Julia). After the **prepared-fit** pass (2026-07-08), Rust **`fit_prepared` matches Julia** on the same fixture (~13 ms vs ~14 ms); cold `lmer()` is still ~2× because setup is timed separately.

MixedModels.jl (see [Bates et al., blocked Cholesky, 2025](https://arxiv.org/html/2505.11674v1)) differs from `lme-rs` in ways that explain the remaining gap:

| Aspect | MixedModels.jl | `lme-rs` today |
|:-------|:---------------|:---------------|
| Per-θ linear algebra | **Blocked in-place Cholesky** on an augmented `(q+p+1)` system (`updateL!`) | **Blocked path** on crossed intercept-only models; sparse LDL + β profile elsewhere |
| Intercept RE structure | **Diagonal / small dense blocks** per RE term; specialized `rmulΛ!` / `lmulΛ!` | Same block layout in `intercept_blocked.rs`; monolithic q LDL when gated off |
| Fill-in | RE term **ordering** to limit Cholesky fill | Largest RE term first; dense fill in second diagonal block |
| Profile likelihood | Deviance without `u`/`b` each θ | Blocked hot path matches; evaluate path still full profile |
| θ search | Tuned derivative-free optimizer (often **fewer** evals) | Grid (ML) or grid + short NM (REML) for |θ|=2 |
| Constant factors | OpenBLAS/MKL, JIT-specialized loops | Pure Rust blocked kernels; `sprs-ldl` fallback |

**How to read the ratios:**

- **Random intercept ~2.6×** across 10k obs → mostly **constant-factor** overhead (allocation, per-eval plumbing).
- **Crossed ~1.0×** on **`fit_prepared`** (2026-07-08) → blocked Cholesky + GEMM/trisolve tuning closed the kernel gap; cold `lmer()` ~2× is setup/post-fit accounting.
- **Nested ~2×** → still on sparse LDL; blocked nested path not yet wired.

## Next experiments (priority order)

1. **Nested blocked path (sparse + column-disjoint)** — scaffolding lives in [`src/intercept_blocked.rs`](src/intercept_blocked.rs) (`CrossBlock::Sparse`, `column_disjoint_partition`, `ReFactor::ColumnBlocks`). Lessons from 2026-07-08 WIP:
   - Densifying nested `batch×cask` (200×2000) and factoring a single 2000×2000 `ReLower` → **~1.8 s/fit**.
   - Ungated sparse blocked with dense Schur loops → **~66–130 ms** vs **~15 ms** reused LDL.
   - Transposing the cross alone is **not** enough: Schur/trisolve indexing must match the stored layout (MixedModels keeps consistent block orientation).
   - Target: cask×batch sparse cross + per-batch diagonal Cholesky blocks (~200×10), sparse Schur along structural nonzeros only.
2. **Cold `lmer()` setup** — `build_design_matrices` + `zt_z` assembly (~10 ms on `crossed_20k`); only matters for one-shot fits (use `prepare_lmer` when repeating).
3. **Fair harness reporting** — prefer `fit_prepared_wall_seconds` in regression JSON; keep cold `lmer` as a separate column.
4. **Fix dense backend** — O(nnz) `A` assembly if revisited for non-blocked cases.
5. **Criterion / fair JSON** — refresh reference after next tagged release.

## Key files

| File | Role |
|:-----|:-----|
| [`src/math.rs`](src/math.rs) | `LmmData`, `InterceptLdlCache`, `InterceptSparseLdl`, `profile_deviance_p2` |
| [`src/intercept_blocked.rs`](src/intercept_blocked.rs) | Blocked augmented Cholesky (`updateL!`) for intercept-only crossed models |
| [`src/optimizer.rs`](src/optimizer.rs) | `optimize_theta_lmm`, intercept golden-section (|θ|=1), 2D log-grid (|θ|=2) |
| [`src/lib.rs`](src/lib.rs) | `lmer`, `prepare_lmer`, `fit_prepared`, `LmerPrepared` |
| [`src/perf_diag.rs`](src/perf_diag.rs) | `LME_PERF_DIAG` phase timing |
| [`comparisons/bench_perf_breakdown.rs`](comparisons/bench_perf_breakdown.rs) | Prepared vs cold fit breakdown JSON |
| [`benches/bench_math.rs`](benches/bench_math.rs) | Criterion size sweeps |
| [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs) | Fair harness example binary |
| [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | Fair harness driver |
| [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | Fair harness driver |

## Related docs

- [BENCHMARKS.md](BENCHMARKS.md) — run benchmarks, reference JSON, result tables
- [AGENTS.md](AGENTS.md) — CI tiers and `benchmarks-fair-rust-julia` smoke
- [CONTRIBUTING.md](CONTRIBUTING.md) — run tests when changing fitting logic
