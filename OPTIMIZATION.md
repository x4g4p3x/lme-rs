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

Fair-harness snapshot (Windows AMD64, 10 repeats) — see [BENCHMARKS.md § 2026-07-07](BENCHMARKS.md#fair-rust-julia-2026-07-07-wip):

| Case | 2026-07-06 Rust | After LDL pass | After θ-search | After blocked Cholesky | vs Julia (2026-07-06) |
|:-----|----------------:|---------------:|---------------:|-----------------------:|----------------------:|
| `random_intercept_10k` | 3.16 ms | 2.78 ms | ~2.5 ms | **~3.0 ms** | ~2.6× slower |
| `crossed_20k` | 272 ms | 109 ms | ~113 ms | **~52 ms** | ~4.8× slower |
| `nested_10k` | 53.8 ms | 17.0 ms | ~15.5 ms | **~16.5 ms** | ~2.4× slower |

Blocked Cholesky is enabled when intercept-only and every cross block passes `fits_blocked_gate` (dense ≤100k elements; sparse crosses still use LDL).

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

**Gate:** `CrossBlock::fits_blocked_gate()` — dense crosses with ≤ **100 000** elements use the blocked path (`crossed_20k` qualifies). **Sparse** crosses (nested `batch/cask`) return `None` from `InterceptBlockedChol::try_new` and keep **reused sparse LDL** until column-disjoint block Cholesky is wired with matching Schur layout.

**Parity:** `tests/test_crossed_mock.rs` and `intercept_blocked` unit tests compare blocked deviance to `evaluate().reml_crit`; golden parity passes with the blocked path enabled for crossed fixtures.

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

**Transient pitfall:** a broken dense hot path briefly showed **`random_intercept_10k` ~480 ms**. Current code is **~2.8 ms** — do not cite the 480 ms figure as a design regression.

## Invariants for contributors

1. **Golden parity** — `cargo test --release --test test_golden_parity` after any `src/math.rs` / optimizer change.
2. **Fast vs slow deviance** — `tests/test_crossed_mock.rs` compares `log_reml_deviance` to `evaluate().reml_crit` on a θ grid for crossed intercept-only models.
3. **Infeasible θ** — hot path returns `f64::MAX` on Cholesky / SPD failure; do not `.expect()` on the optimizer path.
4. **Fair harness** — after meaningful changes:
   ```powershell
   python scripts/run_fair_rust_julia_benchmark.py --implementations rust --cases crossed_20k,nested_10k,random_intercept_10k --repeats 10
   ```
5. **Bimodal `crossed_20k`** — some Windows runs show two clusters (~70 ms and ~100 ms) from OS scheduling; use medians over ≥10 repeats.

## Performance diagnostics (`LME_PERF_DIAG`)

When optimizing crossed/nested fits, enable phase timing with **`LME_PERF_DIAG=1`** (zero overhead when unset).

**Quick run** (fair `crossed_20k` fixture):

```powershell
task benchmarks:perf-breakdown
# or
python scripts/run_perf_breakdown.py --cases crossed_20k,nested_10k
```

Runs **Rust** (`LME_PERF_DIAG=1` phase breakdown) and **Julia** (`m.optsum.feval`, optimizer, mean seconds/eval) on the same CSV, then writes `benchmark-results/perf-breakdown.json` with a `comparison` block (`rust_over_julia_feval`, `julia_over_rust_mean_eval`, …). Skips Julia when not installed.

**Caveat:** Rust `fit_wall_seconds` is end-to-end `lmer()` (setup + optimize + post-fit `evaluate()`). Julia `fit_wall_seconds` is `fit(MixedModel, …)` only (model matrix already built inside `fit`). Compare eval counts and `mean_*_eval_seconds` for optimizer fairness; use Rust `lmer_setup` / `lmer_post_fit` phases for overhead.

**Example (`crossed_20k`, 2026-07-08 workstation):**

| | Rust | Julia |
|:--|-----:|------:|
| Eval count | 43 | 60 |
| Mean s/eval | ~0.33 ms | ~0.22 ms |
| Fit wall | ~32 ms (full `lmer`) | ~13 ms (`fit` only) |
| Optimizer | 5×5 + 4×4 grid | NLopt `LN_NEWUOA` |

Julia uses **more** fevals on crossed but **less time per feval**; Rust’s remaining gap is mostly per-eval algebra (`blocked_rank_chol` ~76% of eval time), not eval budget.

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

**Target:** fair-harness `crossed_20k` ML median should approach MixedModels.jl (~14 ms on the 2026-07-06 reference). After the hot-path pass (2026-07-08) Rust is **~52 ms** (~**3.7×** vs Julia), down from **~68 ms** pre-pass.

**Where time goes (Rust, blocked path):**

| Component | Approx. share | Notes |
|:----------|:--------------|:------|
| θ optimizer evals | ~42 ML grid points | Was ~61 (6×6+5×5); Julia often **fewer** evals via NLopt |
| Per-θ `updateL!` | ~1–2 ms/eval pre-2026-07-08 | Dominated by **cloning** `l_xy_re` / `l_cross` in Schur steps (~200 KB/eval for 100×250 cross) |
| Constant factors | BLAS/MKL vs pure Rust | Shows up on random intercept (~2.6×) |

**2026-07-08 hot-path pass** (`intercept_blocked.rs`):

- In-place Schur on `l_xy_re` / `l_xy_xy` (no per-θ `clone()` of cross / XY blocks).
- Fused `assign_scaled` for dense cross blocks (one pass vs `assign` + `scale`).
- Reused `diag_buf` / `trisolve_buf` / `cross_rhs_buf` (no per-θ `Vec` allocs in trisolve).
- Reused `schur_li_scratch` / `schur_lj_scratch` for `schur_sub_re_cross` (was cloning the full cross block twice per θ).
- ML grid tightened to **5×5 + 4×4** (~42 evals) after parity checks.

**Still to do for Julia parity:**

1. ~~Instrument `optsum.feval` (Julia) vs Rust eval count on the same fixture.~~ → `scripts/run_perf_breakdown.py`
2. Dense specialized kernels for 100×250 cross rank-update / trisolve (or BLAS `syrk`/`trsm`).
3. 2D local search after coarse grid (golden-section on each θ) only if eval budget can drop **without** exceeding ~42 fixed grid evals — coordinate golden at 2 cycles added ~85 evals and regressed crossed to ~94 ms.
4. Optional: deviance-only Criterion bench separate from full `lmer()` to isolate algebra from optimizer.

## Why MixedModels.jl is faster (and what to learn)

The fair harness times only `fit(MixedModel, ...)` after JIT warmup ([`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl)). On the same machine, Julia still leads on all reference cases; **`crossed_20k` is ~5×** at the latest Rust median (~68 ms vs ~14 ms Julia, 2026-07-06 reference).

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

- **Random intercept ~2.6×** across 10k obs → mostly **constant-factor** overhead (BLAS, allocation, per-eval plumbing).
- **Crossed ~5×** (down from ~8× pre-blocked) → blocked Cholesky closed much of the structural gap; remainder is optimizer eval count, constant factors, and nested/large-cross cases still on LDL.

## Next experiments (priority order)

1. **Nested blocked path (sparse + column-disjoint)** — scaffolding lives in [`src/intercept_blocked.rs`](src/intercept_blocked.rs) (`CrossBlock::Sparse`, `column_disjoint_partition`, `ReFactor::ColumnBlocks`). Lessons from 2026-07-08 WIP:
   - Densifying nested `batch×cask` (200×2000) and factoring a single 2000×2000 `ReLower` → **~1.8 s/fit**.
   - Ungated sparse blocked with dense Schur loops → **~66–130 ms** vs **~15 ms** reused LDL.
   - Transposing the cross alone is **not** enough: Schur/trisolve indexing must match the stored layout (MixedModels keeps consistent block orientation).
   - Target: cask×batch sparse cross + per-batch diagonal Cholesky blocks (~200×10), sparse Schur along structural nonzeros only.
2. **θ eval instrumentation** — compare Julia `optsum.feval` vs Rust ~42 ML grid evals on `crossed_20k`; consider 2D golden-section polish if Julia uses fewer.
3. **Parallel multi-RHS** — incremental win on remaining LDL fallback path.
4. **Fix dense backend** — O(nnz) `A` assembly if revisited for non-blocked cases.
5. **Criterion / fair JSON** — refresh reference after next tagged release.

## Key files

| File | Role |
|:-----|:-----|
| [`src/math.rs`](src/math.rs) | `LmmData`, `InterceptLdlCache`, `InterceptSparseLdl`, `profile_deviance_p2` |
| [`src/intercept_blocked.rs`](src/intercept_blocked.rs) | Blocked augmented Cholesky (`updateL!`) for intercept-only crossed models |
| [`src/optimizer.rs`](src/optimizer.rs) | `optimize_theta_lmm`, intercept golden-section (|θ|=1), 2D log-grid (|θ|=2) |
| [`src/lib.rs`](src/lib.rs) | `Arc<LmmData>` wiring for optimize + evaluate |
| [`benches/bench_math.rs`](benches/bench_math.rs) | Criterion size sweeps |
| [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs) | Fair harness example binary |
| [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | Fair harness driver |

## Related docs

- [BENCHMARKS.md](BENCHMARKS.md) — run benchmarks, reference JSON, result tables
- [AGENTS.md](AGENTS.md) — CI tiers and `benchmarks-fair-rust-julia` smoke
- [CONTRIBUTING.md](CONTRIBUTING.md) — run tests when changing fitting logic
