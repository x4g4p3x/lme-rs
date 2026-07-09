# LMM fit optimization

Engineering notes for **LMM variance-component (θ) search** throughput.

| Doc | Purpose |
|:----|:--------|
| [BENCHMARKS.md](BENCHMARKS.md) | Harness methodology and versioned timings |
| [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) row 13 | Completion goals |

**Read this before changing** [`src/math.rs`](src/math.rs) intercept-only paths, **`SingleFactorSlopesCache`** (single-factor random slopes), or [`src/optimizer.rs`](src/optimizer.rs) θ search.

---

## At a glance (2026-07-09)

**Goal:** on the [fair Rust vs Julia harness](BENCHMARKS.md#fair-rust-vs-julia-reference-results), reach **within ~1.5× of MixedModels.jl** on tier-A `cold_fit` cases **without breaking** golden parity. (Legacy bar: **2×**, through 2026-07-08.)

**Hardest synthetic case:** `nested_10k` — borderline at the **1.5×** bar (~**1.51×**); then `crossed_20k` (~**1.29×**).

| Case | Status vs Julia (cold `lmer`) | Hot-path metric |
|:-----|:------------------------------|:----------------|
| `sleepstudy_reml` | **~0.8×** (Rust faster) | `fit_prepared` **~0.60 ms** (Julia ~0.81 ms) |
| `crossed_20k` | **~1.29×** | `fit_prepared` **~10.2 ms** (Julia ~15.0 ms) |
| `random_intercept_10k` | **~1.02×** | `fit_prepared` **~0.31 ms** |
| `nested_10k` | **~1.51×** (stretch) | `fit_prepared` **~3.7 ms** (Julia ~6.5 ms) |

Cold `lmer()` medians: synthetics in [2026-07-09 reference](benchmarks/fair-rust-julia-reference-2026-07-09.json); **`sleepstudy_reml`** in [2026-07-09 sleepstudy-slopes reference](benchmarks/fair-rust-julia-reference-2026-07-09-sleepstudy-slopes.json). Use **`prepare_lmer` + `fit_prepared`** when fitting the same formula repeatedly — hot fit **beats Julia** on all four tier-A LMM cases above.

---

## Architecture

θ optimization calls `LmmData::log_reml_deviance` many times; final `evaluate()` runs once at the converged θ.

### Deviance paths

| Path | Entry | Used for | Must |
|:-----|:------|:---------|:-----|
| **Optimizer hot** | `profile_deviance` → `profile_deviance_diagonal` → `InterceptLdlCache::profile_deviance` | θ search cost | Fast; return `f64::MAX` on infeasible θ (no panic) |
| **Blocked hot (crossed)** | `InterceptBlockedChol::profile_deviance` when cross blocks fit in memory | θ search on intercept-only crossed models | Same deviance as slow path; gated by cross-block size |
| **Random-slopes hot (single factor)** | `SingleFactorSlopesCache::profile_deviance` when `ZᵀZ` is block-diagonal (`k > 1`, one RE term) | θ search on e.g. `(Days \| Subject)` | Block-assembled `A = ΛᵀZᵀZΛ + I` + reused sparse LDL; deviance-only during Nelder–Mead |
| **Full profile** | `solve_profile` → blocked `solve_profile_blocked`, slopes cache, or cached sparse `InterceptLdlCache::solve_profile` / `solve_profile_general` | `evaluate()`, SEs, fitted values | Correct; blocked models reuse `updateL!` factor (no sparse LDL on post-fit) |

> **Invariant:** the θ optimizer hot path must not panic on infeasible θ. Post-fit `evaluate()` on blocked intercept-only models uses **`InterceptBlockedChol::solve_profile_blocked`** (reusing the same `updateL!` factor as θ search), with sparse LDL fallback only if the blocked backsolve fails — golden parity passes on all fixtures including `penicillin_crossed_reml`.

### Intercept-only fast path

Enabled when every RE block has **k = 1** (`intercept_only_re`).

Key pieces in [`src/math.rs`](src/math.rs):

| Piece | Role |
|:------|:-----|
| **`LmmData`** | Precomputed `zt_x`, `zt_y`, `zt_z`, `xt_x`, `xt_y`, `y_norm2`; `Arc` reuse in [`src/optimizer.rs`](src/optimizer.rs) |
| **`InterceptLdlCache`** | `Mutex` holding reused sparse LDL symbolic structure; numeric `update()` per θ |
| **`row_block`** | Maps each random-effect row to its variance-component index |
| **`profile_deviance_p2`** | Hand-unrolled 2×2 SPD solve when p = 2 (no LAPACK on hot path) |
| **`profile_deviance_p1`** | Hand-unrolled 1×1 β solve when p = 1 (random intercept, nested) |
| **`nz_theta_i` / `nz_theta_j`** | Precomputed θ block indices per `A` nonzero in `InterceptSparseLdl::factor_blocks` |

General random-slopes / nested correlated blocks still use `solve_profile_general` when the slopes cache is inactive (multiple RE terms or non-block-diagonal `ZᵀZ`).

### Single-factor random-slopes fast path (2026-07-09)

When `re_blocks.len() == 1`, `k > 1`, and `ZᵀZ` is block-diagonal across groups, [`finish_lmm_data`](src/math.rs) installs **`SingleFactorSlopesCache`**:

| Piece | Role |
|:------|:-----|
| **`s_blocks`** | Precomputed `k × k` diagonal blocks of `ZᵀZ` per group |
| **`a_indptr` / `a_indices` / `a_values`** | Fixed CSR layout for block-diagonal `A = ΛᵀZᵀZΛ + I` |
| **`full_ldl`** | Reused `sprs_ldl` numeric update + solve on full `q × q` system |
| **`profile_deviance`** | Deviance-only finish (no `u` / `b` / `l_x` on the θ hot path) |

θ search remains **Nelder–Mead** (same eval budget as before); the win is **cheaper deviance evals** (~sub-ms on `sleepstudy_reml` vs ~2.7 ms before). Do **not** use `ndarray_linalg` Cholesky for these block solves — it disagrees with `sprs_ldl` on the same `A` (verified in unit tests).

Fair harness: [sleepstudy_reml reference](benchmarks/fair-rust-julia-reference-2026-07-09-sleepstudy-slopes.json).

### θ search ([`src/optimizer.rs`](src/optimizer.rs))

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

---

## Progress timeline

Fair-harness medians on Windows AMD64 (10 repeats). Full tables: [BENCHMARKS.md § 2026-07-07](BENCHMARKS.md#fair-rust-julia-2026-07-07-wip).

### `crossed_20k` (ML)

| Milestone | Median | vs Julia (~14 ms) |
|:----------|-------:|------------------:|
| 2026-07-06 reference | 272 ms | ~19× |
| Sparse LDL reuse | 109 ms | ~8× |
| θ-search tuning (grid, p=2 hand solve) | ~113 ms | ~8× |
| Blocked augmented Cholesky | ~68 ms | ~5× |
| Hot-path tuning (in-place Schur, scratch) | ~52 ms (cold `lmer`) | ~3.6× |
| GEMM + batched trisolve + prepared fit | **~13 ms** (`fit_prepared`) | **~1.0×** |

### All reference cases (latest)

| Case | 2026-07-06 Rust | After GEMM + prepared fit | vs Julia |
|:-----|----------------:|--------------------------:|---------:|
| `random_intercept_10k` | 3.16 ms | **~3.0 ms** | ~2.6× slower |
| `crossed_20k` | 272 ms | **~12–14 ms** (`fit_prepared`) | **~1.0×** (hot) |
| `nested_10k` | 53.8 ms | **~14 ms** | ~2.4× slower |

### Changes that moved the needle

| Change | Cases helped | Notes |
|:-------|:-------------|:------|
| Cached `LmmData` in θ optimizer | All LMM | git `76fdb61`; [CHANGELOG.md](CHANGELOG.md) |
| Precomputed `Z^T X`, `Z^T y` | All LMM | Avoids repeated sparse×dense products per θ |
| Reused **sparse LDL** symbolic + `update()` | `crossed_20k`, `nested_10k` | 2026-07-07 pass |
| Deviance-only intercept path | Intercept-only θ search | Skips `u`, `b`, `v_cols` allocations |
| `profile_deviance_p1` / `p2` | Random intercept, nested, crossed | No LAPACK on 1×1 / 2×2 β finish |
| Golden-section (|θ| = 1) + 2D log-grid (|θ| = 2, ML) | Random intercept, crossed | ~42 fixed evals for crossed ML |
| **Blocked augmented Cholesky** | `crossed_20k` | MixedModels-style `updateL!`; no q solves per θ |
| **GEMM + batched trisolve** | `crossed_20k` | Per-eval ≈ Julia; see [Blocked kernel tuning](#blocked-kernel-tuning) |
| **`prepare_lmer` / `fit_prepared`** | Repeated fits | Amortizes setup; hot path ≈ Julia `fit` only |

Blocked Cholesky is enabled when intercept-only and every cross block passes `fits_blocked_gate` (≤100k elements). Cross blocks under that cap are **always stored dense**; nested sparse crosses still use LDL.

---

## Blocked augmented Cholesky

The **~1.7×** drop on `crossed_20k` (113 ms → ~68 ms) came from adopting the core idea in [MixedModels.jl `updateL!`](https://github.com/JuliaStats/MixedModels.jl/blob/main/src/linearmixedmodel.jl) and [Bates et al. 2025](https://arxiv.org/html/2505.11674v1), implemented in [`src/intercept_blocked.rs`](src/intercept_blocked.rs).

### Old vs new hot path

**Sparse LDL** (still used for nested / large cross blocks):

1. Build and factor a monolithic **q×q** matrix `A = ΛᵀZᵀZΛ + I` (q ≈ 350 for `crossed_20k`).
2. Run **q-dimensional solves** for `ΛZᵀy` and each column of `ΛZᵀX` every θ eval.
3. Finish with a small **p×p** profile solve for β.

**Blocked** (for crossed intercept-only models):

1. **Precompute** θ-free Gram blocks per RE term: diagonal `ZᵢᵀZᵢ`, dense cross `ZⱼᵀZᵢ`, and `Xy` couplings — once at `LmmData` setup.
2. **Sort RE blocks** by level count descending (largest factor first), matching MixedModels.
3. **Per θ:** scale blocks by Λ_θ, then **blocked Cholesky** on the augmented system (RE blocks + fixed/response block).
4. Read **profiled deviance** from the bottom-right of the factored `Xy` block (`r_yy`) — **no q solves, no explicit β** on the hot path.

**Why faster on crossed:** work scales with **per-term block sizes** (e.g. 250×100 cross + 100×100 filled second diagonal) instead of a single sparse **350×350** LDL plus **p** full-q backsolves per θ eval.

### Gate and reliability

| Rule | Detail |
|:-----|:-------|
| **Gate** | `CrossBlock::fits_blocked_gate()` — crosses with ≤ **100 000** elements use the blocked path (`crossed_20k` qualifies) |
| **Nested** | Sparse crosses (e.g. `batch/cask`) return `None` from `InterceptBlockedChol::try_new`; keep reused sparse LDL until column-disjoint block Cholesky matches Schur layout |
| **Reliability fix** | `from_submatrix` used to pick **sparse** storage when cross density ≤ 35%, but `fits_blocked_gate()` rejects sparse crosses — borderline fixtures could **intermittently** fall back to sparse LDL (~60 ms) instead of blocked (~14 ms). Cross blocks with `nrows × ncols ≤ 100_000` are now **always densified** at build time |

**Parity:** `tests/test_crossed_mock.rs` and `intercept_blocked` unit tests compare blocked deviance to `evaluate().reml_crit`; golden parity passes with the blocked path enabled for crossed fixtures.

---

## Blocked kernel tuning

Second optimization pass on [`src/intercept_blocked.rs`](src/intercept_blocked.rs) plus API/setup changes in [`src/lib.rs`](src/lib.rs).

### Kernel changes

| Change | Effect |
|:-------|:-------|
| `general_mat_mul` into reusable `rank_gram_buf` | Rank updates `cross @ crossᵀ` without per-θ `dot()` allocation |
| GEMM Schur on RE crosses (`li @ ljᵀ`) | Replaces per-element row-dot loops in `schur_sub_re_cross` |
| GEMM Schur on `Xy` blocks (`lij @ ljjᵀ`, `beta=-1`) | `schur_sub_xy_re` ~4× faster (`blocked_schur_xy` ~1.7 ms → ~0.4 ms) |
| `ReLower::solve_multi_rhs` | Batched forward substitution for all cross columns at once |
| Dense `trisolve_full_lower` uses multi-RHS | ~250 column `solve_col` calls → one batched solve per cross block |
| `xy_trisolve_buf` + transpose | `trisolve_xy_re` solves all fixed/response rows in one batch |
| Packed-array `chol()` | Direct indexing on lower-triangular storage (no `get`/`set` in inner loops) |
| Always-dense cross ≤100k elements | Fixes intermittent `sparse_ldl` fallback |

### Setup amortization (`prepare_lmer` / `fit_prepared`)

Cold `lmer()` pays **setup every time** (~10–13 ms on `crossed_20k`: formula parse, Polars → design matrices, `LmmData::new_weighted`, blocked LDL cache build). Julia's fair harness times `fit(MixedModel, …)` with model-matrix work **inside** `fit`, so Rust `lmer_setup` was apples-to-oranges overhead when comparing wall times.

```rust
use lme_rs::{prepare_lmer, fit_prepared, lmer};

// One-shot (includes setup)
let fit = lmer(formula, &df, false)?;

// Amortized (CV, bootstrap, grid search on same data)
let prepared = prepare_lmer(formula, &df)?;
let fit = fit_prepared(&prepared, false)?;  // ~12–14 ms on crossed_20k
```

`LmerPrepared` exposes `blocked_kernel` and `blocked_kernel_detail` (`blocked_active`, `blocked_unavailable_*`) for diagnostics.

For **grouped k-fold CV** on different train subsets (not the same data), use [`cv_grouped`](../src/cv.rs) — it splits by grouping unit, fits each train fold with `prepare_lmer` + `fit_prepared` (folds in parallel when `n_jobs > 1`), and assembles out-of-fold population predictions. See [GUIDE.md § Repeated fits and cross-validation](GUIDE.md#repeated-fits-and-cross-validation).

**Post-fit fix:** `fit_prepared` reuses `coefs.fitted` and `coefs.reml_crit` from a single `evaluate()` — removed duplicate `log_reml_deviance` and redundant `Z*b` assembly (~5 ms → ~2 ms post-fit).

### BLAS A/B (`ndarray` `blas` feature)

Tried enabling `ndarray = { features = ["blas"] }` so `general_mat_mul` routes to MKL/OpenBLAS (already linked via `ndarray-linalg` on x86_64).

| Config | `fit_prepared` median (`crossed_20k`) |
|:-------|--------------------------------------:|
| BLAS on | ~12–14 ms |
| BLAS off (default) | **~13.9 ms** (8-run avg) |

**Verdict:** no meaningful win at these block sizes (~100×250); **BLAS left disabled** to avoid extra linking surface. Rank/Schur GEMM still uses `ndarray`'s `matrixmultiply` backend.

### Hot-fit breakdown (`crossed_20k`, ML, blocked path)

| Metric | Rust | Julia |
|:-------|-----:|------:|
| `fit_prepared` (hot) | **~12–14 ms** | ~14–16 ms |
| Cold `lmer()` | ~25–28 ms | — |
| `prepare_lmer` | ~10–13 ms | (inside Julia `fit`) |
| Mean deviance eval | **~0.23 ms** | ~0.24–0.25 ms |
| Eval count | 43 | 60 |
| `blocked_rank_chol` share of eval | ~68% | — |

| Component | Approx. share | Notes |
|:----------|:--------------|:------|
| θ optimizer evals | ~42 ML grid points | Julia uses more fevals (60) but similar per-feval cost |
| Per-θ `updateL!` | **~0.23 ms/eval** | `blocked_rank_chol` ~68%; Schur/trisolve reduced by GEMM + multi-RHS |
| One-time setup | ~10–13 ms | `prepare_lmer` only; not in `fit_prepared` |
| Post-fit | ~2 ms | Single `evaluate()`; DataFrame assembly |

**Takeaway:** once setup is amortized, Rust **matches Julia on crossed fit wall time** and is **at or below Julia per feval**. The remaining ~2× gap on cold `lmer()` is almost entirely **one-time setup + post-fit assembly**, not θ-search algebra.

`bench_perf_breakdown` reports `prepare_wall_seconds`, `fit_prepared_wall_seconds`, `blocked_kernel`, and `blocked_kernel_detail` alongside `LME_PERF_DIAG` phases. Setup sub-phases: `setup_formula`, `setup_design_matrix`, `setup_lmm_data`.

---

## Setup and post-fit pass (2026-07-08 continued)

After blocked Cholesky matched Julia on `fit_prepared`, the remaining cold-`lmer()` gap on `crossed_20k` was **setup (~45%)** and **post-fit (~20%)**, not θ-search algebra. This pass targets both.

### Recorded (Windows AMD64, `rustc 1.96.0`, Julia 1.12.6, MixedModels.jl 5.7.0)

Fair harness: 2 warmups + 10 repeats (`scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia`).

| Case | Rust cold `lmer()` | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **22.4 ms** | 16.1 ms | **1.39×** |
| `nested_10k` | **12.2 ms** | 7.4 ms | 1.65× |
| `random_intercept_10k` | **1.6 ms** | 1.4 ms | **1.14×** |

`bench_perf_breakdown` on `crossed_20k` (single measured run):

| Phase | Before this pass | After |
|:------|----------------:|------:|
| `prepare_lmer` | ~10–13 ms | **~4.8 ms** |
| `fit_prepared` | ~12–14 ms | **~12.1 ms** |
| `lmer_post_fit` | ~4–5 ms | **~1.8 ms** |
| Cold `lmer()` | ~25–26 ms | **~19 ms** |

### Setup changes ([`src/model_matrix.rs`](src/model_matrix.rs), [`src/math.rs`](src/math.rs))

| Change | Effect |
|:-------|:-------|
| **Direct CSR `Zᵀ` build** (`build_zt_csr`) | Skips `TriMat` assembly + transpose for random-effects matrices |
| **Single-pass group indexing** (`HashMap<&str, usize>`) | Avoids per-row `String` allocation when building RE blocks |
| **Pre-sized triplet buffers** | `reserve(n_obs × k)` per intercept RE block |
| **Skip sparse LDL at prepare** when blocked Cholesky is active | No symbolic LDL factorization when `InterceptBlockedChol` handles θ search |
| **No `x`/`zt`/`y` clones** for unweighted `LmmData` | Drops duplicate storage of design matrices in `x_eff`/`zt_eff`/`y_eff` |
| **Row-wise `precompute_zt_products`** | CSR `outer_iterator`; hand-unrolled `p = 1` / `p = 2` |
| **Setup sub-phases** in `LME_PERF_DIAG` | `setup_formula`, `setup_design_matrix`, `setup_lmm_data` |

### Post-fit changes ([`src/math.rs`](src/math.rs))

| Change | Effect |
|:-------|:-------|
| **`InterceptLdlCache::solve_profile`** | Blocked models: `solve_profile_blocked` reuses `updateL!` factor; sparse LDL fallback if backsolve fails |
| **Blocked post-fit backsolve (2026-07-08)** | `backsolve_a_inv` + `solve_profile_blocked` wired into `solve_profile`; skips ~2 ms lazy sparse LDL init on `crossed_20k` |
| **Cached `y_norm2`** in profile finish | Drops redundant `y_eff` norm computation |

### Fair-harness Julia fix

[`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl): removed obsolete `ProgressMeter.enable(false)` (broken on ProgressMeter 1.11+) so Rust vs Julia comparisons run out of the box.

### What we tried and reverted

| Attempt | Outcome |
|:--------|:--------|
| Grouped `ZᵀZ` from per-observation RE indices (dense q×q) | Broke `penicillin_crossed_reml` golden parity; reverted. Nested q=2000 would allocate 4M f64 anyway. |
| Obs-major sparse `ZᵀZ` bucket accumulation | **Shipped** when `q ≥ 256` — matches sparse multiply (unit-tested); gated off tiny models to avoid O(n) bucket alloc on `random_intercept_10k`. |
| Blocked `solve_profile` via `l_xy_re` `w` extraction | Broke golden parity — post-factorization `l_xy_re` ≠ LDL `w` vectors; needs full `updateL!` backsolve. |
| Blocked `ranef!`/`fixef!` backsolve (2026-07-08, first attempt) | Early `backsolve_profile_w` disagreed with sparse LDL; **superseded** by `forward_solve_ld` / `backward_solve_ld` + `solve_profile_blocked` (golden parity OK). |
| `inv_from_chol_lower` in `evaluate()` for all `p` | Regressed `random_intercept_10k` ~0.3 ms vs `l_x.inv()` at `p = 2`; not committed (use only for `p > 2` if revisited). |

---

## Prepare ownership pass (2026-07-08 continued)

Second setup slice after the [setup/post-fit pass](#setup-and-post-fit-pass-2026-07-08-continued): eliminate duplicate design-matrix storage between `DesignMatrices` and `LmmData`.

### Recorded (Windows AMD64, `rustc 1.96.0`, Julia 1.12.6, MixedModels.jl 5.7.0)

Fair harness: 2 warmups + 10 repeats (`scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia`).

| Case | Rust cold `lmer()` | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **21.1 ms** | 15.7 ms | **1.34×** |
| `nested_10k` | **12.5 ms** | 6.9 ms | 1.81× |
| `random_intercept_10k` | **1.3 ms** | 1.7 ms | **Rust faster** |

### Changes ([`src/lib.rs`](src/lib.rs), [`src/model_matrix.rs`](src/model_matrix.rs))

| Change | Effect |
|:-------|:-------|
| **`mem::take` into `LmmData`** | `prepare_lmer` moves `x`, `zt`, `y`, `re_blocks` into `LmmData` instead of cloning; `fit_prepared` reads from `lmm` |
| **Offset path unchanged** | When an offset is present, `matrices.y` keeps the original response; `LmmData` stores the adjusted vector |
| **Categorical dummy encoding** | `HashMap<&str, usize>` level lookup + integer `level_id` per row — O(n_obs) string hashing instead of O(n_obs × n_levels) string compares |

### Simple fixed-effects fast path (2026-07-08 continued)

[`try_build_simple_x_matrix`](src/model_matrix.rs) bypasses the generic fiasto column loop for **`y ~ 1`**, **`y ~ x`**, and **`y ~ 1 + x`** (no interactions, no categorical fixed effects). Fair-harness fixtures and `penicillin_crossed_reml` (`diameter ~ 1 + …`) hit this path.

**Recorded** (same machine, 3 warmups + 20 repeats): `prepare_lmer` on `crossed_20k` **~4.2 ms** (was ~5–6 ms before); `random_intercept_10k` cold `lmer()` **~1.3 ms** (still **beats Julia** ~1.5 ms). Dropped a post-fit experiment (`inv_from_chol_lower` for `p = 2`) after it regressed `random_intercept_10k` ~0.3 ms in A/B — not committed.

---

## Prepare fast paths pass (2026-07-08 continued)

Third setup slice: target cases where Julia still leads on cold `lmer()` — **`nested_10k` prepare** and **`crossed_20k` `setup_lmm_data`** — without touching `random_intercept_10k` (already even or faster).

### Recorded (Windows AMD64, `rustc 1.96.0`, Julia 1.12.6, MixedModels.jl 5.7.0)

Fair harness: 3 warmups + 20 repeats (`scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia`).

| Case | Rust cold `lmer()` | Julia `fit` | vs Julia |
|:-----|-------------------:|------------:|---------:|
| `crossed_20k` | **25.1 ms** | 15.6 ms | 1.61× |
| `nested_10k` | **10.7 ms** | 7.0 ms | **1.53×** (was 1.57–1.81×) |
| `random_intercept_10k` | **1.2 ms** | 1.3 ms | **Rust faster** |

`bench_perf_breakdown` on the same machine:

| Case | `prepare_lmer` | `fit_prepared` Rust | Julia `fit` |
|:-----|-------------:|--------------------:|------------:|
| `nested_10k` | **~5.3 ms** (was ~7 ms) | **~4.0 ms** | ~6.6 ms |
| `crossed_20k` | **~6.4 ms** | **~13 ms** | ~12.4 ms |

**Diagnosis:** θ-search (`fit_prepared`) already beats or matches Julia on both Julia-lagging cases. Cold-`lmer()` gap on `crossed_20k` is now mostly **setup**, not post-fit — blocked `solve_profile_blocked` removed the ~2 ms lazy sparse LDL on `evaluate()`.

### Changes ([`src/math.rs`](src/math.rs), [`src/model_matrix.rs`](src/model_matrix.rs))

| Change | Effect |
|:-------|:-------|
| **`build_zt_z_intercept_from_zt`** | For intercept-only RE with `q ≥ 256`, accumulate `ZᵀZ` per observation (O(n·k²), k ≈ 2) instead of sparse `zt * ztᵀ`. Unit-tested against the generic multiply on crossed-style patterns. |
| **`q ≥ 256` gate** | Skips obs-major bucketing on tiny models (`random_intercept_10k`, q ≈ 100) where O(n) scratch alloc regressed cold `lmer()` ~0.3 ms. |
| **`try_build_interaction_groups`** | Nested `batch:cask` groups: index each factor column once, hash composite level tuples — no per-row `format!` + `join` on 10k observations. |

### What we did not change

- **`random_intercept_10k`** — explicitly out of scope; still beats Julia after the `q` gate.
- **Blocked post-fit backsolve** — **shipped (2026-07-08):** `solve_profile_blocked` wired into `InterceptLdlCache::solve_profile`; unit-tested vs sparse LDL on penicillin; golden parity passes.

---

## Performance diagnostics (`LME_PERF_DIAG`)

Enable phase timing with **`LME_PERF_DIAG=1`** (zero overhead when unset).

```powershell
task benchmarks:perf-breakdown
# or
python scripts/run_perf_breakdown.py --cases crossed_20k,nested_10k
```

Runs **Rust** (phase breakdown) and **Julia** (`m.optsum.feval`, optimizer, mean seconds/eval) on the same CSV, then writes `benchmark-results/perf-breakdown.json` with a `comparison` block. Skips Julia when not installed.

### Comparing Rust vs Julia fairly

| Metric | Use for |
|:-------|:--------|
| `fit_prepared_wall_seconds` | Optimizer-only fairness vs Julia `fit(MixedModel, …)` |
| `fit_wall_seconds` | End-to-end cold `lmer()` (backward compatible) |
| `mean_*_eval_seconds` + eval count | Algebra fairness |
| `prepare_wall_seconds` / `lmer_post_fit` | Overhead outside θ search |

### JSON fields

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

---

## Why MixedModels.jl is faster (and what to learn)

The fair harness times only `fit(MixedModel, ...)` after JIT warmup ([`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl)). After the **prepared-fit** pass (2026-07-08), Rust **`fit_prepared` matches Julia** on `crossed_20k` (~13 ms vs ~14 ms); cold `lmer()` is still ~2× because setup is timed separately.

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
- **Crossed ~1.0×** on **`fit_prepared`** → blocked Cholesky + GEMM/trisolve tuning closed the kernel gap; cold `lmer()` ~2× is setup/post-fit accounting.
- **Nested ~2×** → still on sparse LDL; blocked nested path not yet wired.

---

## What did not work

Do not reintroduce these without re-validating parity and benchmarks.

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
| `ndarray` BLAS for `general_mat_mul` (100×250 blocks) | No win | ~12–14 ms vs ~14 ms without BLAS; not enabled in `Cargo.toml` |
| 2D coordinate golden-section polish (2 cycles) | Reverted earlier | ~85 extra evals; crossed regressed to ~94 ms |

> **Transient pitfall:** a broken dense hot path briefly showed **`random_intercept_10k` ~480 ms**. Current code is **~2.8 ms** — do not cite the 480 ms figure as a design regression.

---

## Contributor checklist

1. **Golden parity** — `cargo test --release --test test_golden_parity` after any `src/math.rs` / optimizer change (including `SingleFactorSlopesCache`).
2. **Fast vs slow deviance** — `tests/test_crossed_mock.rs` compares `log_reml_deviance` to `evaluate().reml_crit` on a θ grid for crossed intercept-only models.
3. **Infeasible θ** — hot path returns `f64::MAX` on Cholesky / SPD failure; do not `.expect()` on the optimizer path.
4. **Fair harness** — after meaningful changes:
   ```powershell
   python scripts/run_fair_rust_julia_benchmark.py --implementations rust --cases crossed_20k,nested_10k,random_intercept_10k --repeats 10
   ```
5. **Bimodal `crossed_20k`** — older builds showed two clusters (~70 ms and ~100 ms) from OS scheduling or intermittent sparse-LDL fallback. The dense-cross reliability fix and `blocked_kernel` reporting in `bench_perf_breakdown` reduce confusion. Use medians over ≥10 repeats on a **fresh** `cargo build --release`.

---

## Next experiments (priority order)

1. ~~**`sleepstudy_reml` random-slopes throughput**~~ — **done (2026-07-09):** `SingleFactorSlopesCache`; fair harness **~0.8×** Julia cold `lmer()`, **~0.74×** `fit_prepared`. See [BENCHMARKS.md § 2026-07-09 random slopes](BENCHMARKS.md#fair-rust-julia-2026-07-09-random-slopes).
2. **Nested blocked path (row-grouped ColumnBlocks)** — **partial (2026-07-08):** nested `batch/cask` now uses `ReFactor::Diagonal` + `trisolve_single_row_cols` on the batch block (`columns_single_row` sparse gate). Fair harness: `nested_10k` cold **~1.7×** Julia, **`fit_prepared` ~0.56×** Julia. Remaining: row-grouped **10×10 cask** ColumnBlocks on block 0 if `prepare_lmer` needs it.
3. ~~**Blocked post-fit backsolve**~~ — **done (2026-07-08):** `solve_profile_blocked` matches sparse LDL; wired into `InterceptLdlCache::solve_profile`.
4. ~~**Large random-intercept setup**~~ — **done (2026-07-09):** direct single-factor `Zᵀ` CSR, diagonal single-membership `ZᵀZ`, and no-op cast avoidance make cold 50k/100k fits **~0.47× / ~0.51×** Julia; see [setup reference](benchmarks/fair-rust-julia-reference-2026-07-09-large-intercept-setup.json).
5. **Post-fit SEs** — `inv_lx` in `evaluate()` still allocates; Cholesky backsolve for `beta_se` would shave the last ~2 ms on crossed.
6. ~~**Fair harness reference JSON**~~ — **refreshed (2026-07-09):** [full LMM snapshot](benchmarks/fair-rust-julia-reference-2026-07-09-full-lmm.json) re-ran the real fixtures plus 10k/50k/100k intercept, crossed, and nested cases at HEAD; [GLMM snapshot](benchmarks/fair-rust-julia-reference-2026-07-09-glmm.json) covers CBPP and grouseticks.
7. **Fix dense backend** — O(nnz) `A` assembly if revisited for non-blocked cases.

---

## Key files

| File | Role |
|:-----|:-----|
| [`src/math.rs`](src/math.rs) | `LmmData`, `InterceptLdlCache`, `SingleFactorSlopesCache`, `InterceptSparseLdl`, `profile_deviance_p2` |
| [`src/intercept_blocked.rs`](src/intercept_blocked.rs) | Blocked augmented Cholesky (`updateL!`) for intercept-only crossed models |
| [`src/optimizer.rs`](src/optimizer.rs) | `optimize_theta_lmm`, intercept golden-section (|θ|=1), 2D log-grid (|θ|=2) |
| [`src/lib.rs`](src/lib.rs) | `lmer`, `prepare_lmer`, `fit_prepared`, `LmerPrepared` |
| [`src/model_matrix.rs`](src/model_matrix.rs) | Design matrices; `build_zt_csr` direct `Zᵀ` assembly |
| [`src/perf_diag.rs`](src/perf_diag.rs) | `LME_PERF_DIAG` phase timing |
| [`comparisons/bench_perf_breakdown.rs`](comparisons/bench_perf_breakdown.rs) | Prepared vs cold fit breakdown JSON |
| [`benches/bench_math.rs`](benches/bench_math.rs) | Criterion size sweeps |
| [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs) | Fair harness example binary |
| [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | Fair harness driver |
| [`scripts/run_perf_breakdown.py`](scripts/run_perf_breakdown.py) | Perf breakdown driver |

## Related docs

- [BENCHMARKS.md](BENCHMARKS.md) — run benchmarks, reference JSON, result tables
- [AGENTS.md](AGENTS.md) — CI tiers and `benchmarks-fair-rust-julia` smoke
- [CONTRIBUTING.md](CONTRIBUTING.md) — run tests when changing fitting logic
