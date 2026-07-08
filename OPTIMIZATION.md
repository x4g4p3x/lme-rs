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
| 2 | 6×6 global log-grid + local log-grid; REML adds short Nelder–Mead polish | ML ~62; REML ~54 grid + ≤20 NM iters |
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
| 2D log-grid θ search (|θ| = 2, ML) | `crossed_20k` | ~62 fixed evals; on par with prior NM median |
| **Blocked augmented Cholesky** (`intercept_blocked.rs`) | `crossed_20k` (cross block ≤100k) | MixedModels-style `updateL!`; no q solves per θ |

Fair-harness snapshot (Windows AMD64, 10 repeats) — see [BENCHMARKS.md § 2026-07-07](BENCHMARKS.md#fair-rust-julia-2026-07-07-wip):

| Case | 2026-07-06 Rust | After LDL pass | After θ-search | After blocked Cholesky | vs Julia (2026-07-06) |
|:-----|----------------:|---------------:|---------------:|-----------------------:|----------------------:|
| `random_intercept_10k` | 3.16 ms | 2.78 ms | ~2.5 ms | **~3.0 ms** | ~2.6× slower |
| `crossed_20k` | 272 ms | 109 ms | ~113 ms | **~68 ms** | ~4.8× slower |
| `nested_10k` | 53.8 ms | 17.0 ms | ~15.5 ms | **~16.5 ms** | ~2.4× slower |

Blocked Cholesky is enabled when intercept-only and every cross block has ≤100k dense elements (crossed fits; nested `batch/cask` falls back to sparse LDL).

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

**Gate:** `blocked_cross_fits()` requires every off-diagonal cross block to have ≤ **100 000** dense elements. `crossed_20k` (250×100) qualifies; `nested_10k` (`batch/cask`, 2000×200) does **not** and correctly keeps sparse LDL.

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
5. **Bimodal `crossed_20k`** — with blocked Cholesky, samples on one workstation cluster near **~68 ms** median; re-run the harness after changes before citing a single number.

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

1. **Sparse cross blocks in blocked path** — nested `batch/cask` (2000×200) should not fall back to full q LDL; use sparse off-diagonals like MixedModels.
2. **θ eval instrumentation** — compare Julia `optsum.feval` vs Rust eval counts on `crossed_20k`.
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
