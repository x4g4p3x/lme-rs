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

Fair-harness snapshot (Windows AMD64, 10 repeats) — see [BENCHMARKS.md § 2026-07-07](BENCHMARKS.md#fair-rust-julia-2026-07-07-wip):

| Case | 2026-07-06 Rust | After LDL pass | After θ-search pass | vs Julia (2026-07-06) |
|:-----|----------------:|---------------:|--------------------:|----------------------:|
| `random_intercept_10k` | 3.16 ms | 2.78 ms | **~2.5 ms** | ~2.3× slower |
| `crossed_20k` | 272 ms | 109 ms | **~113 ms** | ~7.9× slower |
| `nested_10k` | 53.8 ms | 17.0 ms | **~15.5 ms** | ~2.3× slower |

The θ-search pass did **not** materially beat the LDL-only crossed median; it stabilized ML eval count and improved nested / random-intercept slightly.

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

**Transient pitfall:** a broken dense hot path briefly showed **`random_intercept_10k` ~480 ms**. Current code is **~2.8 ms** — do not cite the 480 ms figure as a design regression.

## Invariants for contributors

1. **Golden parity** — `cargo test --release --test test_golden_parity` after any `src/math.rs` / optimizer change.
2. **Fast vs slow deviance** — `tests/test_crossed_mock.rs` compares `log_reml_deviance` to `evaluate().reml_crit` on a θ grid for crossed intercept-only models.
3. **Infeasible θ** — hot path returns `f64::MAX` on Cholesky / SPD failure; do not `.expect()` on the optimizer path.
4. **Fair harness** — after meaningful changes:
   ```powershell
   python scripts/run_fair_rust_julia_benchmark.py --implementations rust --cases crossed_20k,nested_10k,random_intercept_10k --repeats 10
   ```
5. **Bimodal `crossed_20k`** — samples still cluster fast (~103–115 ms) vs slow (~175–195 ms) on one workstation; investigate eval count / CPU freq before claiming a single median.

## Why MixedModels.jl is faster (and what to learn)

The fair harness times only `fit(MixedModel, ...)` after JIT warmup ([`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl)). On the same machine, Julia leads on all reference cases; **`crossed_20k` is still ~8×** at the latest Rust median (~113 ms vs ~14 ms Julia, 2026-07-06 reference).

MixedModels.jl (see [Bates et al., blocked Cholesky, 2025](https://arxiv.org/html/2505.11674v1)) differs from `lme-rs` in ways that explain the gap:

| Aspect | MixedModels.jl | `lme-rs` today |
|:-------|:---------------|:---------------|
| Per-θ linear algebra | **Blocked in-place Cholesky** on an augmented `(q+p+1)` system (`updateL!`) | Sparse LDL on `q×q` `A = ΛᵀZᵀZΛ + I`, then profile β (`p=2` hand-unrolled) |
| Intercept RE structure | **Diagonal / small dense blocks** per RE term; specialized `rmulΛ!` / `lmulΛ!` | Monolithic `q`-dimensional sparse matrix for crossed |
| Fill-in | RE term **ordering** to limit Cholesky fill | Generic sparse pattern from `ZᵀZ` |
| Profile likelihood | Deviance without `u`/`b` each θ (same idea as our hot path) | Deviance-only hot path since 2026-07-07 |
| θ search | Tuned derivative-free optimizer (often **fewer** evals) | Grid (ML) or grid + short NM (REML) for |θ|=2 |
| Constant factors | OpenBLAS/MKL, JIT-specialized loops | `sprs-ldl`, `Mutex` cache |

**How to read the ratios:**

- **Random intercept ~2.3×** across 10k–100k obs → mostly **constant-factor** overhead (BLAS, allocation, per-eval plumbing), not a missing O(n) algorithm.
- **Crossed ~8×** → **structural**: a generic `q≈350` sparse LDL per θ eval vs a **blocked, intercept-specialized** factorization. Further θ-grid tuning alone is unlikely to close this to ~2×.

**Actionable lessons (priority):**

1. **Blocked augmented Cholesky** — adopt the Bates / MixedModels worldview (per-RE blocks, in-place `L` update) instead of only tuning θ search.
2. **Instrument eval counts** — compare Julia `optsum.feval` (or equivalent) vs `OptimizeResult.iterations` on the same fixture before more optimizer work.
3. **Exploit crossed structure** — two intercept RE terms should not route through one undifferentiated `q×q` LDL if a two-block bordered system can be factored with less fill.
4. **Ordering** — compare nnz of our `L` factor vs MixedModels on `crossed_20k`.

## Next experiments (priority order)

1. **Blocked augmented Cholesky** for intercept-only models — primary path to close the crossed gap; see MixedModels / Bates 2025 above.
2. **θ eval instrumentation** — log Rust vs Julia eval counts on `crossed_20k` to separate optimizer vs per-eval cost.
3. **Parallel multi-RHS solves** after `factor_blocks` — plate + sample columns for p = 2 (incremental).
4. **Fix dense backend** — O(nnz) `A` assembly (precomputed triplets), match LDL log-det; benchmark only if parity grid passes.
5. **Criterion / fair JSON** — refresh [benchmarks/fair-rust-julia-reference-2026-07-06.json](benchmarks/fair-rust-julia-reference-2026-07-06.json) successor after the next tagged release.

## Key files

| File | Role |
|:-----|:-----|
| [`src/math.rs`](src/math.rs) | `LmmData`, `InterceptLdlCache`, `InterceptSparseLdl`, `profile_deviance_p2` |
| [`src/optimizer.rs`](src/optimizer.rs) | `optimize_theta_lmm`, intercept golden-section (|θ|=1), 2D log-grid (|θ|=2) |
| [`src/lib.rs`](src/lib.rs) | `Arc<LmmData>` wiring for optimize + evaluate |
| [`benches/bench_math.rs`](benches/bench_math.rs) | Criterion size sweeps |
| [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs) | Fair harness example binary |
| [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | Fair harness driver |

## Related docs

- [BENCHMARKS.md](BENCHMARKS.md) — run benchmarks, reference JSON, result tables
- [AGENTS.md](AGENTS.md) — CI tiers and `benchmarks-fair-rust-julia` smoke
- [CONTRIBUTING.md](CONTRIBUTING.md) — run tests when changing fitting logic
