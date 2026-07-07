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
| **Optimizer hot** | `profile_deviance` → `profile_deviance_diagonal` → `InterceptLdlCache::profile_deviance` | Nelder–Mead cost | Fast; return `f64::MAX` on infeasible θ (no panic) |
| **Full profile** | `solve_profile` → `solve_profile_diagonal` | `evaluate()`, SEs, fitted values | Correct; fresh LDL per θ |

**Invariant:** do **not** route `solve_profile_diagonal` through `InterceptLdlCache` until parity is proven on all golden intercept-only cases. A cached evaluate path broke `crossed_20k` and golden parity during development.

### Intercept-only fast path (when it applies)

Enabled when every RE block has **k = 1** (`intercept_only_re`).

Key pieces in [`src/math.rs`](src/math.rs):

- **`LmmData`** — precomputed `zt_x`, `zt_y`, `zt_z`, `xt_x`, `xt_y`, `y_norm2`; `Arc` reuse in [`src/optimizer.rs`](src/optimizer.rs).
- **`InterceptLdlCache`** — `Mutex` holding reused sparse LDL symbolic structure; numeric `update()` per θ.
- **`row_block`** — maps each random-effect row to its variance-component index (block scaling instead of full `d` vector on the hot path).
- **`profile_deviance_p2`** — hand-unrolled 2×2 SPD solve for fixed effects when p = 2 (avoids LAPACK on the hot path).

General random-slopes / nested correlated blocks still use `solve_profile_general` (no intercept fast path).

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

Fair-harness snapshot (Windows AMD64, 10 repeats) — full tables in [BENCHMARKS.md § 2026-07-07](BENCHMARKS.md#fair-rust-julia-2026-07-07-wip):

| Case | 2026-07-06 Rust median | 2026-07-07 Rust median | vs Julia (2026-07-06) |
|:-----|-----------------------:|----------------:|----------------------:|
| `random_intercept_10k` | 3.16 ms | 2.78 ms | ~2.4× slower |
| `crossed_20k` | 272 ms | 109 ms | ~7.6× slower |
| `nested_10k` | 53.8 ms | 17.0 ms | ~2.5× slower |

## What did not work (do not reintroduce blindly)

| Attempt | Outcome | Lesson |
|:--------|:--------|:-------|
| Dense Cholesky for q ≤ 512 on hot path | Reverted | Wrong deviance + O(q³) rebuild cost; use O(nnz) assembly if revisited |
| Brent / golden-section 2D θ optimizer | Reverted | Slower than Nelder–Mead + fast deviance; unreliable with broken dense path |
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
5. **Bimodal `crossed_20k`** — samples cluster ~107 ms vs ~190 ms on one workstation; investigate eval count / CPU freq before claiming a single median.

## Next experiments (priority order)

1. **2D profile optimizer** for |θ| = 2 — fewer evaluations than Nelder–Mead; must match converged θ vs golden / slow path.
2. **Fix dense backend** — O(nnz) `A` assembly (precomputed triplets), match LDL log-det; benchmark only if parity grid passes.
3. **Parallel multi-RHS solves** after `factor_blocks` — plate + sample columns for p = 2.
4. **Nested q ≫ 350** — structure-aware path without per-θ full symbolic LDL (nested already ~17 ms vs Julia ~6.9 ms on the reference run).
5. **Criterion / fair JSON** — refresh [benchmarks/fair-rust-julia-reference-2026-07-06.json](benchmarks/fair-rust-julia-reference-2026-07-06.json) successor after the next tagged release.

## Key files

| File | Role |
|:-----|:-----|
| [`src/math.rs`](src/math.rs) | `LmmData`, `InterceptLdlCache`, `InterceptSparseLdl`, `profile_deviance_p2` |
| [`src/optimizer.rs`](src/optimizer.rs) | `optimize_theta_lmm`, Nelder–Mead on `log_reml_deviance` |
| [`src/lib.rs`](src/lib.rs) | `Arc<LmmData>` wiring for optimize + evaluate |
| [`benches/bench_math.rs`](benches/bench_math.rs) | Criterion size sweeps |
| [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs) | Fair harness example binary |
| [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) | Fair harness driver |

## Related docs

- [BENCHMARKS.md](BENCHMARKS.md) — run benchmarks, reference JSON, result tables
- [AGENTS.md](AGENTS.md) — CI tiers and `benchmarks-fair-rust-julia` smoke
- [CONTRIBUTING.md](CONTRIBUTING.md) — run tests when changing fitting logic
