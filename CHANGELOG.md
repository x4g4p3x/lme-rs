# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Fair-harness **axis (3) cold-fit target** tightened from **2×** to **1.5×** Julia median on `cold_fit` (default `--target-ratio 1.5` in [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py)); synthetic tier-A medians are now ~1.0–1.5×. See [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md).
- LMM **`prepare_lmer` fast path** for fair-harness formulas (`y ~ x + (1 | g)`, numeric `x`): [`try_build_fair_lmm_design`](src/model_matrix.rs) skips fiasto; categorical fixed effects and nested `/` syntax fall back to generic build. **Lazy blocked Cholesky init** defers `InterceptBlockedChol::try_new` to first deviance/solve. Blocked gate/backsolve use global `zt` row indexing when sort order ≠ packed layout. Fair harness: `prepare_lmer` **~14% faster** on `crossed_20k` vs [2026-07-08 reference](benchmarks/fair-rust-julia-reference-2026-07-08.json); cold `lmer()` and `fit_prepared` within noise. New reference: [benchmarks/fair-rust-julia-reference-2026-07-09.json](benchmarks/fair-rust-julia-reference-2026-07-09.json).
- LMM **nested blocked path**: nested `batch/cask` sparse crosses use `ReFactor::Diagonal` and `trisolve_single_row_cols` on the batch block (`columns_single_row` gate) instead of densifying the cross or taking the sparse-LDL-only path. Blocked kernel active on `nested_10k`; fair harness **~1.4× Julia** cold `lmer()`, **`fit_prepared` ~0.52×** Julia. See [OPTIMIZATION.md](OPTIMIZATION.md).
- LMM **blocked post-fit backsolve**: `solve_profile_blocked` reuses the blocked `updateL!` factor in `evaluate()` (no lazy sparse LDL init on success). When blocked `w` vectors are numerically unstable (observed on `crossed_20k` ML), `solve_profile_finish` returns an error and **`InterceptLdlCache::solve_profile` falls back to sparse LDL** instead of panicking. Fair harness tier-A cases are **within 2× Julia** on cold `lmer()`; `crossed_20k` **~1.2×**, `fit_prepared` **~0.71×** Julia. Reference: [benchmarks/fair-rust-julia-reference-2026-07-08.json](benchmarks/fair-rust-julia-reference-2026-07-08.json).

### Added

- [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md) — sensor calibration workflow guide (independent batch NLS vs pooled `nlmer` `SSpower`; CUDA / [lightcurve-fitting](https://github.com/boom-astro/lightcurve-fitting) scope); CPU batch demo [`examples/batch_sspower_cpu.rs`](examples/batch_sspower_cpu.rs).
- `nlmer` built-in mean **`SSpower`** (`a * x^b + c`, MATLAB Curve Fitter `power2`) with `selfStart` heuristics for grouped calibration-style fits; lme4 golden parity (`sspower_synthetic_self_start`) via custom R `selfStart`; tests in [`tests/test_nlmm_sspower.rs`](tests/test_nlmm_sspower.rs); reference script [`comparisons/nlmm_sspower.R`](comparisons/nlmm_sspower.R).
- Fair-harness reference snapshot [benchmarks/fair-rust-julia-reference-2026-07-09.json](benchmarks/fair-rust-julia-reference-2026-07-09.json) (tier-A LMM cases after prepare fast path; Rust and Julia medians measured together). Documented in [BENCHMARKS.md § 2026-07-09 prepare fast path](BENCHMARKS.md#fair-rust-julia-2026-07-09-prepare-fast-path).
- Fair-harness reference snapshot [benchmarks/fair-rust-julia-reference-2026-07-08.json](benchmarks/fair-rust-julia-reference-2026-07-08.json) (tier-A LMM cases: `crossed_20k`, `nested_10k`, `random_intercept_10k`; Rust and Julia medians measured together). Documented in [BENCHMARKS.md § 2026-07-08 nested/post-fit](BENCHMARKS.md#fair-rust-julia-2026-07-08-nested-postfit).
- Unit tests: `blocked_profile_solve_matches_sparse_on_crossed_ml_grid`, `pastes_categorical_fixed_effect_uses_generic_build`, `nested_slash_formula_skips_fast_path`, `test_fair_lmm_design_matches_generic_for_crossed_formula`; `nested_sparse_gate_uses_diagonal_batch_factor` in [`src/intercept_blocked.rs`](src/intercept_blocked.rs).

## [0.1.9] - 2026-07-08

### Changed

- Python CI and local dev use **uv-native** [`python/uv.lock`](python/uv.lock) (`uv sync --extra dev`) instead of `requirements-ci.txt`.

### Added

- Python bindings accept **pandas `DataFrame`** and **PyArrow `Table`** inputs (converted to Polars at the FFI boundary); Polars remains the only runtime dependency.
- **[BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md)** — maps tier-A (fair MixedModels.jl) vs Rust-only benchmarks; axis (3) thresholds.
- Fair harness: real fixtures (`penicillin`, `pastes`, weighted `sleepstudy`), GLMM (`cbpp`, `grouseticks`), Rust `prepare_lmer`/`fit_prepared` phases (`--with-phases`), multi-metric JSON comparisons.
- Python [`nlmer_with_mean`](python/src/lib.rs) for user-defined nonlinear means; Rust [`parse_nlmer_custom_formula`](src/nlmm/formula.rs).
- **`prepare_lmer` / `fit_prepared` / `LmerPrepared`** — amortize formula parse and design-matrix build when fitting the same formula and data repeatedly (e.g. CV, bootstrap); hot `fit_prepared` wall time on fair `crossed_20k` matches MixedModels.jl (~12–14 ms). See [`OPTIMIZATION.md`](OPTIMIZATION.md).
- **Blocked augmented Cholesky** for intercept-only crossed models ([`src/intercept_blocked.rs`](src/intercept_blocked.rs)): MixedModels.jl-style `updateL!` layout; profile deviance without full-q LDL solves on the θ hot path.
- **LMM performance diagnostics** (`LME_PERF_DIAG=1`): phase timing in [`src/perf_diag.rs`](src/perf_diag.rs), [`comparisons/bench_perf_breakdown.rs`](comparisons/bench_perf_breakdown.rs), Julia runner [`comparisons/bench_fair_julia_perf.jl`](comparisons/bench_fair_julia_perf.jl), and [`scripts/run_perf_breakdown.py`](scripts/run_perf_breakdown.py) (`task benchmarks:perf-breakdown`). Reports `prepare_wall_seconds`, `fit_prepared_wall_seconds`, and `blocked_kernel` alongside optimizer phases.
- Fair Rust vs Julia LMM fit benchmark ([`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py), [`comparisons/bench_fair_rust_julia.rs`](comparisons/bench_fair_rust_julia.rs), [`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl)); documented 2026-07-04 Windows reference in [`BENCHMARKS.md`](BENCHMARKS.md) and [`benchmarks/fair-rust-julia-reference-2026-07-04.json`](benchmarks/fair-rust-julia-reference-2026-07-04.json).
- `nlmer` built-in means **`SSmicmen`** and **`SSgompertz`** (`stats::SSmicmen`; `stats::SSgompertz` as `Asym * exp(-b2 * b3^x)`).
- Custom nonlinear means via [`NlmmMeanEval`](src/nlmm/mean_fn.rs) / [`CustomNlmmMean`](src/nlmm/mean_fn.rs) and [`nlmer_with_mean`](src/nlmm/mod.rs); Python [`nlmer_with_mean`](python/src/lib.rs) with a callable mean function.
- Scalar **AGQ** for `nlmer` (`NlmerOptions::n_agq`, `k = 1` random effects); tests in [`tests/test_nlmm_agq.rs`](tests/test_nlmm_agq.rs).
- Python docs for `nlmer(..., start=None)` / `selfStart` in [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md).
- Python `nlmer(..., n_agq=1)` binding (scalar AGQ for `k = 1` RE).
- Golden parity cases `ssmicmen_synthetic_self_start` and `ssgompertz_synthetic_self_start`.
- Shared Gauss–Hermite quadrature in [`src/quadrature.rs`](src/quadrature.rs) (deduped from `glmm_math`).
- Type **I** fixed-effects ANOVA via `AnovaType::Type1` / `anova_type="I"` (sequential contrasts in formula term order).
- `car::linearHypothesis`-style wrappers: `LmeFit::linear_hypothesis` / `linear_hypothesis_terms` (Rust) and `fit.linear_hypothesis()` / `fit.linear_hypothesis_terms()` (Python).
- Python golden parity: `orange_nlmer_sslogis` in [`python/tests/test_golden_parity.py`](python/tests/test_golden_parity.py).

### Changed

- LMM **prepare fast paths**: obs-major `ZᵀZ` accumulation for intercept-only models with `q ≥ 256` ([`build_zt_z_intercept_from_zt`](src/math.rs)); fast nested interaction group indexing ([`try_build_interaction_groups`](src/model_matrix.rs) for `batch:cask`-style factors). `prepare_lmer` on `nested_10k` **~5.3 ms** (was ~7 ms); cold `nested_10k` **~1.53× Julia** (was ~1.6–1.8×); `random_intercept_10k` still **beats Julia** (~1.2 ms vs ~1.3 ms). See [OPTIMIZATION.md § Prepare fast paths pass](OPTIMIZATION.md#prepare-fast-paths-pass-2026-07-08-continued) and [BENCHMARKS.md § 2026-07-08 prepare fast paths](BENCHMARKS.md#fair-rust-julia-2026-07-08-prepare-fast-paths).
- LMM **simple fixed-effects fast path**: [`try_build_simple_x_matrix`](src/model_matrix.rs) for `y ~ 1`, `y ~ x`, and `y ~ 1 + x` skips the generic fiasto loop on fair-harness fixtures. `prepare_lmer` on `crossed_20k` **~4.2 ms**; `random_intercept_10k` still **beats Julia** (~1.3 ms vs ~1.5 ms, 20 repeats). See [OPTIMIZATION.md § Prepare ownership pass](OPTIMIZATION.md#prepare-ownership-pass-2026-07-08-continued) and [BENCHMARKS.md § 2026-07-08 simple X](BENCHMARKS.md#fair-rust-julia-2026-07-08-simple-x).
- LMM **prepare ownership**: [`prepare_lmer`](src/lib.rs) moves `x` / `zt` / `y` / `re_blocks` into [`LmmData`](src/math.rs) via `mem::take` (no duplicate clones); [`build_x_matrix`](src/model_matrix.rs) uses `HashMap<&str, usize>` for categorical dummy encoding. Fair harness on Windows AMD64: cold `crossed_20k` **~1.3× Julia** (was ~1.4×), **`random_intercept_10k` beats Julia** (~1.3 ms vs ~1.7 ms). See [OPTIMIZATION.md § Prepare ownership pass](OPTIMIZATION.md#prepare-ownership-pass-2026-07-08-continued) and [BENCHMARKS.md § 2026-07-08 prepare ownership](BENCHMARKS.md#fair-rust-julia-2026-07-08-prepare-ownership).
- LMM **setup and post-fit** throughput: direct CSR `Zᵀ` build ([`build_zt_csr`](src/model_matrix.rs)), single-pass `&str` group indexing, skip sparse LDL symbolic setup when blocked Cholesky is active, no unweighted `x`/`zt`/`y` clones, row-wise `precompute_zt_products`, lazy sparse LDL reuse in `evaluate()` ([`InterceptLdlCache::solve_profile`](src/math.rs)), and `LME_PERF_DIAG` setup sub-phases (`setup_formula`, `setup_design_matrix`, `setup_lmm_data`). Fair harness on Windows AMD64: cold `crossed_20k` **~1.4× Julia** (was ~1.7×), `random_intercept_10k` **~1.1×** (was ~2.4×), `fit_prepared` on crossed **beats Julia** (~12 ms vs ~16 ms). See [OPTIMIZATION.md § Setup and post-fit pass](OPTIMIZATION.md#setup-and-post-fit-pass-2026-07-08-continued) and [BENCHMARKS.md § 2026-07-08 setup/post-fit](BENCHMARKS.md#fair-rust-julia-2026-07-08-setup-postfit).
- LMM fit throughput (intercept-only / crossed): blocked augmented Cholesky with in-place Schur, reused workspaces, GEMM rank/Schur updates, batched multi-RHS triangular solves, and always-dense cross blocks ≤100k elements (reliable blocked gate on `crossed_20k`). ML 2D log-grid tightened to 5×5 + 4×4 (~42 evals). Post-fit reuses a single `evaluate()` (no duplicate deviance / `Z*b` pass). Documented in [`OPTIMIZATION.md`](OPTIMIZATION.md) and [`BENCHMARKS.md`](BENCHMARKS.md#fair-rust-julia-2026-07-08-gemm-prepared). Fair `crossed_20k`: **`fit_prepared` ~1× Julia**; cold `lmer()` ~2× (explicit setup + post-fit).
- LMM fit throughput: reuse [`LmmData`](src/math.rs) across Nelder–Mead evaluations, precompute `Z^T X` / `Z^T y`, deviance-only optimizer path, and intercept-only diagonal-Λ fast path in [`src/math.rs`](src/math.rs) / [`src/optimizer.rs`](src/optimizer.rs). Golden-section θ search for |θ|=1; 2D log-grid for |θ|=2 (ML). Updated fair-harness snapshots in [`BENCHMARKS.md`](BENCHMARKS.md).
- Test suite speed: `[profile.test] opt-level = 2`, parallel golden-parity cases (`rayon`), smaller debug smoke fixtures, `task test:fast` / `lme_ci.py test-fast`, and leaner CI test step (no separate `cargo build` before `cargo test`).

- Bump `rand` to **0.9.3+** (and transitive **0.8.6** where applicable) for [RUSTSEC-2026-0097](https://rustsec.org/advisories/RUSTSEC-2026-0097).

### Added

- `nlmer` **`SSfol`** mean (`stats::SSfol`, same formula as `SSasymp`) with parity on synthetic grouped data ([`tests/test_nlmm_ssfol.rs`](tests/test_nlmm_ssfol.rs)).
- R-style **`selfStart`** automatic starting values when `start` is empty: [`src/nlmm/self_start.rs`](src/nlmm/self_start.rs) (`stats::getInitial` heuristics for `SSlogis` / `SSasymp` / `SSfol`), with multistart fallback to static defaults in [`fit_nlmer`](src/nlmm/fit.rs). Tests: [`tests/test_nlmm_self_start.rs`](tests/test_nlmm_self_start.rs).
- Golden parity case **`ssfol_synthetic_self_start`** (no explicit `nlmm_start`) in [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json); fixture [`tests/data/ssfol_nlmer.json`](tests/data/ssfol_nlmer.json); R regeneration in [`tests/generate_test_data.R`](tests/generate_test_data.R).

- Golden parity expansion: offset LMM/GLMM, probit/weighted binomial GLMM, and Orange `nlmer` predictions in [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) (regenerated via [`tests/generate_test_data.R`](tests/generate_test_data.R)). Rust checks in [`tests/test_golden_parity.rs`](tests/test_golden_parity.rs), [`tests/test_glmm_offset_grouseticks.rs`](tests/test_glmm_offset_grouseticks.rs); Python in [`python/tests/test_golden_parity.py`](python/tests/test_golden_parity.py).
- Cross-language coefficient parity exporters under [`comparisons/parity/`](comparisons/parity/) (R, Julia) plus Rust [`parity_export`](comparisons/parity_export.rs) example; orchestration via [`scripts/verify_cross_language_parity.py`](scripts/verify_cross_language_parity.py).
- Explicit GLMM link selection: [`family::Link`](src/family.rs), [`glmer_with_link`](src/lib.rs) / [`glmer_weighted_with_link`](src/lib.rs). Python: `glmer(..., link_name="probit")`. Response-scale prediction respects the fitted link. Tests in [`tests/test_glmm_links.rs`](tests/test_glmm_links.rs).
- `nlmer` population and conditional prediction: `LmeFit::predict()` / `predict_conditional()` evaluate `SSlogis` at fixed parameters (`re.form = NA`) or with stored random effects on `Asym` (`re.form = NULL`). Python `fit.predict()` / `fit.predict_conditional()` work unchanged. Tests in [`tests/test_nlmm_orange.rs`](tests/test_nlmm_orange.rs).

### Fixed

- Julia fair harness ([`comparisons/bench_fair_julia_timing.jl`](comparisons/bench_fair_julia_timing.jl)): removed obsolete `ProgressMeter.enable(false)` so MixedModels.jl timing runs on ProgressMeter 1.11+.
- GLMM PIRLS with `offset(...)`: penalized WLS now regresses `z − offset` on fixed and random effects so coefficients match `lme4::glmer` (e.g. grouseticks Poisson offset case).
- **`nlmer` expansion:** `SSasymp` mean with R parity on synthetic data ([`tests/test_nlmm_ssasymp.rs`](tests/test_nlmm_ssasymp.rs)); multivariate RE parsing (`Asym + xmid | Tree`) with relative-θ Cholesky covariance, correlated Orange parity, and conditional prediction ([`tests/test_nlmm_orange_multi_re.rs`](tests/test_nlmm_orange_multi_re.rs)).

## [0.1.8] - 2026-07-01

### Added

- Python bindings pushed to ~95% Rust parity: `lm_matrix`, `contrast_matrix`, structured results (`PyConfintResult`, `PyFixedEffectsAnova`, `PyContrastTest`, `PyLikelihoodRatioAnova`, `PySimulateResult`), extra [`PyLmeFit`](python/src/lib.rs) getters (`b`, `u`, `beta_se`, `fixed_term_assign`, `categorical_levels`, `v_beta_unscaled`), and [`lme_python.pyi`](python/lme_python.pyi). Prior: `nlmer`, `glmer_weighted`, `contrast_matrix_from_names`, `test_contrast_vs`, metadata getters. Tests in [`python/tests/test_api_parity.py`](python/tests/test_api_parity.py).
- Weighted GLMMs: [`glmer_weighted`](src/lib.rs) with prior observation weights in PIRLS / Laplace deviance (delegates to [`lmer_weighted`](src/lib.rs) for Gaussian). Python: `lme_python.glmer_weighted(..., weights=...)`. Tests: [`tests/test_glmm_weighted.rs`](tests/test_glmm_weighted.rs).
- Nonlinear mixed models (`nlmer`-style): [`nlmer`](src/nlmm/mod.rs) with three-part formulas (`response ~ SSlogis(cov, Asym, xmid, scal) ~ Asym|group`), penalized Gauss–Newton inner loop, and golden-section profiling of a single random-effect standard deviation. Parity test [`tests/test_nlmm_orange.rs`](tests/test_nlmm_orange.rs) against `lme4::nlmer(Orange)` (ML, random effect on `Asym` only).
- User-defined fixed-effects contrast tests: [`LmeFit::test_contrast`](src/contrast.rs), [`test_contrast_vs`](src/contrast.rs), and [`contrast_matrix_from_names`](src/contrast.rs) ([`tests/test_contrast.rs`](tests/test_contrast.rs)). Python: `fit.test_contrast(l_matrix)`.
- Kenward–Roger multi-DoF ANOVA via `pbkrtest::KRmodcomp` / `.KR_adjust` ([`src/kr_modcomp.rs`](src/kr_modcomp.rs)) and structural `vcovAdj16` ([`src/kr_vcov_adj.rs`](src/kr_vcov_adj.rs)), wired from [`src/anova.rs`](src/anova.rs). Parity test [`tests/test_kr_modcomp_pastes.rs`](tests/test_kr_modcomp_pastes.rs); golden case `pastes_cask_multi_dof_reml` now checks KR ANOVA.

### Changed

- Kenward–Roger multi-DoF rows no longer use marginal-df pooling only; they use full `KRmodcomp` when `vcovAdj` differs from `vcov`, with pooling when adjustment is negligible (e.g. pastes batch RE).
- CI and local dev tooling: single cross-platform [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py) runner (Task, Lefthook, GitHub Actions, legacy `local_ci` scripts); [`Taskfile.yml`](Taskfile.yml), [`lefthook.yml`](lefthook.yml), [`mise.toml`](mise.toml) pins; Ruff on `python/tests` and `python/examples`. See [`AGENTS.md`](AGENTS.md).

## [0.1.7] - 2026-05-18

### Added

- **Type II** fixed-effects ANOVA via `LmeFit::anova_typed(AnovaType::Type2, ddf)` ([`src/anova_contrasts.rs`](src/anova_contrasts.rs), `lmerTest`-style Doolittle contrasts for contained terms). `anova(ddf)` remains Type III. Python: `fit.anova("satterthwaite", anova_type="II")`.
- Design-matrix support for **`FixedEffect`** / **`InteractionTerm`** fiasto roles and two-way categorical interactions (`y ~ a * b`).
- Golden parity case **`pastes_cask_multi_dof_reml`** in [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) (Type III Satterthwaite ANOVA for `cask` on the pastes dataset), with fixture [`tests/data/pastes_cask_reml.json`](tests/data/pastes_cask_reml.json) and R regeneration via [`tests/generate_test_data.R`](tests/generate_test_data.R).
- Numeric parity checks for categorical multi-DoF ANOVA in [`tests/categorical_anova_test.rs`](tests/categorical_anova_test.rs) and [`python/examples/verification_project/parity.py`](python/examples/verification_project/parity.py).

### Changed

- Multi-DoF Type III ANOVA denominator df for Satterthwaite now follows **`lmerTest::contestMD()`** ([`src/ddf.rs`](src/ddf.rs)): eigen-decomposed Wald contrasts, per-direction Satterthwaite dfs, and `get_Fstat_ddf()` pooling (replacing `min()` of marginal dfs). Kenward–Roger multi-DoF rows use the same pooling on marginal dfs.

## [0.1.6] - 2026-03-27

### Added

- Python `examples/plotting_demo`: matplotlib figures from `lme_python`, optional **lme4** parity (`plot_r.R`), side-by-side PNGs (`figures_compare/`), numeric overlay on shared axes (`figures_overlay/` via `figures_data/` JSON/CSV), `run_all.py`, Windows `Rscript` discovery (`find_rscript.py`), and raster fallback (`figures_overlay_raster/`).
- `scripts/local_ci.sh` and `scripts/local_ci.ps1` to run the same checks as [`.github/workflows/ci.yml`](.github/workflows/ci.yml) locally (`fmt`, `clippy`, build, test, `doc`).

## [0.1.5] - 2026-03-22

### Added

- Added automatic categorical dummy encoding (dropping collinear intercepts) directly from categorical/string Polars dataframe columns natively.
- Implemented Multi-DoF Joint Wald F-Tests tracking dummy groups in `anova()` (Type III Satterthwaite representations) for valid categorical fixed-effect testing.
- Added support for generating `Gaussian` families in Python's `glmer` mapping, alongside exposing an equivalent `lmer_weighted` function. 
- Plumbed explicit Adaptive Gauss-Hermite Quadrature (`nAGQ`) scaling configuration arrays throughout the optimizer targeting upcoming structural convergence capabilities.

## [0.1.4] - 2026-03-22

### Added

- Formula-string entry point `lm()` and corresponding Python binding for fitting ordinary least squares models using DataFrames.
- New Python examples directory `python/examples` with five runnable demonstration scripts mapping to common mixed-modeling workflows.

### Changed

- Renamed `examples/` directory to `comparisons/` to clarify that it contains cross-language parity scripts, updating all build and documentation references accordingly.
- Promoted Kenward-Roger degrees of freedom from "provisional" status after validating numerical parity (within 0.01 df) against R's `pbkrtest`.

## [0.1.3] - 2026-03-12

### Added

- Expanded the benchmark suite to cover parsing, matrix construction, prediction paths, inference helpers, weighted fits, Kenward-Roger timing, and size sweeps.
- Added automated cross-language benchmark scripting and a GitHub workflow that saves benchmark artifacts in CI and on release tags.
- Added `CONTRIBUTING.md`, `RELEASING.md`, and `BENCHMARKS.md` to document contributor setup, release flow, and benchmark methodology.

### Changed

- Reworked the README, Rust guide, Python guide, and comparison documentation to better reflect current capabilities, limitations, and release workflow.
- Switched the crate homepage to the docs.rs site and expanded repository metadata sync for GitHub description, topics, and website fields.
- Hardened the benchmark automation and example scripts so cross-language benchmark runs complete consistently on GitHub Actions.

## [0.1.2] - 2026-03-09

### Fixed

- Fixed a NaN deviance issue in the fitting path.
- Expanded Python test coverage for the 0.1.2 release.

### Changed

- Aligned `Cargo.lock` with the 0.1.2 version bump.

## [0.1.1] - 2026-03-09

### Changed

- Bumped the crate and Python extension versions for the 0.1.1 release.
- Updated lockfiles to remove yanked dependencies.
- Fixed missing documentation warnings needed for publishing.

### Fixed

- Hardened the GitHub Actions and wheel build pipeline across Linux, macOS, and Windows.
- Restored missing dependencies and build flags needed for OpenBLAS-based wheel builds.
- Fixed several maturin and PyO3 packaging workflow issues for the Python release path.

## [0.1.0] - 2026-03-08

### Added

- **Core LMM** - `lmer()` with REML and ML estimation, matching R's `lme4` to 4 decimal places
- **GLMM** - `glmer()` with Poisson, Binomial, Gaussian, and Gamma families via Laplace approximation
- **Predictions** - `predict()`, `predict_conditional()`, `predict_response()`, `predict_conditional_response()` with `allow_new_levels` support
- **Inference** - Satterthwaite and Kenward-Roger degrees of freedom, robust sandwich standard errors
- **Model comparison** - `anova()` for likelihood ratio tests between nested models
- **Diagnostics** - `confint()` for Wald confidence intervals, `simulate()` for parametric bootstrap
- **Formula parser** - Wilkinson notation with nested (`a/b`), crossed (`a + b`), and correlated slope+intercept (`x | group`) random effects
- **Sparse matrix architecture** - Compressed sparse column format with sparse Cholesky for large datasets
- **Python bindings** - `lme_python` package via PyO3/maturin with `lmer()`, `glmer()`, `predict()`
- **Documentation** - `GUIDE.md`, `PYTHON_GUIDE.md`, and cross-language `COMPARISONS.md`
