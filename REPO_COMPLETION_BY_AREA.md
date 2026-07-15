# Repository completion by area

This file gives **approximate completion percentages** for major parts of the `lme-rs` repository. The numbers are **judgment calls** for planning and transparency, not precise metrics (they are not derived from line coverage or a formal roadmap).

**This is a coverage map, not a usability guide.** For “can I use this on my problem?” (workflows, validation posture, limited field experience), see **[USABILITY.md](USABILITY.md)**.

**Last assessed:** 2026-07-09.

**Versions checked:** `lme-rs` **0.1.9** (root [`Cargo.toml`](Cargo.toml)); Python extension **`lme_python` 0.1.9** ([`python/Cargo.toml`](python/Cargo.toml)).

Repository completion is judged on **three** axes, not features alone:

1. **Correctness / lme4-aligned behavior** — golden tests, comparisons, documented scope.
2. **Shipped API surface** — Rust crate, Python bindings, docs, CI.
3. **Competitive fit throughput** — on the [fair Rust vs Julia harness](BENCHMARKS.md#fair-rust-vs-julia-reference-results), `lme-rs` should be **in the same ballpark as MixedModels.jl** on core LMM cases (random intercept, crossed, nested), not an order of magnitude slower. This axis is **usability for performance-sensitive workflows**, not a separate concern from [USABILITY.md](USABILITY.md); instrumentation exists and optimization remains on the critical path until medians are competitively close.

Until axis (3) closes, overall completion percentages are capped in practice even when feature rows look high.

## How to read the percentages

| Range | Meaning |
|:------|:--------|
| **90–100%** | Mature for the scope described in docs; exercised by tests/CI where applicable, and documented limitations are narrow. |
| **75–89%** | Feature-complete for core workflows; known gaps are documented (e.g. parity caveats, API subset). |
| **55–74%** | Usable and supported, but intentionally partial vs a larger reference (e.g. full `lme4` or the full Rust API from Python). |
| **Below 55%** | Experimental, optional, or not a product goal—treated as auxiliary. |

## Summary table

| # | Area | Completion | Notes |
|---|------|:----------:|-------|
| 1 | **Rust crate: linear & mixed (LMM)** — [`lm`](src/lib.rs) / [`lm_df`](src/lib.rs), [`lmer`](src/lib.rs), [`lmer_weighted`](src/lib.rs), REML/ML, [`predict`](src/lib.rs) variants | **95%** | Core path is stable; parity-style checks in [`tests/test_numerical_parity.rs`](tests/test_numerical_parity.rs), e2e and optimization tests under [`tests/`](tests/). Fair-harness synthetic tier-A throughput is **&lt;1.0×** Julia on cold `lmer()` (2026-07-16) — see row 13 and [OPTIMIZATION.md](OPTIMIZATION.md). |
| 2 | **Rust crate: GLMM** — [`glmer`](src/lib.rs), [`glmer_weighted`](src/lib.rs), [`family`](src/family.rs), PIRLS in [`glmm_math`](src/glmm_math.rs), Laplace vs scalar AGQ (`n_agq`) | **88%** | Implemented and tested ([`tests/test_glmm.rs`](tests/test_glmm.rs), [`tests/test_glmm_weighted.rs`](tests/test_glmm_weighted.rs), [`tests/test_glmm_links.rs`](tests/test_glmm_links.rs)); explicit [`Link`](src/family.rs) API; README documents Laplace optimization vs AGQ-in-final-eval and deviance constants. |
| 3 | **Rust crate: formula & model matrices** — [`formula`](src/formula.rs), [`model_matrix`](src/model_matrix.rs) | **87%** | Broad Wilkinson + RE support; remaining gap is breadth of edge cases vs R, not missing baseline features ([`tests/test_formula.rs`](tests/test_formula.rs), [`tests/test_crossed_mock.rs`](tests/test_crossed_mock.rs), etc.). |
| 4 | **Rust crate: post-fit inference** — [`confint`](src/lib.rs), [`simulate`](src/lib.rs), [`boot_lmer`](src/bootstrap.rs), [`with_robust_se`](src/lib.rs), [`with_satterthwaite`](src/lib.rs), [`with_kenward_roger`](src/lib.rs) | **88%** | Covered by targeted tests ([`test_confint_simulate.rs`](tests/test_confint_simulate.rs), [`test_bootstrap.rs`](tests/test_bootstrap.rs), [`test_robust.rs`](tests/test_robust.rs), [`test_satterthwaite.rs`](tests/test_satterthwaite.rs), [`test_kenward_roger.rs`](tests/test_kenward_roger.rs)); scope matches guides and comparisons, not every GLMM edge case or full `bootMer` option set. |
| 5 | **Rust crate: ANOVA & model comparison** — Type III: [`LmeFit::anova`](src/anova.rs); nested LRT: [`anova`](src/lib.rs) (`AnovaResult`) | **92%** | Type I/II/III; `linear_hypothesis`; 1-DoF marginal KR; joint multi-DoF Wald; user contrasts [`test_contrast`](src/contrast.rs) ([`tests/test_contrast.rs`](tests/test_contrast.rs)). Not full `car` / `lmerTest` superset (e.g. `glht`). |
| 6 | **Python bindings** (`python/`, import `lme_python`) | **~99%** | Near-full Rust parity: formula `lm`/`lmer`/`glmer`/`nlmer` (incl. `n_agq`), [`prepare_lmer`](python/src/lib.rs) / [`fit_prepared`](python/src/lib.rs) / [`cv_grouped`](python/src/lib.rs) / [`boot_lmer`](python/src/lib.rs), [`nlmer_with_mean`](python/src/lib.rs) for custom nonlinear means, matrix [`lm_matrix`](python/src/lib.rs), `contrast_matrix` / `contrast_matrix_from_names`, structured result types (`PyConfintResult`, `PyFixedEffectsAnova`, `PyLikelihoodRatioAnova`, `PyBootLmerResult`, …), [`lme_python.pyi`](python/lme_python.pyi). Remaining gap: low-level matrix [`lm(y, x)`](src/lib.rs) without a DataFrame (Rust-only). [`python/tests/`](python/tests/) including [`test_api_parity.py`](python/tests/test_api_parity.py) and [`test_cv.py`](python/tests/test_cv.py). |
| 7 | **Cross-language validation** — [`comparisons/`](comparisons/), JSON/CSV fixtures, Rust tests | **93%** | [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md) explains what is regression-tested vs manual; [`tests/test_golden_parity.rs`](tests/test_golden_parity.rs) includes pastes / `cask` multi-DoF ANOVA and nlmer cases (`orange_nlmer_sslogis`, `ssfol` / `ssmicmen` / `ssgompertz` / **`sspower`** self-start); Python mirror in [`python/tests/test_golden_parity.py`](python/tests/test_golden_parity.py). |
| 8 | **Benchmarks (instrumentation)** — [`benches/bench_math.rs`](benches/bench_math.rs), [`BENCHMARKS.md`](BENCHMARKS.md), [`BENCHMARK_COVERAGE.md`](BENCHMARK_COVERAGE.md), fair Rust/Julia harness, [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) | **88%** | Criterion + tier-A fair harness (LMM + GLMM fixtures, optional `prepare`/`fit_prepared` phases); [coverage map](BENCHMARK_COVERAGE.md) separates measured vs Rust-only workloads. Workflow runs on `v*` tags and `workflow_dispatch`. |
| 9 | **CI, release, and repo automation** | **98%** | Single runner [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py) via [`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`Taskfile.yml`](Taskfile.yml), Lefthook, and legacy `local_ci` scripts; GitHub Actions are automatic only on `v*` tags plus manual dispatch, while local hooks/Task cover ordinary pushes and PR prep. Release CI uses **`--locked`**, **`cargo check --all-targets`**, **`cargo test --doc`**, multi-OS, Python **3.11** + wheel reinstall + second **`pytest`**, and **3.10 / 3.12 / 3.13** on Ubuntu. [`.github/workflows/audit.yml`](.github/workflows/audit.yml), [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml), [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml), and [`.github/workflows/repo-metadata.yml`](.github/workflows/repo-metadata.yml) are also tag/manual oriented. |
| 10 | **End-user documentation** — [`GUIDE.md`](GUIDE.md), [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md), [`USABILITY.md`](USABILITY.md), [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md), [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`RELEASING.md`](RELEASING.md), [`BENCHMARKS.md`](BENCHMARKS.md), [`OPTIMIZATION.md`](OPTIMIZATION.md) | **92%** | README documentation map is accurate; [USABILITY.md](USABILITY.md) separates workflow scope from [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) coverage. Some release links (e.g. benchmark artifacts version on README) may lag the current crate version—prefer [`CHANGELOG.md`](CHANGELOG.md) for history. |
| 11 | **Examples & optional demos** — Cargo `[[example]]` entries in [`Cargo.toml`](Cargo.toml) under `comparisons/`, [`python/examples/`](python/examples/), [`scripts/run_cross_language_benchmarks.py`](scripts/run_cross_language_benchmarks.py) | **76%** | Comparison binaries are first-class; plotting and cross-language scripts are useful but partly manual or environment-dependent. |
| 12 | **Experimental / exploratory code** — [`scripts/ast_explorations/`](scripts/ast_explorations/) | **35%** | Standalone Rust snippets; not wired into the crate or CI. Other `scripts/` helpers (benchmark drivers, R dumps) are **tooling**, not “library completion.” |
| 13 | **LMM fit throughput vs MixedModels.jl** — optimization to be **competitive** on fair harness cases | **~98%** | **Tier-A cases:** [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md). Axis (3) target is **&lt;1.0×** Julia on `cold_fit`. Jul 16 prepare/gate pass: `crossed_20k` / `nested_10k` **~0.91× / ~0.93×** ([reference](benchmarks/fair-rust-julia-reference-2026-07-16-cold-fit-lt1.json)). **`fit_prepared` beats Julia on every measured LMM case.** |

## Weighted “overall” (illustrative only)

Simple mean of the thirteen percentages:  
(94 + 88 + 87 + 86 + 92 + 99 + 93 + 88 + 98 + 92 + 76 + 35 + 94) ÷ 13 = 1122 ÷ 13 ≈ **86.3%**.

The Jul 9 setup pass shows **~0.38–1.31× Julia** cold fits on the real, crossed, nested, 10k-intercept, and random-slopes cases, plus **~0.47× / ~0.51×** at 50k/100k random-intercept scale. `fit_prepared` beats Julia on every measured LMM case. Axis (3) now meets its current tier-A throughput criterion; retain the fair harness as a regression guard.

## Evidence pointers (verified)

| Topic | Primary sources |
|:------|:----------------|
| Scope and limitations | [`README.md`](README.md) (“Current status”, “Limitations and compatibility notes”) |
| Type III ANOVA (incl. categorical joint tests) | [`README.md`](README.md); [`src/anova.rs`](src/anova.rs); [`src/ddf.rs`](src/ddf.rs); [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) (`pastes_cask_multi_dof_reml`) |
| Python vs Rust breadth | [`README.md`](README.md); [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |
| Numerical validation | [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md); [`tests/test_numerical_parity.rs`](tests/test_numerical_parity.rs); [`tests/test_glmm.rs`](tests/test_glmm.rs) |
| Benchmarks / throughput | [`BENCHMARK_COVERAGE.md`](BENCHMARK_COVERAGE.md); [`BENCHMARKS.md`](BENCHMARKS.md); [`OPTIMIZATION.md`](OPTIMIZATION.md); [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) |
| Usability vs coverage | [`USABILITY.md`](USABILITY.md); row 13 vs workflow traffic lights |
| LMM throughput optimization backlog | [OPTIMIZATION.md](OPTIMIZATION.md); summary table row 13; [`benches/bench_math.rs`](benches/bench_math.rs) size/crossed/nested sweeps |
| Rust workflows | [`GUIDE.md`](GUIDE.md) |
| CI layout | [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py); [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (`v*` tag / manual triggers, `--locked`, all-targets check, doctests, `pytest tests/`, Python version matrix on Ubuntu); [`AGENTS.md`](AGENTS.md) |
| nlmer means, AGQ, custom μ | [`src/nlmm/`](src/nlmm/); [`tests/test_nlmm_ssmicmen.rs`](tests/test_nlmm_ssmicmen.rs), [`tests/test_nlmm_sspower.rs`](tests/test_nlmm_sspower.rs), [`tests/test_nlmm_custom_mean.rs`](tests/test_nlmm_custom_mean.rs), [`tests/test_nlmm_agq.rs`](tests/test_nlmm_agq.rs); [`comparisons/nlmm_sspower.R`](comparisons/nlmm_sspower.R); [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |
| Integration tests | **40** Rust modules under [`tests/`](tests/) (including `test_nlmm_orange.rs`, `test_nlmm_ssmicmen.rs`, `test_glmm_weighted.rs`, `test_contrast.rs`; counted 2026-07-04) |

## Gaps vs the full R ecosystem (not in the summary table)

The summary table scores the **intended shipped surface** of this repo. This section lists **extensions relative to the whole `lme4` / `nlme` / `car` stack** — things that are either **not product goals**, **partially shipped**, or **explicitly not started**. Percentages here are vs that **larger reference**, not “missing from `lme-rs`.”

### Partial — usable subset shipped; not a full ecosystem replacement

| Topic | vs ecosystem | Notes |
|:------|:-------------|:------|
| **Nonlinear mixed models** (`nlmer`-style) | **~90%** | [`nlmer`](src/nlmm/mod.rs) + Python `nlmer()` / [`nlmer_with_mean`](python/src/lib.rs); built-in `SSlogis` / `SSasymp` / `SSfol` / `SSmicmen` / `SSgompertz` / **`SSpower`**; optional population `lower`/`upper` box bounds; `selfStart` when `start` is empty (built-in means only); population/conditional [`predict`](src/nlmm/predict.rs); scalar and multivariate RE on one grouping factor; scalar AGQ. **Not yet:** full `stats::SS*` catalog, multivariate AGQ, AGQ inside the θ optimizer (same caveat as GLMM), per-group (β+b) bounds. |
| **Weighted GLMMs** (`glmer` + observation weights) | **~80%** | [`glmer_weighted`](src/lib.rs) mirrors [`lmer_weighted`](src/lib.rs). **Not yet:** frequency weights / offsets combined with weights beyond what is documented and tested. |

> **Throughput vs MixedModels.jl** is scored in summary **row 13** (~84%, Jul 2026) — not duplicated here.

### Not started (≈0% as a product goal)

| Topic | Status | Notes |
|:------|:-------|:------|
| **Publication-grade cross-language benchmark harness** | **0%** | [`BENCHMARKS.md`](BENCHMARKS.md): fair fit-only harness exists for engineering, but there is **no** machine-locked, publication-normalized speed product for public claims. |

### Already in the summary table (do not list here as “unrealized”)

These are **shipped and scored above** — they are incomplete only vs a much larger reference (full `car` / `lmerTest`, every GLMM edge case, etc.):

- Fixed-effects ANOVA (Type I–III), contrasts, `linear_hypothesis` — summary **row 5**
- GLMM families / links — summary **row 2**
- Python ↔ Rust formula API — summary **row 6**

If a gap closes (new golden case, new mean, benchmark product), update the relevant row in **this** section or the summary table and bump **Last assessed**.

## Maintenance

When a major capability lands or a limitation is removed, update the relevant row, bump **Last assessed**, and recompute the simple mean if you still report it. Re-check **Versions checked** against [`Cargo.toml`](Cargo.toml) and [`python/Cargo.toml`](python/Cargo.toml) on each release.

When fit throughput improves, re-run [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py), update [`benchmarks/fair-rust-julia-reference-*.json`](benchmarks/fair-rust-julia-reference-2026-07-06.json), and raise row **13** (and the overall mean) only when medians are competitively close to MixedModels.jl on the reference cases.
