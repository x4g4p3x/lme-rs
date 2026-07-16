# Repository completion by area

This file gives **approximate completion percentages** for major parts of the `lme-rs` repository. The numbers are **judgment calls** for planning and transparency, not precise metrics (they are not derived from line coverage or a formal roadmap).

**This is a coverage map, not a usability guide.** For “can I use this on my problem?” (workflows, validation posture, limited field experience), see **[USABILITY.md](USABILITY.md)**.

**Last assessed:** 2026-07-16.

**Versions checked:** `lme-rs` **0.1.11** (root [`Cargo.toml`](Cargo.toml)); Python extension **`lme_python` 0.1.11** ([`python/Cargo.toml`](python/Cargo.toml)).

Repository completion is judged on **three** axes, not features alone:

1. **Correctness / lme4-aligned behavior** — golden tests, comparisons, documented scope.
2. **Shipped API surface** — Rust crate, Python bindings, docs, CI.
3. **Competitive fit throughput** — on the [fair Rust vs Julia harness](BENCHMARKS.md#fair-rust-vs-julia-reference-results), `lme-rs` should be **in the same ballpark as MixedModels.jl** on core LMM cases (random intercept, crossed, nested), not an order of magnitude slower. This axis is **usability for performance-sensitive workflows**, not a separate concern from [USABILITY.md](USABILITY.md). **Status (2026-07-16):** tier-A `cold_fit` target **&lt;1.0×** Julia is **met** ([reference](benchmarks/fair-rust-julia-reference-2026-07-16-cold-fit-lt1.json)); remaining [OPTIMIZATION.md](OPTIMIZATION.md) items are optional polish, not blockers for rows 1 or 13.

Axis (3) no longer caps rows 1 / 13 once the declared tier-A bar is met; keep the fair harness as a regression guard.

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
| 1 | **Rust crate: linear & mixed (LMM)** — [`lm`](src/lib.rs) / [`lm_df`](src/lib.rs), [`lmer`](src/lib.rs), [`lmer_weighted`](src/lib.rs), REML/ML, [`predict`](src/lib.rs) variants | **100%** | Intended LMM surface complete: parity/e2e/goldens (incl. `dyestuff_intercept_reml`), REML/ML, weights, predict. Tier-A cold `lmer()` **&lt;1.0×** Julia (row 13); optional [OPTIMIZATION.md](OPTIMIZATION.md) leftovers are non-blocking. |
| 2 | **Rust crate: GLMM** — [`glmer`](src/lib.rs), [`glmer_weighted`](src/lib.rs), [`family`](src/family.rs), PIRLS in [`glmm_math`](src/glmm_math.rs), Laplace vs scalar AGQ (`n_agq`) | **90%** | Implemented and tested ([`tests/test_glmm.rs`](tests/test_glmm.rs), [`tests/test_glmm_weighted.rs`](tests/test_glmm_weighted.rs), [`tests/test_glmm_links.rs`](tests/test_glmm_links.rs)); explicit [`Link`](src/family.rs) API; scalar RE uses AGQ inside the θ objective when `n_agq > 1`. |
| 3 | **Rust crate: formula & model matrices** — [`formula`](src/formula.rs), [`model_matrix`](src/model_matrix.rs) | **87%** | Broad Wilkinson + RE support; remaining gap is breadth of edge cases vs R, not missing baseline features ([`tests/test_formula.rs`](tests/test_formula.rs), [`tests/test_crossed_mock.rs`](tests/test_crossed_mock.rs), etc.). |
| 4 | **Rust crate: post-fit inference** — [`confint`](src/lib.rs), [`confint_profile`](src/profile_ci.rs), [`simulate`](src/lib.rs), [`boot_lmer`](src/bootstrap.rs) / [`boot_glmer`](src/bootstrap.rs), [`with_robust_se`](src/lib.rs), [`with_satterthwaite`](src/lib.rs), [`with_kenward_roger`](src/lib.rs) | **92%** | Wald + profile (incl. `parms=`); parametric LMM/GLMM bootstrap; targeted tests (`test_confint_profile.rs`, `test_bootstrap.rs`, …). |
| 5 | **Rust crate: ANOVA & model comparison** — Type III: [`LmeFit::anova`](src/anova.rs); nested LRT: [`anova`](src/lib.rs) (`AnovaResult`) | **92%** | Type I/II/III; `linear_hypothesis`; 1-DoF marginal KR; joint multi-DoF Wald; user contrasts [`test_contrast`](src/contrast.rs) ([`tests/test_contrast.rs`](tests/test_contrast.rs)). Not full `car` / `lmerTest` superset (e.g. `glht`). |
| 6 | **Python bindings** (`python/`, import `lme_python`) | **~99%** | Near-full Rust parity including [`prepare_lmer`](python/src/lib.rs) / [`prepare_glmer`](python/src/lib.rs) / [`fit_prepared_glmer`](python/src/lib.rs) / [`cv_grouped`](python/src/lib.rs) / [`cv_grouped_glmer`](python/src/lib.rs) / boot APIs / profile `parms=`. Remaining gap: low-level matrix [`lm(y, x)`](src/lib.rs) without a DataFrame (Rust-only). |
| 7 | **Cross-language validation** — [`comparisons/`](comparisons/), JSON/CSV fixtures, Rust tests | **96%** | Golden parity includes dyestuff intercept LMM, `ssfpl` / `ssasympoff` / `ssasymporig` / `ssbiexp` / `ssweibull`, CBPP AGQ-7, sleepstudy profile CIs; remaining gap is breadth vs full R edge-case matrix. |
| 8 | **Benchmarks (instrumentation)** — [`benches/bench_math.rs`](benches/bench_math.rs), [`BENCHMARKS.md`](BENCHMARKS.md), [`BENCHMARK_COVERAGE.md`](BENCHMARK_COVERAGE.md), fair Rust/Julia harness, [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) | **88%** | Criterion + tier-A fair harness (LMM + GLMM fixtures, optional `prepare`/`fit_prepared` phases); [coverage map](BENCHMARK_COVERAGE.md) separates measured vs Rust-only workloads. Workflow runs on `v*` tags and `workflow_dispatch`. |
| 9 | **CI, release, and repo automation** | **98%** | Single runner [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py) via [`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`Taskfile.yml`](Taskfile.yml), Lefthook, and legacy `local_ci` scripts; GitHub Actions are automatic only on `v*` tags plus manual dispatch, while local hooks/Task cover ordinary pushes and PR prep. Release CI uses **`--locked`**, **`cargo check --all-targets`**, **`cargo test --doc`**, multi-OS, Python **3.11** + wheel reinstall + second **`pytest`**, and **3.10 / 3.12 / 3.13** on Ubuntu. [`.github/workflows/audit.yml`](.github/workflows/audit.yml), [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml), [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml), and [`.github/workflows/repo-metadata.yml`](.github/workflows/repo-metadata.yml) are also tag/manual oriented. |
| 10 | **End-user documentation** — [`GUIDE.md`](GUIDE.md), [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md), [`USABILITY.md`](USABILITY.md), [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md), [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`RELEASING.md`](RELEASING.md), [`BENCHMARKS.md`](BENCHMARKS.md), [`OPTIMIZATION.md`](OPTIMIZATION.md) | **100%** | README documentation map accurate; guides cover prepare/CV/boot (LMM+GLMM), profile `parms=`, AGQ-in-θ, full built-in `SS*` catalog + bounds; USABILITY traffic lights current for **0.1.11** / 2026-07-16; BENCHMARKS points to `releases/latest` + in-repo fair JSON (not a stale tag as SoT). |
| 11 | **Examples & optional demos** — Cargo `[[example]]` entries in [`Cargo.toml`](Cargo.toml) under `comparisons/`, [`python/examples/`](python/examples/), [`scripts/run_cross_language_benchmarks.py`](scripts/run_cross_language_benchmarks.py) | **76%** | Comparison binaries are first-class; plotting and cross-language scripts are useful but partly manual or environment-dependent. |
| 12 | **Experimental / exploratory code** — [`scripts/ast_explorations/`](scripts/ast_explorations/) | **35%** | Standalone Rust snippets; not wired into the crate or CI. Other `scripts/` helpers (benchmark drivers, R dumps) are **tooling**, not “library completion.” |
| 13 | **LMM fit throughput vs MixedModels.jl** — optimization to be **competitive** on fair harness cases | **100%** | **Tier-A cases:** [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md). Declared axis-(3) target **&lt;1.0×** Julia on `cold_fit` is **met** ([2026-07-16](benchmarks/fair-rust-julia-reference-2026-07-16-cold-fit-lt1.json): `crossed_20k` / `nested_10k` **~0.91× / ~0.93×**). **`fit_prepared` beats Julia on every measured LMM case.** Remaining OPTIMIZATION items are optional polish / regression-guard only. |

## Weighted “overall” (illustrative only)

Simple mean of the thirteen percentages:  
(100 + 90 + 87 + 92 + 92 + 99 + 96 + 88 + 98 + 100 + 76 + 35 + 100) ÷ 13 = 1153 ÷ 13 ≈ **88.7%**.

The Jul 16 axis-(3) pass locks tier-A cold `lmer()` **&lt;1.0×** Julia (`crossed_20k` / `nested_10k` **~0.91× / ~0.93×**). `fit_prepared` beats Julia on every measured LMM case. Rows **1** and **13** are complete for the declared intended surface; keep the fair harness as a regression guard. Optional [OPTIMIZATION.md](OPTIMIZATION.md) polish does not reopen those rows.

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
| **Nonlinear mixed models** (`nlmer`-style) | **~98%** | [`nlmer`](src/nlmm/mod.rs) + Python; built-in `SSlogis` / `SSasymp` / `SSfol` / `SSmicmen` / `SSgompertz` / `SSpower` / `SSfpl` / `SSbiexp` / `SSweibull` / `SSasympOff` / `SSasympOrig`; population and group-level (`β+b`) bounds; scalar AGQ-in-θ. **Not yet:** multivariate AGQ for `k_re > 1`. |
| **Weighted GLMMs** (`glmer` + observation weights) | **~80%** | [`glmer_weighted`](src/lib.rs) mirrors [`lmer_weighted`](src/lib.rs). **Not yet:** frequency weights / offsets combined with weights beyond what is documented and tested. |

> **Throughput vs MixedModels.jl** is scored in summary **row 13** (**100%**, Jul 2026 — tier-A cold `lmer()` &lt;1.0×) — not duplicated here.

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
