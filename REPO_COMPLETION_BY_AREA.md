# Repository completion by area

This file gives **approximate completion percentages** for major parts of the `lme-rs` repository. The numbers are **judgment calls** for planning and transparency, not precise metrics (they are not derived from line coverage or a formal roadmap).

**Last assessed:** 2026-07-01.

**Versions checked:** `lme-rs` **0.1.8** (root [`Cargo.toml`](Cargo.toml)); Python extension **`lme_python` 0.1.8** ([`python/Cargo.toml`](python/Cargo.toml)).

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
| 1 | **Rust crate: linear & mixed (LMM)** — [`lm`](src/lib.rs) / [`lm_df`](src/lib.rs), [`lmer`](src/lib.rs), [`lmer_weighted`](src/lib.rs), REML/ML, [`predict`](src/lib.rs) variants | **93%** | Core path is stable; parity-style checks in [`tests/test_numerical_parity.rs`](tests/test_numerical_parity.rs), e2e and optimization tests under [`tests/`](tests/). |
| 2 | **Rust crate: GLMM** — [`glmer`](src/lib.rs), [`glmer_weighted`](src/lib.rs), [`family`](src/family.rs), PIRLS in [`glmm_math`](src/glmm_math.rs), Laplace vs scalar AGQ (`n_agq`) | **82%** | Implemented and tested ([`tests/test_glmm.rs`](tests/test_glmm.rs), [`tests/test_glmm_weighted.rs`](tests/test_glmm_weighted.rs)); README documents Laplace optimization vs AGQ-in-final-eval, deviance constants, and default links ([`README.md`](README.md) “Limitations”). |
| 3 | **Rust crate: formula & model matrices** — [`formula`](src/formula.rs), [`model_matrix`](src/model_matrix.rs) | **87%** | Broad Wilkinson + RE support; remaining gap is breadth of edge cases vs R, not missing baseline features ([`tests/test_formula.rs`](tests/test_formula.rs), [`tests/test_crossed_mock.rs`](tests/test_crossed_mock.rs), etc.). |
| 4 | **Rust crate: post-fit inference** — [`confint`](src/lib.rs), [`simulate`](src/lib.rs), [`with_robust_se`](src/lib.rs), [`with_satterthwaite`](src/lib.rs), [`with_kenward_roger`](src/lib.rs) | **86%** | Covered by targeted tests ([`test_confint_simulate.rs`](tests/test_confint_simulate.rs), [`test_robust.rs`](tests/test_robust.rs), [`test_satterthwaite.rs`](tests/test_satterthwaite.rs), [`test_kenward_roger.rs`](tests/test_kenward_roger.rs)); scope matches guides and comparisons, not every GLMM edge case. |
| 5 | **Rust crate: ANOVA & model comparison** — Type III: [`LmeFit::anova`](src/anova.rs); nested LRT: [`anova`](src/lib.rs) (`AnovaResult`) | **90%** | Type II/III; 1-DoF marginal KR; joint multi-DoF Wald; user contrasts [`test_contrast`](src/contrast.rs) ([`tests/test_contrast.rs`](tests/test_contrast.rs)). Not full `car` / `lmerTest` superset (e.g. `glht`, automatic Type I). |
| 6 | **Python bindings** (`python/`, import `lme_python`) | **95%** | Near-full Rust parity: formula `lm`/`lmer`/`glmer`/`nlmer`, matrix [`lm_matrix`](python/src/lib.rs), `contrast_matrix` / `contrast_matrix_from_names`, structured result types (`PyConfintResult`, `PyFixedEffectsAnova`, `PyLikelihoodRatioAnova`, …), [`lme_python.pyi`](python/lme_python.pyi). Remaining gaps: NLMM `predict`, custom `nlme` means, AGQ tuning from Python. [`python/tests/`](python/tests/) including [`test_api_parity.py`](python/tests/test_api_parity.py). |
| 7 | **Cross-language validation** — [`comparisons/`](comparisons/), JSON/CSV fixtures, Rust tests | **92%** | [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md) explains what is regression-tested vs manual; [`tests/test_golden_parity.rs`](tests/test_golden_parity.rs) now includes pastes / `cask` multi-DoF ANOVA; not universal ecosystem parity. |
| 8 | **Benchmarks** — [`benches/bench_math.rs`](benches/bench_math.rs), [`BENCHMARKS.md`](BENCHMARKS.md), [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) | **82%** | Criterion coverage is substantive but explicitly **not** comprehensive ([`BENCHMARKS.md`](BENCHMARKS.md) states this plainly); workflow runs on `v*` tags and `workflow_dispatch`, not every PR. |
| 9 | **CI, release, and repo automation** | **98%** | Single runner [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py) via [`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`Taskfile.yml`](Taskfile.yml), Lefthook, and legacy `local_ci` scripts; **`--locked`**, **`cargo check --all-targets`**, **`cargo test --doc`**; multi-OS; Python **3.11** + wheel reinstall + second **`pytest`**; **3.10 / 3.12 / 3.13** on Ubuntu. [`.github/workflows/audit.yml`](.github/workflows/audit.yml), [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml). [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml): maturin wheels. [`.github/workflows/repo-metadata.yml`](.github/workflows/repo-metadata.yml): GitHub metadata. |
| 10 | **End-user documentation** — [`GUIDE.md`](GUIDE.md), [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md), [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md), [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`RELEASING.md`](RELEASING.md), [`BENCHMARKS.md`](BENCHMARKS.md) | **90%** | README documentation map is accurate; some release links (e.g. benchmark artifacts version on README) may lag the current crate version—prefer [`CHANGELOG.md`](CHANGELOG.md) for history. |
| 11 | **Examples & optional demos** — Cargo `[[example]]` entries in [`Cargo.toml`](Cargo.toml) under `comparisons/`, [`python/examples/`](python/examples/), [`scripts/run_cross_language_benchmarks.py`](scripts/run_cross_language_benchmarks.py) | **76%** | Comparison binaries are first-class; plotting and cross-language scripts are useful but partly manual or environment-dependent. |
| 12 | **Experimental / exploratory code** — [`scripts/ast_explorations/`](scripts/ast_explorations/) | **35%** | Standalone Rust snippets; not wired into the crate or CI. Other `scripts/` helpers (benchmark drivers, R dumps) are **tooling**, not “library completion.” |

## Weighted “overall” (illustrative only)

Simple mean of the twelve percentages:  
(93 + 82 + 87 + 86 + 90 + 95 + 92 + 82 + 98 + 90 + 76 + 35) ÷ 12 = 1006 ÷ 12 ≈ **83.8%**.

So the rough figure is **~84%** toward a hypothetical “full statistics stack with universal parity.” That target is **not** the project goal—[`README.md`](README.md) describes the crate as usable with **documented**, narrower scope than the full R ecosystem.

## Evidence pointers (verified)

| Topic | Primary sources |
|:------|:----------------|
| Scope and limitations | [`README.md`](README.md) (“Current status”, “Limitations and compatibility notes”) |
| Type III ANOVA (incl. categorical joint tests) | [`README.md`](README.md); [`src/anova.rs`](src/anova.rs); [`src/ddf.rs`](src/ddf.rs); [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) (`pastes_cask_multi_dof_reml`) |
| Python vs Rust breadth | [`README.md`](README.md); [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |
| Numerical validation | [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md); [`tests/test_numerical_parity.rs`](tests/test_numerical_parity.rs); [`tests/test_glmm.rs`](tests/test_glmm.rs) |
| Rust workflows | [`GUIDE.md`](GUIDE.md) |
| CI layout | [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py); [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (`--locked`, all-targets check, doctests, `pytest tests/`, Python version matrix on Ubuntu); [`AGENTS.md`](AGENTS.md) |
| Integration tests | **32** Rust modules under [`tests/`](tests/) (including `test_nlmm_orange.rs`, `test_glmm_weighted.rs`, `test_contrast.rs`; counted 2026-07-01) |

## Largely unrealized in this repository (≈0% or “not started”)

These are **major gaps** relative to a very large reference (full `lme4` + related R packages + full Python mirror). They are **not** scored in the summary table above because the table tracks how complete the **intended** shipped surface is, not every conceivable extension.

| Topic | Status | Notes |
|:------|:-------|:------|
| **Nonlinear mixed models** (`nlmer`-style) | **~40%** | [`nlmer`](src/nlmm/mod.rs) + Python `nlmer()`. Not yet: custom nonlinear functions, multiple RE params, `selfStart`, AGQ, `predict()` on NL fits. |
| **Weighted GLMMs** (`glmer` + observation weights) | **~80%** | [`glmer_weighted`](src/lib.rs) mirrors [`lmer_weighted`](src/lib.rs); prior weights in PIRLS and deviance. Gaussian delegates to LMM. Not yet: frequency weights / offsets combined with weights in docs only. |
| **Publication-grade cross-language benchmark harness** | **0%** (as a product) | [`BENCHMARKS.md`](BENCHMARKS.md) states the repo does **not** provide a fully normalized, machine-locked harness for public speed claims; example-level timing exists, but not that system. |

**Narrow / partial rather than “zero”** (already shipped, but incomplete vs a big reference):

- **Fixed-effects ANOVA:** Type **II** and **III** ([`README.md`](README.md)). User-defined **q × p** contrasts via [`src/contrast.rs`](src/contrast.rs). Multi-df joint Wald for grouped categoricals and arbitrary `L` matrices use the same Satterthwaite / Kenward–Roger engines ([`src/ddf.rs`](src/ddf.rs), [`src/kr_modcomp.rs`](src/kr_modcomp.rs)). Type I and `car::linearHypothesis` convenience wrappers are not implemented.
- **GLMM families / links:** Public [`glmer`](src/lib.rs) + [`Family`](src/family.rs) enum cover binomial, Poisson, Gaussian, gamma with **default** links ([`README.md`](README.md)); arbitrary link choice per fit is not a documented first-class path.
- **Python ↔ Rust API:** Formula-based surface and [`lm_matrix`](python/src/lib.rs) are mirrored; low-level matrix [`lm(y, x)`](src/lib.rs) without a DataFrame remains Rust-only.

If you add a capability that moves an item off this list, delete or rewrite the row and adjust the relevant summary-table percentage.

## Maintenance

When a major capability lands or a limitation is removed, update the relevant row, bump **Last assessed**, and recompute the simple mean if you still report it. Re-check **Versions checked** against [`Cargo.toml`](Cargo.toml) and [`python/Cargo.toml`](python/Cargo.toml) on each release.
