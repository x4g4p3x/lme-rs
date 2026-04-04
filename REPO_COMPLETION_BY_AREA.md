# Repository completion by area

This file gives **approximate completion percentages** for major parts of the `lme-rs` repository. The numbers are **judgment calls** for planning and transparency, not precise metrics (they are not derived from line coverage or a formal roadmap).

**Last assessed:** 2026-04-04.

**Versions checked:** `lme-rs` **0.1.6** (root [`Cargo.toml`](Cargo.toml)); Python extension **`lme_python` 0.1.6** ([`python/Cargo.toml`](python/Cargo.toml)).

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
| 2 | **Rust crate: GLMM** — [`glmer`](src/lib.rs), [`family`](src/family.rs), PIRLS in [`glmm_math`](src/glmm_math.rs), Laplace vs scalar AGQ (`n_agq`) | **80%** | Implemented and tested ([`tests/test_glmm.rs`](tests/test_glmm.rs)); README documents Laplace optimization vs AGQ-in-final-eval, deviance constants, and default links ([`README.md`](README.md) “Limitations”). |
| 3 | **Rust crate: formula & model matrices** — [`formula`](src/formula.rs), [`model_matrix`](src/model_matrix.rs) | **87%** | Broad Wilkinson + RE support; remaining gap is breadth of edge cases vs R, not missing baseline features ([`tests/test_formula.rs`](tests/test_formula.rs), [`tests/test_crossed_mock.rs`](tests/test_crossed_mock.rs), etc.). |
| 4 | **Rust crate: post-fit inference** — [`confint`](src/lib.rs), [`simulate`](src/lib.rs), [`with_robust_se`](src/lib.rs), [`with_satterthwaite`](src/lib.rs), [`with_kenward_roger`](src/lib.rs) | **86%** | Covered by targeted tests ([`test_confint_simulate.rs`](tests/test_confint_simulate.rs), [`test_robust.rs`](tests/test_robust.rs), [`test_satterthwaite.rs`](tests/test_satterthwaite.rs), [`test_kenward_roger.rs`](tests/test_kenward_roger.rs)); scope matches guides and comparisons, not every GLMM edge case. |
| 5 | **Rust crate: ANOVA & model comparison** — Type III: [`LmeFit::anova`](src/anova.rs); nested LRT: [`anova`](src/lib.rs) (`AnovaResult`) | **76%** | Type III only; 1-DoF rows for continuous terms, joint multi-DoF Wald rows for grouped categorical dummies ([`src/anova.rs`](src/anova.rs), [`CHANGELOG.md`](CHANGELOG.md) 0.1.5). Not Type II ANOVA or a full `car` / `lmerTest` superset. |
| 6 | **Python bindings** (`python/`, import `lme_python`) | **68%** | Top-level fitters + rich [`PyLmeFit`](python/src/lib.rs) mirror major Rust workflows; README states Rust exposes a **broader** surface ([`README.md`](README.md)). Not exposed from Python: e.g. matrix-only [`lm(y, x)`](src/lib.rs) without a formula. Tests live under [`python/tests/`](python/tests/) (run locally per [`CONTRIBUTING.md`](CONTRIBUTING.md)). |
| 7 | **Cross-language validation** — [`comparisons/`](comparisons/), JSON/CSV fixtures, Rust tests | **91%** | [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md) explains what is regression-tested vs manual; strong for listed fixtures, not universal ecosystem parity. |
| 8 | **Benchmarks** — [`benches/bench_math.rs`](benches/bench_math.rs), [`BENCHMARKS.md`](BENCHMARKS.md), [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) | **82%** | Criterion coverage is substantive but explicitly **not** comprehensive ([`BENCHMARKS.md`](BENCHMARKS.md) states this plainly); workflow runs on `v*` tags and `workflow_dispatch`, not every PR. |
| 9 | **CI, release, and repo automation** | **92%** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml): `cargo` build/test on Ubuntu/Windows/macOS; `fmt`, `clippy`, `doc` on Ubuntu. [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml): maturin wheels when `python/**` or `src/**` changes. [`.github/workflows/repo-metadata.yml`](.github/workflows/repo-metadata.yml): GitHub metadata. **Gap:** default Rust CI does **not** run `pytest` (Python validated via contributor workflow + wheel builds). |
| 10 | **End-user documentation** — [`GUIDE.md`](GUIDE.md), [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md), [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md), [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`RELEASING.md`](RELEASING.md), [`BENCHMARKS.md`](BENCHMARKS.md) | **90%** | README documentation map is accurate; some release links (e.g. benchmark artifacts version on README) may lag the current crate version—prefer [`CHANGELOG.md`](CHANGELOG.md) for history. |
| 11 | **Examples & optional demos** — Cargo `[[example]]` entries in [`Cargo.toml`](Cargo.toml) under `comparisons/`, [`python/examples/`](python/examples/), [`scripts/run_cross_language_benchmarks.py`](scripts/run_cross_language_benchmarks.py) | **76%** | Comparison binaries are first-class; plotting and cross-language scripts are useful but partly manual or environment-dependent. |
| 12 | **Experimental / exploratory code** — [`scripts/ast_explorations/`](scripts/ast_explorations/) | **35%** | Standalone Rust snippets; not wired into the crate or CI. Other `scripts/` helpers (benchmark drivers, R dumps) are **tooling**, not “library completion.” |

## Weighted “overall” (illustrative only)

Simple mean of the twelve percentages:  
(93 + 80 + 87 + 86 + 76 + 68 + 91 + 82 + 92 + 90 + 76 + 35) ÷ 12 = 956 ÷ 12 ≈ **79.7%**.

So the rough figure is **~80%** toward a hypothetical “full statistics stack with universal parity.” That target is **not** the project goal—[`README.md`](README.md) describes the crate as usable with **documented**, narrower scope than the full R ecosystem.

## Evidence pointers (verified)

| Topic | Primary sources |
|:------|:----------------|
| Scope and limitations | [`README.md`](README.md) (“Current status”, “Limitations and compatibility notes”) |
| Type III ANOVA (incl. categorical joint tests) | [`README.md`](README.md); [`src/anova.rs`](src/anova.rs) (`FixedEffectsAnovaResult`, `DdfMethod`) |
| Python vs Rust breadth | [`README.md`](README.md); [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |
| Numerical validation | [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md); [`tests/test_numerical_parity.rs`](tests/test_numerical_parity.rs); [`tests/test_glmm.rs`](tests/test_glmm.rs) |
| Rust workflows | [`GUIDE.md`](GUIDE.md) |
| CI layout | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| Integration tests | **22** Rust modules under [`tests/`](tests/) (including `categorical_anova_test.rs`; counted 2026-04-04) |

## Largely unrealized in this repository (≈0% or “not started”)

These are **major gaps** relative to a very large reference (full `lme4` + related R packages + full Python mirror). They are **not** scored in the summary table above because the table tracks how complete the **intended** shipped surface is, not every conceivable extension.

| Topic | Status | Notes |
|:------|:-------|:------|
| **Nonlinear mixed models** (`nlmer`-style nonlinear predictors) | **0%** | No API or docs in-repo; search finds no `nlmer` / NLMM workflow. |
| **Weighted GLMMs** (`glmer` + observation weights) | **0%** | [`lmer_weighted`](src/lib.rs) exists for LMMs; [`glmer`](src/lib.rs) has **no** parallel weights parameter. [`BENCHMARKS.md`](BENCHMARKS.md) lists weighted GLMM as a possible **future** workload. |
| **Publication-grade cross-language benchmark harness** | **0%** (as a product) | [`BENCHMARKS.md`](BENCHMARKS.md) states the repo does **not** provide a fully normalized, machine-locked harness for public speed claims; example-level timing exists, but not that system. |
| **`pytest` in default GitHub Actions** | **0%** | Rust CI does not run [`python/tests/`](python/tests/); see row 9 above. |

**Narrow / partial rather than “zero”** (already shipped, but incomplete vs a big reference):

- **Fixed-effects ANOVA:** Type **III** only ([`README.md`](README.md)). **Type II** tables and arbitrary user-defined contrasts are **not** implemented. Multi-df **joint Wald** rows for grouped categorical dummies **are** implemented ([`src/anova.rs`](src/anova.rs)); general multi-df designs beyond that are not.
- **GLMM families / links:** Public [`glmer`](src/lib.rs) + [`Family`](src/family.rs) enum cover binomial, Poisson, Gaussian, gamma with **default** links ([`README.md`](README.md)); arbitrary link choice per fit is not a documented first-class path.
- **Python ↔ Rust API:** Intentionally **partial** ([`README.md`](README.md)); matrix-only [`lm(y, x)`](src/lib.rs) and other Rust-only entry points are not in `lme_python`.

If you add a capability that moves an item off this list, delete or rewrite the row and adjust the relevant summary-table percentage.

## Maintenance

When a major capability lands or a limitation is removed, update the relevant row, bump **Last assessed**, and recompute the simple mean if you still report it. Re-check **Versions checked** against [`Cargo.toml`](Cargo.toml) and [`python/Cargo.toml`](python/Cargo.toml) on each release.
