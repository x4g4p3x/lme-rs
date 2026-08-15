# Repository completion by area

This file gives an **evidence-weighted implementation-coverage score** for major parts of the `lme-rs` repository. The scores are computed from the binary, weighted scope commitments in [`completion_manifest.json`](completion_manifest.json), not estimated from lines of code, test counts, or an informal impression.

**This is a coverage map, not a usability guide.** For “can I use this on my problem?” (workflows, validation posture, limited field experience), see **[USABILITY.md](USABILITY.md)**.

**Last assessed:** 2026-08-15 (reevaluation of the 2026-08-14 100% headline).

**Versions checked:** `lme-rs` **0.2.3-dev.0** (root [`Cargo.toml`](Cargo.toml)); Python extension **`lme_python` 0.2.3-dev.0** ([`python/Cargo.toml`](python/Cargo.toml)).

Repository completion is judged on **three** axes, not features alone:

1. **Correctness / lme4-aligned behavior** — golden tests, comparisons, documented scope.
2. **Shipped API surface** — Rust crate, Python bindings, docs, CI.
3. **Competitive fit throughput** — on the [fair Rust vs Julia harness](BENCHMARKS.md#fair-rust-vs-julia-reference-results), `lme-rs` should be **in the same ballpark as MixedModels.jl** on core LMM cases (random intercept, crossed, nested), not an order of magnitude slower. This axis is **usability for performance-sensitive workflows**, not a separate concern from [USABILITY.md](USABILITY.md). **Status (2026-07-22 LMM cases, still current):** the ten LMM tier-A `cold_fit` ratios passed the strict **&lt;1.0×** Julia gate ([full reference](benchmarks/fair-rust-julia-reference-2026-07-22-full-tier-a.json)). The two GLMM cases in that file predate later PIRLS/AGQ changes, so they do not lock the full-suite criterion.

## Scoring algorithm

`completion_manifest.json` is the source of truth. Each row is made of explicit scope commitments with a positive integer weight, a locked `scope` string, and a binary `complete` state. Incomplete commitments must declare a `gap`. A commitment earns all of its weight only when its locked scope is met and its stated evidence is current; missing, partial, stale, or **substituted** evidence earns zero. Do not mark a criterion complete by narrowing or replacing its locked scope.

```text
area score    = round(100 × completed weight in area / total weight in area)
repository    = round(100 × completed weight in all areas / total weight in all areas)
```

The repository score is therefore **not** the mean of rounded rows. `100%` on a row requires every commitment in that row to be complete. Run `python scripts/ci/check_completion_score.py` (or `task completion:check`) to verify the manifest, every table percentage, and the README headline together.

`task completion:check` also requires schema version 2: every area has a name, every criterion has a `scope`, and every incomplete criterion has a `gap`. File existence alone is not enough to keep a `complete: true` bit honest.

## How to read the percentages

| Range | Meaning |
|:------|:--------|
| **90–100%** | Every locked commitment in that row has current evidence; documented limitations inside the locked scope are narrow. |
| **75–89%** | Feature-complete for core workflows in that row; known gaps are declared on incomplete criteria. |
| **55–74%** | Usable and supported, but intentionally partial vs the locked scope. |
| **Below 55%** | Experimental, optional, or not a product goal—treated as auxiliary. |

These bands describe **coverage of locked commitments**, not production maturity. A repository score of 100% would mean every locked scope has current evidence, not that the library is finished, field-proven, or a drop-in `lme4` replacement.

## 2026-08-15 reevaluation

The 2026-08-14 headline **100% (236/236)** was an overstatement of the repository's own rules:

| Problem | What happened | Correction |
|:--------|:--------------|:-----------|
| Checker tautology | `complete: true` plus existing files produced 100% with no locked meaning per criterion. | Schema v2 requires `scope` / `gap`. |
| Thin stretch closeout | Area 7 “full R edge-case matrix” became eight LMM cases; area 4 “broad” VC coverage became one intercept-only sleepstudy golden; area 8 external timings shipped a 1×5 smoke with R Kenward–Roger skipped. | Those three criteria are incomplete until the locked scope is met. |
| Area 12 substitution | After fiasto removal, area 12 was honestly **0%**. Three new Cargo examples then restored **20/20**. | Standalone probes stay complete (weight 7). The weight-13 “integrated” criterion is incomplete: the same three examples are not a second crate-integrated surface. |
| Stale full-suite throughput | Jul 22 tier-A JSON includes GLMM cases, then Gamma PIRLS and multivariate AGQ landed. | LMM case criteria stay complete (`math.rs` / `optimizer.rs` unchanged). `current-strict-full-tier-a` is incomplete until a post-0.2.2 rerun. |
| NLMM omitted from the denominator | README presents nonlinear mixed models as a first-class capability, but the 236-unit map had no NLMM row (only a ~99% “vs ecosystem” aside). | New **row 14** scores the intended `nlmer` surface. Multi-group RE and nlmer post-fit inference remain incomplete. |
| Stale usability identity | [USABILITY.md](USABILITY.md) still said the crate was **0.1.x**. | Updated to **0.2.x** / 0.2.3-dev.0 in the same change. |

Real Aug 14 product work (Gamma θ parity, `poly`/`ns`/`y ~ .`, VC profile API, `glht`, LMM `emmeans`, Python `lm(y, x)`, PR CI, `docs:check`, consumer smoke) **stays complete**. This reevaluation does not take those features away; it stops counting partial or substituted evidence as 100%.

## Summary table

| # | Area | Completion | Notes |
|---|------|:----------:|-------|
| 1 | **Rust crate: linear & mixed (LMM)** — [`lm`](src/lib.rs) / [`lm_df`](src/lib.rs), [`lmer`](src/lib.rs), [`lmer_weighted`](src/lib.rs), REML/ML, [`predict`](src/lib.rs) variants | **100%** | Broad intended LMM surface with parity/e2e/goldens (incl. `dyestuff_intercept_reml`), REML/ML, weights, predict. LMM strict-target cases in the Jul 22 artifact remain current (`math.rs` / `optimizer.rs` unchanged). |
| 2 | **Rust crate: GLMM** — [`glmer`](src/lib.rs), [`glmer_weighted`](src/lib.rs), [`family`](src/family.rs), PIRLS in [`glmm_math`](src/glmm_math.rs), Laplace vs AGQ (`n_agq`) | **100%** | Binomial/poisson/gaussian/gamma; canonical + golden non-canonical links (probit, cloglog); weights; AGQ-in-θ for scalar, vector (product), and small-`q` joint RE (CBPP AGQ-7). Gamma log-link Dyestuff locks mean, residual φ, and RE θ. **Non-goals:** R-identical AIC/BIC and extra `stats` families. |
| 3 | **Rust crate: formula & model matrices** — [`formula`](src/formula.rs), [`model_matrix`](src/model_matrix.rs) | **100%** | Wilkinson + RE, two-way `a:b`, `log`/`sqrt`/`exp`, `I()`, `offset(log(x))`, orthogonal/raw `poly()`, df-based `ns()`, and `y ~ .`. Remaining R syntax (`ns(knots=)`, `cbind()`) is a documented non-goal of `formula-r-edge-cases`. |
| 4 | **Rust crate: post-fit inference** — [`confint`](src/lib.rs), [`confint_profile`](src/profile_ci.rs), [`simulate`](src/lib.rs), [`boot_lmer`](src/bootstrap.rs) / [`boot_glmer`](src/bootstrap.rs), [`with_robust_se`](src/lib.rs), [`with_satterthwaite`](src/lib.rs), [`with_kenward_roger`](src/lib.rs) | **92%** | Wald + profile β (`parms=`); sleepstudy intercept-only VC profile golden; GLMM VC is smoke-only. **Gap:** random-slopes VC golden, GLMM VC vs R, nlmer profile CIs. |
| 5 | **Rust crate: ANOVA & model comparison** — Type III: [`LmeFit::anova`](src/anova.rs); nested LRT: [`anova`](src/lib.rs) (`AnovaResult`) | **100%** | Type I/II/III; `linear_hypothesis`; joint multi-DoF Wald; Tukey/Dunnett [`glht`](src/mcp.rs); LMM [`emmeans`](src/emmeans.rs) (pastes R golden). Locked scope is that subset, not full `multcomp` / `emmeans`. |
| 6 | **Python bindings** (`python/`, import `lme_python`) | **100%** | Formula API parity including prepare/CV/boot, `glht`, `emmeans`, `nlmer`, and numeric [`lm(y, x)`](python/src/lib.rs). |
| 7 | **Cross-language validation** — [`comparisons/`](comparisons/), JSON/CSV fixtures, Rust tests | **96%** | Golden parity includes dyestuff LMM/Gamma, CBPP cloglog + AGQ-7, additional `SS*` means, sleepstudy profile CIs. **Gap:** the R edge-case matrix is eight LMM fits; `ns()` is fitted-value locked; no GLMM formula edge. |
| 8 | **Benchmarks (instrumentation)** — [`benches/bench_math.rs`](benches/bench_math.rs), [`BENCHMARKS.md`](BENCHMARKS.md), [`BENCHMARK_COVERAGE.md`](BENCHMARK_COVERAGE.md), fair Rust/Julia harness, [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) | **88%** | Criterion + tier-A fair harness are in place. **Gap:** the [2026-08-14 external JSON](benchmarks/external-nlmm-inference-python-timings-2026-08-14.json) is a smoke (R KR skipped, ~10ms R clock), not comparable nlmer/KR/Python timings. |
| 9 | **CI, release, and repo automation** | **100%** | [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py) via [`.github/workflows/ci.yml`](.github/workflows/ci.yml), Task, Lefthook. CI runs on pull requests and `v*` tags. SHA-pinned Actions; weekly audit and fuzz smoke. Heavy production-load cases stay tag/manual, which is outside this row's locked scope. |
| 10 | **End-user documentation** — [`GUIDE.md`](GUIDE.md), [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md), [`python/README.md`](python/README.md), [`USABILITY.md`](USABILITY.md), [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md), [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`RELEASING.md`](RELEASING.md), [`BENCHMARKS.md`](BENCHMARKS.md), [`OPTIMIZATION.md`](OPTIMIZATION.md) | **100%** | Guides cover the checked APIs. `task docs:check` is in pull-request/tag CI. USABILITY identity matches 0.2.x as of this assessment. |
| 11 | **Examples & optional demos** — Cargo `[[example]]` entries in [`Cargo.toml`](Cargo.toml) under `comparisons/`, [`python/examples/`](python/examples/), [`scripts/run_cross_language_benchmarks.py`](scripts/run_cross_language_benchmarks.py) | **100%** | `task consumer:smoke` runs the Rust sleepstudy workflow and portable Python examples. R/Julia and plotting demos remain optional. |
| 12 | **Experimental / exploratory code** — [`scripts/explorations/`](scripts/explorations/) | **35%** | Three Cargo examples exist (AST dump, θ-grid, MCP adjust). The weight-13 integrated criterion is **not** met by those same examples. This row is auxiliary. |
| 13 | **LMM fit throughput vs MixedModels.jl** — optimization to be **competitive** on fair harness cases | **75%** | Jul 22 LMM cases (including `crossed_20k`, `nested_10k`, `sleepstudy_reml`) still pass the strict gate. **Gap:** no current full tier-A artifact after GLMM fit-math changes. |
| 14 | **Rust crate: nonlinear mixed models** — [`nlmer`](src/nlmm/mod.rs), [`nlmer_with_mean`](src/nlmm/mod.rs), Python `nlmer` | **82%** | Eleven built-in `SS*` means, bounds, custom μ, scalar/vector AGQ, and named goldens. **Gaps:** single grouping factor only; no nlmer VC profile / emmeans path. |

## Overall completion

**Evidence-weighted overall: 91% (235/258 scope units).**

The Jul 22 LMM cold fits passed the strict target (ratios about **0.03× to 0.96×** Julia on that workstation). These are versioned engineering measurements, not machine-independent speed guarantees. Do not read 91% as “almost done with mixed models in general”; it is 235 of 258 **locked** commitments.

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
| CI layout | [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py); [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (pull-request / `v*` tag / manual triggers, locked dependencies, all-targets check, doctests, isolated-wheel tests, Python version matrix); [`.github/workflows/audit.yml`](.github/workflows/audit.yml); [`.github/workflows/fuzz-smoke.yml`](.github/workflows/fuzz-smoke.yml); [`AGENTS.md`](AGENTS.md) |
| nlmer means, AGQ, custom μ | [`src/nlmm/`](src/nlmm/); [`tests/test_nlmm_ssmicmen.rs`](tests/test_nlmm_ssmicmen.rs), [`tests/test_nlmm_sspower.rs`](tests/test_nlmm_sspower.rs), [`tests/test_nlmm_custom_mean.rs`](tests/test_nlmm_custom_mean.rs), [`tests/test_nlmm_agq.rs`](tests/test_nlmm_agq.rs); [`comparisons/nlmm_sspower.R`](comparisons/nlmm_sspower.R); [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |
| Integration tests | **53** Rust modules under [`tests/`](tests/) excluding the consolidated harness (including `test_nlmm_orange.rs`, `test_mcp.rs`, `test_emmeans.rs`, `test_r_edge_case_matrix.rs`, `test_explorations.rs`; counted 2026-08-15) |

## Gaps vs the full R ecosystem (not in the summary table)

The summary table scores the **locked shipped surface** of this repo. This section lists **extensions relative to the whole `lme4` / `nlme` / `car` stack** — things that are either **not product goals**, **partially shipped**, or **explicitly not started**. Percentages here are vs that **larger reference**, not “missing from `lme-rs`.”

### Partial — usable subset shipped; not a full ecosystem replacement

| Topic | vs ecosystem | Notes |
|:------|:-------------|:------|
| **Nonlinear mixed models** (`nlmer`-style) | **~70% vs `lme4::nlmer`; much lower vs `nlme`** | [`nlmer`](src/nlmm/mod.rs) + Python; eleven built-ins (`SSlogis` … `SSgompertz`, `SSpower`, `SSfpl`, `SSbiexp`, `SSweibull`, `SSasympOff`, `SSasympOrig`); population and group-level (`β+b`) bounds; AGQ-in-θ for scalar and vector RE on **one grouping factor**. No `nlme` correlation structures, `pdMat` classes, or multi-group random formulas. Summary **row 14** scores the intended `lme-rs` surface, not this ecosystem percentage. |
| **Weighted GLMMs** (`glmer` + observation weights) | **~90%** | [`glmer_weighted`](src/lib.rs) mirrors [`lmer_weighted`](src/lib.rs); golden `cbpp_binomial_weighted`. Formula `offset()` + weights is **supported** but not a separate golden matrix — validate combined use on your data. |
| **emmeans / multcomp** | **subset** | LMM reference-grid EMMs and Tukey/Dunnett `glht` are scored in row 5. Missing vs R: single-step `mvtnorm`, compact-letter display, formula offsets, GLMM response-scale marginalization. |

> **Throughput vs MixedModels.jl** is scored in summary **row 13**. The Jul 22 artifact still supports the three named LMM case criteria; it does not currently lock the full tier-A-including-GLMM criterion.

### Not started (≈0% as a product goal)

| Topic | Status | Notes |
|:------|:-------|:------|
| **Publication-grade cross-language benchmark harness** | **0%** | [`BENCHMARKS.md`](BENCHMARKS.md): fair fit-only harness exists for engineering, but there is **no** machine-locked, publication-normalized speed product for public claims. |

### Already in the summary table (do not list here as “unrealized”)

These are **shipped and scored above** — they are incomplete only vs a much larger reference (full `car` / `lmerTest`, every GLMM edge case, etc.):

- Fixed-effects ANOVA (Type I–III), contrasts, `linear_hypothesis`, Tukey/Dunnett `glht` — summary **row 5**
- GLMM families / links — summary **row 2**
- Python ↔ Rust formula API — summary **row 6**
- `nlmer` intended surface — summary **row 14**

If a gap closes (new golden case, new mean, benchmark product), update the relevant **locked scope** in [`completion_manifest.json`](completion_manifest.json) and bump **Last assessed**. Do not edit table percentages by hand.

## Maintenance

When a major capability lands or a limitation is removed, update the relevant commitment in [`completion_manifest.json`](completion_manifest.json) **without silently rewriting `scope`**, bump **Last assessed**, and run `task completion:check`. Re-check **Versions checked** against [`Cargo.toml`](Cargo.toml) and [`python/Cargo.toml`](python/Cargo.toml) on each release.

When fit throughput improves, re-run **all applicable tier-A cases** with [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py), commit a dated full-suite artifact under [`benchmarks/`](benchmarks/), and raise row **13** only when every applicable case meets the declared target. Do not use a selected-case rerun to certify the whole row.
