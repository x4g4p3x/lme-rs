# Usability guide

This document answers **“can I use this for my problem?”** — a different question from **“how much of the API exists?”**

For feature breadth and internal planning percentages, see [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md). That file tracks **coverage** (what is implemented). This file tracks **usability** (what is safe and practical to rely on).

**Last assessed:** 2026-07-22 · `lme-rs` / `lme_python` **0.2.0** (unreleased release candidate on `master`)

---

## Two kinds of “done”

| Question | Where to look | What it means |
|:---------|:--------------|:--------------|
| How much is implemented? | [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) | APIs shipped, docs written, tests added. Grows when features land. |
| Can I depend on it for my analysis? | **This file** | Whether your workflow is in scope, regression-locked, **fast enough for how you will call it**, and reasonable to trust on *your* data. |

A new API (for example `nlmer_with_mean`) increases **coverage**. It only increases **usability** if it solves a workflow you actually need, behaves predictably on real inputs, and does not make that workflow impractically slow.

---

## Usability has three legs

Coverage percentages in [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) mostly track **correctness and API surface**. In practice, “usable” also depends on **performance for your call pattern** and **field experience** (below).

| Leg | Question | If it fails |
|:----|:---------|:------------|
| **Correctness & scope** | Does the model fit and match a reference on cases like mine? | Wrong answers — not usable regardless of speed. |
| **Performance fit** | Is wall time acceptable for how often I will fit? | Correct but too slow — **not usable** for batch, interactive, or embedded Rust paths even when fine for a one-off analysis. |
| **Field experience** | Has this shape of problem been exercised beyond the repo fixtures? | Higher risk on odd data and formulas; validate before you rely on it. |

**Performance and usability are not separate.** Optimization work ([OPTIMIZATION.md](OPTIMIZATION.md), fair harness in [BENCHMARKS.md](BENCHMARKS.md)) is usability work for anyone who fits more than occasionally or embeds `lmer` in a Rust pipeline. A library that is correct but an order of magnitude slower than alternatives is a poor fit for those workflows — that is a usability failure, not merely a benchmark nit.

The traffic-light tables below combine scope and typical performance posture. When in doubt, benchmark on your RE layout and call frequency.

---

## Engineering validation vs field experience

`lme-rs` is **well tested in the repository**, but **young in production**.

### What we have (engineering validation)

- Golden parity fixtures against R `lme4` / related packages on named datasets and model shapes ([`tests/test_golden_parity.rs`](tests/test_golden_parity.rs), [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json))
- Cross-language comparison scripts and fixtures ([`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md))
- Broad Rust integration tests (40+ modules under [`tests/`](tests/)); Python pytest suite under [`python/tests/`](python/tests/)
- Documented limitations in [README.md](README.md) and [GUIDE.md](GUIDE.md)
- Release-oriented CI on version tags ([`CONTRIBUTING.md`](CONTRIBUTING.md), [`AGENTS.md`](AGENTS.md))

That is real assurance — not a prototype — but it is **assurance on fixed, curated cases**.

### What we do not have yet (field experience)

- A long history of diverse **production** deployments (services, pipelines, published studies) feeding back into the API
- Semver **1.0** stability; the crate is **0.1.x** and the API can still evolve
- Community scale comparable to `lme4` (issue volume, odd formulas, dirty data, idiosyncratic workflows)
- A guarantee that **your** formula and dataset will behave like the fixtures without you checking

**This is not a reason to avoid the library.** It is the normal profile of capable early-stage scientific software: strong internal QA, limited miles in the wild. The honest posture is:

> Validate on your data before you stake a decision on it. Compare key outputs to an R reference fit when possible. Treat 0.1.x as “capable, but still earning trust.”

That is cautious, not pessimistic.

---

## Recommended adoption pattern

1. **Match your model to a green or yellow workflow below** (not “does the function exist?”).
2. **Run a reference fit** in R `lme4` / `glmer` / `nlmer` on the same data (or use an existing fixture if your case is listed in [`comparisons/COMPARISONS.md`](comparisons/COMPARISONS.md)).
3. **Compare quantities you care about**: fixed effects, variance components, predictions, test statistics — not necessarily raw AIC/BIC for GLMMs (see [README.md](README.md)).
4. **Benchmark if fit time matters**, especially for crossed random effects or tight Rust loops; consider `prepare_lmer` / `fit_prepared` ([OPTIMIZATION.md](OPTIMIZATION.md)).
5. **Pin the version** in `Cargo.toml` / your Python environment and read [CHANGELOG.md](CHANGELOG.md) before upgrading.

---

## Workflow traffic light

Statuses are **practical**, not formal support tiers.

### Green — reasonable default for new work

| Workflow | Rust | Python | Notes |
|:---------|:----:|:------:|:------|
| OLS / `lm` from formula | ✓ | ✓ | [`lm_df`](src/lib.rs) / `lm()` |
| Standard LMM (`lmer`): random intercept, **random slopes**, nested, crossed | ✓ | ✓ | sleepstudy, dyestuff, pastes, penicillin patterns |
| LMM with weights, offset | ✓ | ✓ | Tested; compare to R on your data |
| REML / ML, population & conditional predict | ✓ | ✓ | |
| Satterthwaite / Kenward–Roger, Type I–III ANOVA, contrasts | ✓ | ✓ | Scoped to tested LMM shapes; see golden `pastes_cask` |
| Nested model LRT (`anova`) | ✓ | ✓ | |
| Group-preserving CV (`cv_grouped` / `cv_grouped_glmer`) | ✓ | ✓ | LMM + GLMM; population / response-scale OOF on held-out groups |
| Bootstrap refits (`boot_lmer` / `boot_glmer`) | ✓ | ✓ | LMM parametric & residual; GLMM parametric (incl. binomial trials); percentile CIs |
| GLMM: binomial / Poisson / gamma / gaussian | ✓ | ✓ | Coeffs & variance params; Laplace default; scalar AGQ via `n_agq ≥ 2`. Gamma: mean/dispersion golden (`gamma_dyestuff_log_laplace`); RE θ may hit the boundary vs lme4 |
| Profile CIs (`confint_profile` / `parms=`) | ✓ | ✓ | LMM/GLMM; sleepstudy vs R fixture; subset via `parms` |
| `confint` (Wald), `simulate`, robust SE | ✓ | ✓ | LMM-focused Wald/t paths; GLMM z / profile as documented |

**Caveat:** “Green” means **the repo exercises these paths seriously**. It does not mean every R formula variant works.

### Yellow — works, but check assumptions

| Workflow | Issue | What to do |
|:---------|:------|:-----------|
| Random-slopes LMM (e.g. `(Days \| Subject)`) | Fair-harness competitive on sleepstudy (~0.8× Julia cold `lmer()`); validate on your data | Compare to R; use `prepare_lmer` + `fit_prepared` for repeated fits — [BENCHMARKS.md](BENCHMARKS.md#fair-rust-julia-2026-07-09-random-slopes) |
| Crossed RE at scale in Rust hot loops | One-shot `lmer()` includes setup/post-fit overhead | Use `prepare_lmer` + `fit_prepared`; see [BENCHMARKS.md](BENCHMARKS.md) |
| GLMM non-canonical links, weights | Probit **and** cloglog golden-locked; weights via `cbpp_binomial_weighted` | Remaining link variants (e.g. poisson sqrt) implemented — validate if non-golden |
| Weights + formula `offset()` together | Supported (`glmer_weighted` + `offset()`); **not** a separate golden matrix | Validate on your data |
| `nlmer` built-in `SS*` means | Eleven built-ins (`SSlogis`…`SSgompertz`, **`SSpower`**, **`SSfpl`**, **`SSbiexp`**, **`SSweibull`**, **`SSasympOff`**, **`SSasympOrig`**); one grouping factor | Orange / synthetic golden cases; `SSpower` uses custom R `selfStart` for lme4 reference — not general `nlme` |
| Grouped calibration (`SSpower`, `a·x^b+c`) | `nlmer` + golden `sspower_synthetic_self_start`; optional population and group-level (`β+b`) bounds | Requires **x > 0**; pool sensors with `~ c\|sensor`. See [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md) |
| Independent `power2` per sensor (MATLAB / lmfit lane) | **Out of scope** for `lme-rs` core; use batch NLS (CPU/GPU) | Demo: [`examples/batch_sspower_cpu.rs`](examples/batch_sspower_cpu.rs); decision guide in [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md) |
| `nlmer_with_mean` (custom μ) | No R `selfStart` for arbitrary custom means; defaults are naive | Supply `start`; verify predictions |
| Scalar AGQ (`n_agq ≥ 2`) | Inside θ search for scalar RE; CBPP AGQ-7 golden | Same for `nlmer` scalar RE |
| Python bindings | Polars, pandas, or PyArrow `Table` accepted; Polars canonical internally | [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |

### Red — not a substitute yet

| Expectation | Reality |
|:------------|:--------|
| Drop-in replacement for all of `lme4` + `lmerTest` + `car` + `nlme` | Intentionally partial API |
| Arbitrary R formula edge cases | Wilkinson coverage is broad but not universal |
| Full `stats::SS*` / general nonlinear mixed modeling | Eleven built-ins (`SSlogis` … `SSgompertz`, **`SSpower`**, **`SSfpl`**, **`SSbiexp`**, **`SSweibull`**, **`SSasympOff`**, **`SSasympOrig`**) + custom means; `SSpower` is lme-rs / MATLAB-aligned, not R `stats` |
| Identical GLMM AIC/BIC / log-likelihood to R | Deviance omits data-dependent constants |
| “Proven in production” without your own validation | 0.1.x; limited public field track record |
| Competitive cold `lmer()` vs MixedModels.jl on **unmeasured** RE layouts | Tier-A fair harness meets **&lt;1.0×** Julia ([2026-07-16](benchmarks/fair-rust-julia-reference-2026-07-16-cold-fit-lt1.json)); benchmark exotic layouts locally |

---

## Assurance levels (what “tested” means)

| Level | Examples | Trust for your problem |
|:------|:---------|:-----------------------|
| **Golden parity** | Manifest in [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) — sleepstudy, dyestuff, pastes, cbpp, orange nlmer, etc. | High **if your case matches** (dataset shape, formula family, REML/ML) |
| **Integration tests** | [`tests/test_numerical_parity.rs`](tests/test_numerical_parity.rs), [`tests/test_glmm.rs`](tests/test_glmm.rs), nlmer suites | High for the pattern covered; does not generalize automatically |
| **Cross-language comparisons** | [`comparisons/`](comparisons/) scripts | Regression aids; some runs are manual or tag CI |
| **Documented only** | Mentioned in GUIDE with fewer tests | Validate yourself |
| **Not implemented** | Rows under “Largely unrealized” in [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) | Do not use |

Numerical parity is a **goal on covered workflows**, not a blanket warranty. See [README.md](README.md) (“Limitations and compatibility notes”).

---

## Rust vs Python

| Concern | Rust (`lme-rs`) | Python (`lme_python`) |
|:--------|:----------------|:----------------------|
| API breadth | Full formula + matrix paths | Formula mirror + `lm_matrix`; matrix `lm(y, x)` Rust-only |
| Data | `polars::DataFrame` | Polars / pandas / PyArrow `Table` (normalized to Polars IPC) |
| Maturity | Same engine | Same engine; stubs in [`python/lme_python.pyi`](python/lme_python.pyi) |
| When to prefer | Native pipelines, amortized `fit_prepared` / `fit_prepared_glmer`, `cv_grouped*`, embedding in Rust services | Notebooks, Polars-centric Python stacks; prepare/CV/boot/GLMM APIs exposed |

---

## Performance is part of usability

There is no sharp line between “analysis” and “throughput” use — only **how often you pay the fit cost** and **whether that cost fits your budget**.

| Call pattern | Performance bar | Typical `lme-rs` posture (2026-07-16) |
|:-------------|:----------------|:----------------------------------------|
| **One-off** fit, inspect, publish | Seconds are usually fine | Most green LMM/GLMM workflows are usable |
| **Interactive** exploration (many refits, tuning) | Multi-second fits feel broken quickly | Yellow for crossed RE via one-shot `lmer()`; `prepare_lmer` / `fit_prepared` improves this |
| **Batch / CV / bootstrap** (same formula, many fits) | Linear cost in repetitions; setup amortization matters | Use `prepare_lmer` / `prepare_glmer`, `boot_lmer` / `boot_glmer(..., n_jobs=…)`, or `cv_grouped` / `cv_grouped_glmer(..., n_jobs=…)`; see [OPTIMIZATION.md](OPTIMIZATION.md) and [GUIDE.md](GUIDE.md#bootstrap-refits-boot_lmer--boot_glmer) |
| **Embedded Rust service** (fits on the request path) | Latency SLOs are hard requirements | Benchmark your RE structure; random-slopes sleepstudy pattern is ~sub-ms hot fit on the reference workstation |

**Practical rule:** if correctness checks pass but the fit is too slow for how you will call the API, treat that workflow as **downgraded** (green → yellow, or yellow → red) until you have measured it or switched to an amortized path.

Before committing to a hot path, read [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md) for tier-A cases and run [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) with `--with-phases` on workloads that match your RE structure.

---

## Related docs

| Doc | Role |
|:----|:-----|
| [README.md](README.md) | Overview, quick start, limitations |
| [GUIDE.md](GUIDE.md) / [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md) | How to call APIs |
| [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) | What is regression-tested vs manual |
| [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) | Internal coverage map (not a usability score) |
| [BENCHMARK_COVERAGE.md](BENCHMARK_COVERAGE.md) / [BENCHMARKS.md](BENCHMARKS.md) / [OPTIMIZATION.md](OPTIMIZATION.md) | Tier-A cases, fit timing scope, engineering notes |
| [docs/CALO_CALIBRATION.md](docs/CALO_CALIBRATION.md) | Sensor calibration: independent batch NLS vs pooled `nlmer`; CUDA / lightcurve-fitting |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## Maintenance

When a workflow moves from yellow → green (new golden case, clearer docs, production feedback, **or performance that meets the intended call pattern**), update the tables and bump **Last assessed**. When adding features, update [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) separately — do not treat a higher completion percentage as automatic usability improvement.
