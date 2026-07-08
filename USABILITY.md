# Usability guide

This document answers **“can I use this for my problem?”** — a different question from **“how much of the API exists?”**

For feature breadth and internal planning percentages, see [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md). That file tracks **coverage** (what is implemented). This file tracks **usability** (what is safe and practical to rely on).

**Last assessed:** 2026-07-08 · `lme-rs` / `lme_python` **0.1.8**

---

## Two kinds of “done”

| Question | Where to look | What it means |
|:---------|:--------------|:--------------|
| How much is implemented? | [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) | APIs shipped, docs written, tests added. Grows when features land. |
| Can I depend on it for my analysis? | **This file** | Whether your workflow is in scope, regression-locked, and reasonable to trust on *your* data. |

A new API (for example `nlmer_with_mean`) increases **coverage**. It only increases **usability** if it solves a workflow you actually need and behaves predictably on real inputs.

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
| Standard LMM (`lmer`): random intercept, nested, crossed | ✓ | ✓ | sleepstudy, dyestuff, pastes, penicillin patterns |
| LMM with weights, offset | ✓ | ✓ | Tested; compare to R on your data |
| REML / ML, population & conditional predict | ✓ | ✓ | |
| Satterthwaite / Kenward–Roger, Type I–III ANOVA, contrasts | ✓ | ✓ | Scoped to tested LMM shapes; see golden `pastes_cask` |
| Nested model LRT (`anova`) | ✓ | ✓ | |
| GLMM: binomial / Poisson / gamma (canonical links) | ✓ | ✓ | Coefficients & variance params; Laplace default |
| `confint`, `simulate`, robust SE | ✓ | ✓ | LMM-focused; not every GLMM edge case |

**Caveat:** “Green” means **the repo exercises these paths seriously**. It does not mean every R formula variant works.

### Yellow — works, but check assumptions

| Workflow | Issue | What to do |
|:---------|:------|:-----------|
| Random-slopes LMM (e.g. `(Days \| Subject)`) | Supported; less optimization work than intercept-only paths | Compare to R; benchmark if fitting many models |
| Crossed RE at scale in Rust hot loops | One-shot `lmer()` includes setup/post-fit overhead | Use `prepare_lmer` + `fit_prepared`; see [BENCHMARKS.md](BENCHMARKS.md) |
| GLMM non-canonical links, weights | Implemented; narrower test matrix | Golden checks where listed; validate otherwise |
| `nlmer` built-in `SS*` means | Subset of R `stats::SS*`; one grouping factor | Orange / synthetic parity cases; not general `nlme` |
| `nlmer_with_mean` (custom μ) | No R `selfStart` for custom means; defaults are naive | Supply `start`; verify predictions |
| Scalar AGQ (`n_agq ≥ 2`) | Applied at final θ, not inside optimizer | Same pattern as `glmer`; compare Laplace vs AGQ |
| Python bindings | Polars `DataFrame` required | [`python/PYTHON_GUIDE.md`](python/PYTHON_GUIDE.md) |

### Red — not a substitute yet

| Expectation | Reality |
|:------------|:--------|
| Drop-in replacement for all of `lme4` + `lmerTest` + `car` + `nlme` | Intentionally partial API |
| Arbitrary R formula edge cases | Wilkinson coverage is broad but not universal |
| Full `stats::SS*` / general nonlinear mixed modeling | Five built-ins + custom means |
| Identical GLMM AIC/BIC / log-likelihood to R | Deviance omits data-dependent constants |
| “Proven in production” without your own validation | 0.1.x; limited public field track record |
| Competitive cold `lmer()` on every RE layout vs MixedModels.jl | Improving; see [OPTIMIZATION.md](OPTIMIZATION.md) row in [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) |

---

## Assurance levels (what “tested” means)

| Level | Examples | Trust for your problem |
|:------|:---------|:-----------------------|
| **Golden parity** | Manifest in [`tests/data/golden_parity_manifest.json`](tests/data/golden_parity_manifest.json) — sleepstudy, pastes, cbpp, orange nlmer, etc. | High **if your case matches** (dataset shape, formula family, REML/ML) |
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
| Data | `polars::DataFrame` | Polars `DataFrame` (IPC across the FFI boundary) |
| Maturity | Same engine | Same engine; stubs in [`python/lme_python.pyi`](python/lme_python.pyi) |
| When to prefer | Native pipelines, amortized `fit_prepared`, embedding in Rust services | Notebooks, Polars-centric Python stacks, quick parity with Rust |

---

## Performance and “usable”

Usability for **analysis** (fit once, inspect, predict) is largely separate from usability for **high-throughput fitting** (millions of fits, tight latency budgets).

- **Analysis-style use:** LMM green workflows are usable today; yellow items need a timing check on your hardware.
- **Throughput-sensitive use:** Read [OPTIMIZATION.md](OPTIMIZATION.md) and run [`scripts/run_fair_rust_julia_benchmark.py`](scripts/run_fair_rust_julia_benchmark.py) on cases that resemble your RE structure before committing to `lme-rs` in a hot path.

---

## Related docs

| Doc | Role |
|:----|:-----|
| [README.md](README.md) | Overview, quick start, limitations |
| [GUIDE.md](GUIDE.md) / [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md) | How to call APIs |
| [comparisons/COMPARISONS.md](comparisons/COMPARISONS.md) | What is regression-tested vs manual |
| [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) | Internal coverage map (not a usability score) |
| [BENCHMARKS.md](BENCHMARKS.md) / [OPTIMIZATION.md](OPTIMIZATION.md) | Fit timing scope and engineering notes |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## Maintenance

When a workflow moves from yellow → green (new golden case, clearer docs, production feedback), update the tables and bump **Last assessed**. When adding features, update [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) separately — do not treat a higher completion percentage as automatic usability improvement.
