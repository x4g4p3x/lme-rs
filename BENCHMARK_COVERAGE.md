# Benchmark coverage

[Documentation](docs/README.md) · [Run the benchmarks](BENCHMARKS.md) · [Dashboard](https://x4g4p3x.github.io/lme-rs/benchmarks/) · [Numerical comparisons](comparisons/COMPARISONS.md)

Use this map to check whether a published result covers your workload.
Timings, numerical correctness, and supported API scope are separate kinds of
evidence. The presence of a benchmark does not establish a speed advantage.

## Harness tiers

| Suite | Coverage | Reference | Measures |
|:---|:---|:---|:---|
| **A · Fair fits** | 10 LMM + 2 GLMM cases | MixedModels.jl | Complete fit; optional Rust preparation and reuse |
| **B · Phase breakdown** | Crossed 20k, nested 10k, random intercept 10k | Julia evaluation counts | Setup, objective work, solver phases, post-fit cost |
| **C · Whole scripts** | 7 example cases in 4 languages | R, Julia, Python, Rust | Whole process, including imports and JIT |
| **D · Criterion** | Formula, design, fitting, prediction, inference, scaling | Rust baselines | Operation-level regression evidence |
| **E · External operations** | LMM, Orange NLMM, Satterthwaite, Kenward–Roger, Python binding | R and Rust | Selected fit/inference calls after setup |
| **Production Criterion** | Large rows/groups/slopes, bursts, prediction | Rust baselines | Optional larger workload suite |

## Tier A case catalog

The [dashboard](https://x4g4p3x.github.io/lme-rs/benchmarks/) carries current
measurements and their qualifications. This table describes the test coverage,
so historical speed ratios do not become stale claims here.

| Case | Data and structure | Mode | Special consideration |
|:---|:---|:---|:---|
| `sleepstudy_reml` | 180 rows; correlated random intercept/slope | REML | Canonical random-slope fixture |
| `sleepstudy_weighted_reml` | Same fixture with shared weights | REML | Objective normalization must agree |
| `penicillin_crossed_reml` | Real crossed random intercepts | REML | Multiple grouping factors |
| `pastes_nested_reml` | Real nested random intercepts | REML | Nested grouping structure |
| `random_intercept_10k` | Synthetic, 10,000 rows | ML | Small end of the scaling sweep |
| `random_intercept_50k` | Synthetic, 50,000 rows | ML | Same model family at larger scale |
| `random_intercept_100k` | Synthetic, 100,000 rows | ML | Large single-factor workload |
| `large_random_slopes_100k` | 100,000 rows; 2,000 groups | ML | Correlated intercept/slope, three variance parameters |
| `crossed_20k` | Synthetic, 20,000 rows | ML | Two crossed grouping factors |
| `nested_10k` | Synthetic, 10,000 rows | ML | Batch/cask-style nesting |
| `cbpp_binomial_ml` | Real binary response | ML / Laplace | GLMM objective equivalence is not automatically qualified |
| `grouseticks_poisson_ml` | Real count response | ML / Laplace | GLMM likelihood conventions differ |

All cases time complete fitting. Prepared-fit phases are available for the LMM
cases. Comparing a reused Rust design with a new Julia model remains diagnostic.

## Coverage outside the fair fit suite

| Area | Available evidence | Still missing |
|:---|:---|:---|
| Nonlinear mixed models | R golden fixtures; Orange fit timing | Broad matched timing across nonlinear means and difficult starts |
| GLMM adaptive quadrature | Rust Criterion and golden fixtures | Fair external throughput across matched quadrature settings |
| Inference | Rust Criterion; selected R Satterthwaite/KR timings | Broad cross-language inference timing |
| Estimated marginal means | Rust Criterion and R correctness fixtures | Matched external throughput |
| Prediction | Rust group/observation sweeps | Matched R/Julia throughput |
| Python bindings | Shared Rust engine; one direct binding timing | Broad binding and data-conversion overhead survey |
| Optimizer choice | Argmin/Basin tests and paired timings | Evidence must cover every workload used to justify a default change |

## Running benchmarks

The [benchmark guide](BENCHMARKS.md#run-the-suites) contains complete commands,
runtime requirements, interpretation rules, and publishing steps.

For a quick functionality check:

```sh
task benchmarks:preflight
python scripts/run_fair_rust_julia_benchmark.py --cases sleepstudy_reml --warmups 1 --repeats 2
```

Those short runs are smoke tests. Use at least two warmups and ten measured fits
for a timing assessment, then repeat independent sessions for optimizer decisions.

## What each completion row may claim

The [completion manifest](completion_manifest.json) is authoritative. Its locked
scopes, current evidence, and open gaps govern the generated report.

| Claim | Evidence required |
|:---|:---|
| An LMM throughput target is met | The specified current cases qualify numerically and satisfy their timing threshold |
| GLMM throughput is established | Matching GLMM settings and numerical objectives, not an LMM result |
| An inference workflow is faster | Timings for that operation, not the existence of its API |
| The full suite is competitive | Complete applicable coverage, including regressions and unqualified cases |

Older thresholds and results remain in [the measurement archive](BENCHMARK_HISTORY.md).
Do not edit completion percentages or narrow a commitment to fit a benchmark result.

## Maintenance

Add new cases when a workflow changes, retain raw dated results, record skipped
tools, and update links to current evidence. Run `task completion:check` whenever
the completion manifest or generated report changes.
