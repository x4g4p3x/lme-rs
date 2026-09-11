# Fit-quality-aware benchmark check — 2026-09-11

[Raw measurements and fit checks](fair-rust-julia-evidence-2026-09-11.json)
· [Methodology](../BENCHMARKS.md#evidence-rules-for-new-fair-harness-runs)

This is a six-case validation of the revised harness, not a complete throughput
baseline or a before/after compiler performance comparison. It does not replace
the completion manifest's reference evidence.

Measured on Windows, AMD Ryzen 5 8600G, Rust 1.98.1, Julia 1.12.6,
MixedModels.jl 5.7.0, with Julia OpenBLAS and the existing Rust static MKL build.
One worker/BLAS thread was requested for both implementations; Julia reports one
BLAS thread. Rust's configured MKL backend is sequential. Each runtime had two
warmups and ten measured fits. Julia ran first per case, with no concurrent Cargo
builds. The JSON records base revision `349e07f`, a dirty working tree, source and
binary hashes, and the shared CSV hashes.

| Case | Rust/Julia median | Within-process 95% interval | Assessment |
|:-----|------------------:|:---------------------------|:-----------|
| sleepstudy REML | 0.75× | 0.71–0.77× | Fits agree; Rust faster in this session |
| weighted sleepstudy REML | 0.68× | 0.57–0.70× | Unverified: objective values differ |
| random intercept, 10k rows | 0.70× | 0.46–0.94× | Unverified: objective/fixed effects differ |
| crossed, 20k rows | 2.32× | 2.21–2.39× | Fits agree; Julia faster in this session |
| nested, 10k rows | 1.79× | 1.61–1.87× | Fits agree; Julia faster in this session |
| CBPP binomial | 1.07× | 1.04–1.10× | GLMM objective equivalence is not established |

All twelve implementation/case executions completed. The three unverified
comparisons remain in the artifact but cannot count as speed wins or target
passes. Prepared Rust versus full Julia fit ratios are diagnostic only.

The weighted fit coefficients agree closely, but the raw objectives differ by
about 68.01. Objective normalization needs investigation before treating this as
a comparable likelihood or a numerical defect. The random-intercept objectives
differ by about 25.53 and the fixed effects also differ beyond the declared
bounds. Both solvers report convergence; convergence alone is insufficient.
No numerical-library behavior or tolerance was changed to hide either mismatch.

The bootstrap intervals describe repeated fits within each process. They do not
establish between-session stability or general hardware-independent superiority.

## Validation

- 13 benchmark evidence/reporting regressions passed.
- Rust 1.98.1: 127 unit tests, 306 integration tests, and six doctests passed;
  four heavy production tests remain intentionally ignored locally.
- Preflight passed: Rust/Python lint, all-target compilation, Cargo advisories,
  legal checks, and repository metadata dry-run.
- Documentation checks and benchmark dashboard data-drift validation passed.
- Julia comparison formatting and benchmark Python lint passed. R is not
  installed locally; the changed Ubuntu dependency-install step needs hosted
  validation. Workflow YAML parsing passed.
- Browser inspection confirmed unverified and inconclusive labels and interval
  text. The published dashboard and historical baseline were not replaced.

Reproduce this run:

```bash
python scripts/run_fair_rust_julia_benchmark.py --cases sleepstudy_reml,sleepstudy_weighted_reml,random_intercept_10k,crossed_20k,nested_10k,cbpp_binomial_ml --warmups 2 --repeats 10 --threads 1 --order julia-first --with-phases --output benchmark-results/fair-evidence.json
```
