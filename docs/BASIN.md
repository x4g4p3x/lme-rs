# Choosing an optimizer

[Documentation](README.md) · [Benchmark guide](../BENCHMARKS.md) · [Dashboard](https://x4g4p3x.github.io/lme-rs/benchmarks/) · [Engineering](../OPTIMIZATION.md)

**Argmin is the default; Basin is opt-in.** Both serve the shared simplex-search
paths. Choose using evidence from the models you fit, including convergence and
boundary behavior as well as runtime.

| Choice | What it provides | How to select it |
|:---|:---|:---|
| Argmin | Existing shared Nelder–Mead backend | Normal build |
| Basin | Projected Nelder–Mead with boundary recovery | `--features basin` |

## Enable Basin

The development checkout provides a `basin` Cargo feature. Default builds use
Argmin. Enable the feature when building this checkout:

```sh
cargo test --locked --features basin
cargo run --release --locked --features basin --example sleepstudy
```

For a local path dependency:

```toml
lme-rs = { path = "../lme-rs", features = ["basin"] }
```

The feature selects Basin's projected Nelder–Mead implementation for the
shared optimizer. This includes vector LMM and GLMM searches, the Nelder–Mead
refinement after the two-dimensional intercept grid search, and NLMM callers
of the shared helper. The specialized scalar and grid searches retain their
existing dispatch rules. There is no runtime optimizer selector, and enabling
the feature in any dependency enables it for the unified Cargo build.

## Bounds and numerical controls

| Control | Meaning with either backend |
|:---|:---|
| `tolerance` | Strict sample standard deviation threshold for simplex costs |
| `max_iterations` | Iteration budget per search stage, not an objective-call budget |
| `require_convergence` | Fail when the fit does not meet its convergence contract |


Basin projects simplex vertices onto the supplied lower bounds before each
objective evaluation. LMM and GLMM Cholesky diagonals are nonnegative, and
off-diagonal parameters remain unbounded. NLMM's existing additional parameter
constraints still apply inside its objectives. The initial simplex uses the
existing `0.2` coordinate steps around a feasible starting point.

`FitControl::tolerance` keeps its existing meaning: the sample standard deviation
of the simplex objective values must be strictly below the tolerance. A private
termination criterion implements this rule for Basin; it does not use Basin's
position-and-cost simplex tolerance. The default is `1e-6`.

`max_iterations` remains an iteration budget per search stage, rather than an
objective-evaluation budget. Exhausting it produces an unsuccessful convergence
flag. `require_convergence` retains its error behavior. Invalid objective values
cannot establish convergence, and objective errors propagate to the caller.
As before, reported iteration totals can include grid evaluations or several
search stages.

Projected Nelder–Mead can collapse a simplex onto a boundary face. Boundary and
singular-fit tests therefore accompany the ordinary fixture tests. This feature
does not imply that Basin is faster or more accurate for every model.

For LMMs with correlated random effects, recovery uses the actual Cholesky block
layout. At a near-zero diagonal it tests inward moves in both equivalent column
orientations, so projection cannot hide the other covariance direction. A
successful probe starts a smaller simplex and its cost is reused once. Recovery
requires improvement beyond the requested cost tolerance and floating-point
roundoff, and shares the original iteration budget. GLMM and nonlinear searches
retain their existing boundary recovery.

The [covariance-recovery report](../benchmarks/basin-covariance-recovery-2026-09-11.md)
records the corrected R-verified fit and the additional cost of checking some
boundary faces. It does not establish readiness to switch the default.

## Validation and comparison

The [11 September full refresh](../benchmarks/refresh-2026-09-11.md) retains
Argmin as the default. All 12 paired workloads reproduce their measured fits,
but the predeclared 5% runtime margin is not established across complete and
prepared fits. The report separates inconclusive timing from numerical failures
and includes the broader operation and production suites.

The current default decision is assessed against complete, order-balanced
measurements. Before switching the default, require matching fits, boundary and
budget regressions, competitive runtime across affected model families, and
validation on supported platforms. A small difference in one timing run is
insufficient. Full-suite evidence and remaining gaps are recorded in the
[benchmark guide](../BENCHMARKS.md).

The earlier [contributor comparison](../benchmarks/basin-optimizer-2026-09-10.md)
contains paired measurements and validation results for Basin 1.10.0 and Argmin
on its recorded upstream revision.

`task basin:check` runs feature-enabled Clippy, all-target checks, tests, and
documentation. Ubuntu CI and `task ci` run it alongside the default backend.

For a reproducible alternating comparison of both builds:

```sh
python scripts/run_optimizer_comparison.py --output benchmark-results/optimizer-comparison.json
```

For individual fair-harness runs:

```sh
python scripts/run_fair_rust_julia_benchmark.py --implementations rust \
  --warmups 3 --repeats 11 --with-phases --output benchmark-results/argmin.json
python scripts/run_fair_rust_julia_benchmark.py --implementations rust \
  --rust-features basin --warmups 3 --repeats 11 --with-phases \
  --output benchmark-results/basin.json
```

Reports identify the compiled backend and include fitted parameters, objective,
convergence, and iterations from an untimed fit. Timing samples exclude data
loading and numerical reporting. Both commands use identical fixture recipes.
Each measured cold and prepared fit also retains its own `fit_checks` or
`prepared_fit_checks`, extracted after timing. Rust/Julia accuracy and speed
qualifications use these per-fit checks, rather than the extra untimed summary.
`objective` comes from the optimizer diagnostics; `evaluated_objective` comes
from the final fitted model's deviance calculation. Compare both when checking
numerical results.
When using `--skip-rust-build`, pass matching `--rust-features`; the driver rejects
a backend mismatch.

Use the `bench_perf_breakdown` example, built with the same feature selection,
to collect LMM deviance-evaluation counts separately. Its counts include the
documented post-fit evaluations and are distinct from optimizer iterations.
Keep `LME_PERF_DIAG` disabled for timing runs.
