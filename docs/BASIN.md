# Optional Basin optimizer

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

## Validation and comparison

The [comparison report](../benchmarks/basin-optimizer-2026-09-10.md)
contains paired measurements and validation results for Basin 1.10.0 and Argmin
on the current upstream base.

`task basin:check` runs feature-enabled Clippy, all-target checks, tests, and
documentation. Ubuntu CI and `task ci` run it alongside the default backend.

Compare both builds with the existing fair harness:

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
`objective` comes from the optimizer diagnostics; `evaluated_objective` comes
from the final fitted model's deviance calculation. Compare both when checking
numerical results.
When using `--skip-rust-build`, pass matching `--rust-features`; the driver rejects
a backend mismatch.

Use the `bench_perf_breakdown` example, built with the same feature selection,
to collect LMM deviance-evaluation counts separately. Its counts include the
documented post-fit evaluations and are distinct from optimizer iterations.
Keep `LME_PERF_DIAG` disabled for timing runs.
