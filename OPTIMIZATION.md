# Performance engineering

[Documentation](docs/README.md) · [Benchmark guide](BENCHMARKS.md) · [Basin](docs/BASIN.md) · [Experiment archive](OPTIMIZATION_HISTORY.md)

This guide is for changes to fitting, model preparation, caches, and numerical
solvers. Start with the cost you measured, preserve the statistical contract,
and verify the affected workloads before claiming an improvement.

> **Correctness comes first.** A fit that stops early or returns the wrong
> objective is not an optimization. Keep convergence failures visible.

## Find the relevant path

| Work | Start here | What is reused |
|:---|:---|:---|
| Formula and design construction | `src/formula.rs`, `src/model_matrix.rs` | Parsed structure and sparse group layout |
| Prepared LMM fits | `src/prepared.rs`, `src/math.rs` | Immutable design plus response workspaces |
| Intercept-only LMM algebra | `src/math.rs`, `src/intercept_blocked.rs` | Gram products, symbolic structure, solver buffers |
| Single-factor random slopes | `SingleFactorSlopesCache` in `src/math.rs` | Group blocks and sparse factorization structure |
| Variance-parameter search | `src/optimizer.rs` | Shared objective and specialized search paths |
| Optional Basin optimizer | `src/optimizer/basin_backend.rs` | Projected simplex search and boundary recovery |
| GLMM / nonlinear models | `src/glmm_math.rs`, `src/nlmm/` | Model-specific workspaces and structured solves |

## Architecture

An LMM fit prepares the design, evaluates an objective repeatedly while
optimizing variance parameters, then constructs the returned fit. These phases
have different costs; measuring only one does not establish a whole-fit speedup.

| Phase | Contract |
|:---|:---|
| Preparation | Build the same design matrix, labels, weights, and offsets as the public API |
| Objective evaluation | Agree with the independent/full deviance path; handle infeasible parameters without panicking |
| Optimization | Respect bounds, tolerance, iteration budgets, and convergence reporting |
| Post-fit work | Return consistent coefficients, covariance, predictions, and diagnostics |

The intercept-only paths reuse sparse LDL structure and, where the block-size
gate permits it, blocked augmented Cholesky. A single-factor correlated slope
model can use `SingleFactorSlopesCache`. Unsupported structures retain the
general fallback. Preserve the gate and the fallback when changing a fast path.

The specialized scalar search and two-dimensional grid remain separate from
the shared Nelder–Mead helper. Both ML and REML two-dimensional searches include
numerical refinement: the historical grid-only ML shortcut is no longer the
contract. Argmin is the default shared optimizer; Basin is optional.

## Repeated fits

<a id="setup-amortization-prepare_lmer--fit_prepared"></a>

Use `prepare_lmer` when the formula and data structure are reused. For changing
responses, retain a workspace per worker:

```rust,ignore
let prepared = lme_rs::prepare_lmer("Reaction ~ Days + (Days | Subject)", &data)?;
let mut workspace = prepared.workspace();
let control = lme_rs::FitControl::default();
let fit = workspace.fit_response(response, true, &control)?;
```

The immutable design can be shared; mutable response work belongs to its worker.
The library respects the caller's Rayon context. Configure BLAS/OpenMP thread
counts before starting the process, and measure the thread configuration you
actually intend to deploy.

A prepared Rust fit and a newly constructed Julia model have different timing
boundaries. Report the reuse benefit directly; do not present that ratio as an
equivalent cold-fit comparison.

## Performance diagnostics (`LME_PERF_DIAG`)

Run the phase-breakdown harness separately from timing measurements:

```sh
python scripts/run_perf_breakdown.py --cases crossed_20k,nested_10k,random_intercept_10k
```

| Field | Use it to answer |
|:---|:---|
| `lmer_setup` / preparation report | Is formula/design construction dominant? |
| `lmer_optimize` | Is the search doing too much work? |
| `deviance_eval_count` | How many objective calls were made? |
| `mean_deviance_eval_seconds` | Are individual evaluations expensive? |
| `lmer_post_fit` | Is constructing the returned result expensive? |
| `blocked_*` and sparse-LDL phases | Which linear-algebra operation dominates? |

Reported objective evaluations can include post-fit diagnostic work. They are
not interchangeable with optimizer iterations. Keep `LME_PERF_DIAG` disabled
for throughput runs, and inspect the compiled backend label in the report.

## Contributor checklist

1. Reproduce the issue with a focused test or benchmark. Record the revision,
   compiler, inputs, optimizer, and thread settings.
2. Preserve formulas, ML/REML, families, weights, offsets, constraints, and
   convergence requirements. Do not loosen fixture tolerances to gain speed.
3. Run the required checks in [AGENTS.md](AGENTS.md). For optimizer or LMM math
   changes, run release golden parity and the affected fair-harness cases.
4. Check hot-path deviance against the independent full evaluation, including
   infeasible parameters and boundary/singular fits.
5. Compare complete fits as well as the changed phase. Use warmed runs,
   independent sessions, and reversed execution order.
6. Record regressions, inconclusive comparisons, and unmet targets alongside
   improvements. Update the changelog and current evidence links.

```sh
cargo test --release --locked --test test_golden_parity
python scripts/run_fair_rust_julia_benchmark.py --cases crossed_20k,nested_10k,random_intercept_10k --warmups 3 --repeats 11 --with-phases
```

For Basin changes, repeat the relevant Rust tests and benchmarks with
`--features basin` / `--rust-features basin` respectively.

## Choosing the default optimizer

The [Basin guide](docs/BASIN.md) records the current decision and evidence.
Evaluate numerical agreement, difficult starting points, bound behavior,
iteration-budget semantics, objective-evaluation counts, and runtime together.
Changing the default also affects consumers that do not choose Cargo features.

Similar medians from a single process are insufficient. Require repeated,
order-balanced measurements, no material regressions across affected LMM,
GLMM, and nonlinear workloads, and supported-platform validation. A faster
specialized scalar path is not evidence about the shared simplex optimizer.

## Lessons from earlier experiments

The [archive](OPTIMIZATION_HISTORY.md) retains measurements and rejected
approaches. Particularly useful cautions are:

- Unbounded dense assembly can dominate a nominally faster solver.
- A grid result needs numerical refinement before claiming convergence.
- A blocked solver needs a memory/shape gate and a tested fallback.
- An optimizer cache does not automatically satisfy post-fit solve contracts.
- More objective evaluations can erase a cheaper evaluation kernel.

Historical figures are evidence about their recorded revision. Use the
[current benchmark guide](BENCHMARKS.md) to reproduce present behavior.
