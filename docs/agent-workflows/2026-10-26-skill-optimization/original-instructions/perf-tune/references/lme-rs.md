# lme-rs performance tuning

Use this reference only in `lme-rs`. Read the checkout's current `AGENTS.md`,
`OPTIMIZATION.md`, `BENCHMARKS.md`, and relevant `Taskfile.yml` tasks. Their current
contracts and acceptance rules govern the work; dated measurements are historical.
Do not start an application server for this statistical library.

## Locate the cost

| Measured cost | Start with |
|---|---|
| Formula and design construction | `src/formula.rs`, `src/model_matrix.rs`, `src/basis.rs` |
| Preparation and repeated responses | `src/prepared.rs`, `src/model.rs`, `src/math.rs` |
| LMM objective and structured solves | `src/math.rs`, `src/intercept_blocked.rs` |
| Variance-parameter searches | `src/optimizer.rs`, `src/optimizer/basin_backend.rs` |
| GLMM / nonlinear fitting | `src/glmm_math.rs`, `src/nlmm/` |
| Inference and reference grids | The affected post-fit module, such as `src/emmeans.rs` |
| Python overhead | `python/src/lib.rs` and the local extension build |

`LME_PERF_DIAG` separates setup, optimization, objective evaluations, and post-fit
costs. Objective evaluation counts can include diagnostic work and are not
optimizer iterations. Collect diagnostic reports separately and keep diagnostics
disabled for throughput measurements. Check the compiled backend label.

## Choose the matching evidence

Use the fair harness for complete fits, Criterion for affected operations, and
phase reports for attribution. Read each driver's arguments before choosing cases;
do not automatically run every suite for a bounded change.

```sh
python scripts/run_perf_breakdown.py --cases crossed_20k,nested_10k,random_intercept_10k
cargo test --release --locked --test test_golden_parity
python scripts/run_fair_rust_julia_benchmark.py --implementations rust,julia --cases crossed_20k,nested_10k,random_intercept_10k --warmups 3 --repeats 11 --threads 1 --with-phases --output benchmark-results/fair-candidate.json
cargo bench --locked --bench bench_math -- --noplot
```

Choose affected cases instead of copying the example list regardless of scope.
Preserve a separate baseline report. The Rust/Julia ratio describes cross-library
performance; compare baseline Rust with candidate Rust to quantify the change.
Repeat fair comparisons in a separate session with `--order julia-first` when
making cross-library claims. Use named Criterion baselines when appropriate;
do not silently reuse evidence from another compiler or machine.

For optimizer backend decisions, use `scripts/run_optimizer_comparison.py` and
read `docs/BASIN.md`. The repository's driver balances backend order across
independent processes and applies its declared margin. Changing the default
requires broader correctness, convergence, workload, and platform evidence than
a faster specialized scalar path. Keep Argmin as the current default unless the
task authorizes and validates a default change. Basin tests and benchmarks must
use matching `--features basin` / `--rust-features basin` builds.

The current guide requires at least two warmups and ten measured samples for a
speed assessment; shorter runs are smoke tests. An interval spanning the target
is inconclusive. Honor stricter rules from the affected harness.

## Preserve statistical contracts

- Match rows, formulas, family/link, weights, offsets, ML/REML, tolerances, bounds,
  and convergence requirements. Coefficient agreement alone is not objective or
  uncertainty agreement. Retain visible failures and unqualified cases.
- Check hot-path deviance against the independent full evaluation, including
  infeasible parameters and boundary/singular fits. Do not restore grid-only or
  premature-stop behavior to reduce elapsed time.
- Keep structured-solver memory/shape gates and general fallbacks. Avoid unbounded
  dense assembly that hides a regression on large or sparse inputs.
- Share immutable prepared designs; keep mutable response workspaces per worker.
  Check different response vectors, weights/offsets, and fit controls where relevant.
  Respect the caller's Rayon context and set BLAS/OpenMP limits before process start.
- Report `cold_fit`, preparation, and `fit_prepared` separately. Prepared Rust
  versus newly constructed Julia is a reuse comparison, not equivalent cold fits.

## Required checks and delivery

- Rust changes require `task lint`, `task test:fast`, and affected integration
  tests; cross-module/public-API changes require `task rust` plus Python lint.
  Use the current guide's consolidated equivalent when Windows linking is costly.
- LMM math/optimizer changes additionally require release golden parity and
  applicable fair-harness cases. Binding changes require `task lint:python` and
  `task python`; benchmark tooling changes require the relevant benchmark tests
  and the current CI/tooling tier.
- Register new integration files in `tests/ci_consolidated.rs`. Add user-visible
  improvements under `Unreleased` in `CHANGELOG.md`. Documentation changes require
  `task docs:check`; completion-related files require `task completion:check`.
- Serialize Cargo commands sharing a target directory and keep timing runs free
  of competing work. Leave the CI runner's profiles unchanged. Inspect exit codes
  and actual test counts; compilation in progress is not a failed test.
- Use the repository's CI runner if Task is unavailable. Locate installed tools
  before reinstalling them. Verify the local Python extension corresponds to the
  candidate before measuring bindings; `uv sync` can remove an editable extension.
- Retain raw reports, identity metadata, failed cases, and skips. When updating
  maintained benchmark reports/dashboard inputs, follow `BENCHMARKS.md` and run
  `task benchmarks:test` and `task docs:check`. Do not rewrite old evidence or
  claim locked completion commitments from a partial benchmark.

This skill does not authorize changing optimizer defaults, publishing results to
an external service, worker delegation, or Git delivery beyond the user's scope.
