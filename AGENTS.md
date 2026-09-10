# Agent guide

This is a Rust statistical library (`lme-rs`) with PyO3/maturin Python bindings
(`lme_python`). There is no application server to start. Run commands from the
repository root unless stated otherwise.

## Start here

1. Inspect `git status --short --branch` and the relevant diff. Preserve unrelated
   changes; do not reset, stash, switch branches, or update dependencies as routine setup.
2. Use `rg` and the map below to find the implementation and existing tests. Read
   relevant guides rather than scanning the whole repository before a focused fix.
3. Choose required checks before editing. Reproduce bugs with focused tests, fix
   the cause, and verify both the regression and affected behavior.
4. Update [CHANGELOG.md](CHANGELOG.md) under `Unreleased` for user-visible fixes,
   features, and compatibility changes. Instruction-only edits need no entry.
5. Review the final diff. Report changes, validation, and failures/skips/limitations;
   do not claim unrun checks.

## Code map

| Area | Start with |
|---|---|
| Public API, contracts, reusable fits | `src/lib.rs`, `src/model.rs`, `src/prepared.rs` |
| Formulas and design matrices | `src/formula.rs`, `src/model_matrix.rs`, `src/basis.rs` |
| LMM / GLMM / nonlinear fitting | `src/math.rs`, `src/optimizer.rs`, `src/intercept_blocked.rs` / `src/glmm_math.rs`, `src/family.rs` / `src/nlmm/` |
| Inference and post-fit behavior | `src/contrast.rs`, `src/ddf.rs`, `src/satterthwaite.rs`, `src/kenward_roger.rs`, `src/kr_modcomp.rs`; named modules such as `predict.rs`, `simulate.rs`, `emmeans.rs`, `mcp.rs`, `robust.rs` |
| Python API and tests | `python/src/lib.rs`, `python/tests/` |
| Regression tests and reference evidence | `tests/`, `tests/data/`, `comparisons/` |
| Checks, aliases, hooks, tool versions | [scripts/ci/lme_ci.py](scripts/ci/lme_ci.py), [Taskfile.yml](Taskfile.yml), [lefthook.yml](lefthook.yml), [mise.toml](mise.toml) |

## Required validation

Use the smallest tier covering the change. Requirements from multiple rows combine;
a successful broader check can satisfy its component checks.

| Change | Minimum checks and escalation |
|---|---|
| Rust code | `task lint`, `task test:fast`, and affected integration tests; use `task rust` for cross-module or public-API changes |
| Python bindings | `task lint:python`, `task python` |
| CI, manifests, release tooling | `task preflight`; use `task ci` before a release or broad refactor |
| R / Julia comparison scripts | `task lint:comparisons`; use `task lint:comparisons:required` when formatters are installed |
| Documentation (including this file) or portable examples | `task docs:check`; add `task consumer:smoke` when install or example behavior changes |
| LMM throughput paths (`src/math.rs`, `src/optimizer.rs`, related caches/solvers) | Read [OPTIMIZATION.md](OPTIMIZATION.md) and run applicable fair-harness cases in addition to Rust checks |
| `README.md`, `REPO_COMPLETION_BY_AREA.md`, `completion_manifest.json`, `scripts/ci/check_completion_score.py` | `task completion:check` |

### Avoid redundant work

- Start with `cargo test --locked --test <test_target> <test_filter>` or
  `cargo test --locked --lib <test_filter>`. Selecting zero tests proves nothing.
- `task rust` includes Rust lint, all-target compilation, unit/integration tests,
  doctests, and documentation generation. Add `task lint:python` for the Ruff part
  of `task lint`; do not separately repeat `task test:fast`.
- `task test` runs the full Rust suite. On Windows/macOS, `task test:consolidated`
  runs unit and integration tests with one integration executable, checks examples,
  and runs doctests. It can replace `task test` when linking is costly. An equivalent
  Rust validation slice is `task lint:rust`, `task check`, `task test:consolidated`,
  and `python scripts/ci/lme_ci.py doc`.
- `task ci` includes Rust, bindings, portable examples, lint, compilation, legal,
  documentation, and completion checks. It does not replace audits/metadata in
  `task preflight`, R/Julia checks, or performance harnesses. `task ci:fast` skips
  the isolated-wheel pass; report that limitation.
- Reuse passing results for unchanged files/dependencies in the same task. Rerun
  affected checks after edits; broaden when scope or evidence requires it.
  Keep hooks enabled even when they repeat earlier checks.
- Serialize Cargo commands sharing a target directory. Capture verbose logs and
  inspect exit status and summaries. A timeout or compilation in progress is not a failed test.

## Correctness and evidence

- Add regression coverage for bug fixes. Prefer statistical identities, explicit
  expected behavior, and independent fixtures over duplicating the implementation.
- Register new integration test files in [tests/ci_consolidated.rs](tests/ci_consolidated.rs).
- For R/Julia parity, match rows, formula, family/link, weights, offsets, and ML/REML.
  Explain tolerances; do not loosen them just to make a failure pass.
- Tie benchmark claims to the revision, environment, and measured cases. Follow
  [BENCHMARKS.md](BENCHMARKS.md) and [OPTIMIZATION.md](OPTIMIZATION.md); a focused
  speedup does not establish general parity or completion.

### Completion scores are generated claims

[completion_manifest.json](completion_manifest.json) governs the README headline
and all percentages in [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md).
It declares weighted binary commitments, locked `scope` strings, evidence paths,
and a `gap` for every incomplete criterion.

- Complete an item only when its locked scope is met and evidence is current.
  Partial, stale, or substituted evidence earns zero; an API or focused benchmark is insufficient.
- Do not narrow/replace `scope` to earn credit or edit generated percentages by hand.
  Update the manifest and supporting evidence together.
- `task completion:check` validates schema version 2, names/scopes/gaps, evidence
  paths, arithmetic, report rows, and the README headline. `task ci` includes it.

## Setup and recovery

- [mise.toml](mise.toml) configures Rust stable, Python 3.11, uv, Task, and Lefthook.
  Use `task setup` for tools and hooks; run `mise install` first if Task is missing.
  With tools already installed, use `task hooks:install`. Pre-push audits also
  require `cargo-audit` (`cargo install cargo-audit`).
- For missing PATH entries, use `mise exec -- task <name>`. Locate installed tools
  before reinstalling. Verify subprocess discovery in the actual environment with
  `python -c "import shutil; print(shutil.which('uv'))"` when necessary.
- Without Task, use `python scripts/ci/lme_ci.py <subcommand>` (`python3` where
  appropriate). Mappings are in [Taskfile.yml](Taskfile.yml): `test:fast` → `test-fast`,
  `rust` → `rust-all`, `lint:python` → `ruff-lint`, `docs:check` → `docs-check`.
  Use `--help` for all commands. Implement new checks in this runner, not duplicate shell logic.
- For Git dubious ownership, use `git -c safe.directory=C:/path/to/lme-rs <command>`
  with this checkout's absolute forward-slash path; do not trust all directories globally.
- x86_64 builds link static Intel MKL via `ndarray-linalg`; first builds can be slow.
  Run numerical examples with `--release`, e.g. `cargo run --release --locked --example sleepstudy`.
  Leave the CI runner's own build profiles unchanged.
- Windows linker errors `LNK1318`/`LNK1106` can indicate disk exhaustion. Check space
  and active builds first. Before cleaning, verify the target belongs to this checkout
  and no build uses it; avoid broad cache deletion.
- `uv sync` can uninstall the editable Python extension. Restore it from `python/`
  with `uv run --no-sync maturin develop --release`, or rerun `task python`.
  Run Python examples there with `uv run --no-sync python examples/<name>.py`.

## Git, hooks, and release boundaries

- Follow the user's commit/push scope; commit authorization alone does not authorize
  pushing. When already authorized, complete validation and delivery without asking again.
- For requested updates, fetch before comparing branches; fast-forward only when
  ancestry and local changes permit it. After pushing, verify HEAD, the tracking ref,
  and live remote branch SHA agree, and report remaining local changes.
- Lefthook checks matching staged files: Rust format/Clippy, Python Ruff, comparison
  formatting, manifests, benchmarks/dashboard, and metadata. Hooks may auto-stage
  formatting; inspect the result. They do not run full Rust/bindings suites or `pip-audit`.
- Pre-push runs `task preflight`: lint, all-target compilation, Cargo audits,
  legal/provenance, and metadata validation. Use `--no-verify` only when explicitly
  necessary; report the bypass and omitted checks.
- Hosted validation runs on PRs, `v*` tags, and manual dispatch, not ordinary branch
  pushes. Cache-prime is not validation. Hosted coverage adds the OS/Python matrix,
  production-load gates, and `pip-audit`; PRs skip the four heavy ignored cases.
- Inspect the exact failing CI job/log before changing code. Do not merge with a
  failing security audit. For metadata authentication, use `REPO_ADMIN_TOKEN` with
  `task repo-metadata`; a hosted `401` requires checking/rotating the Actions secret.
- Preserve full-SHA action pins and readable version comments. After BLAS target-table
  or release-workflow changes, run `task ci` or manually dispatch CI before tagging;
  Windows/Linux cannot validate macOS Apple Silicon BLAS.
- Tag CI gates crates.io publishing and dispatches PyPI only after validation succeeds.
  Keep PyPI top-level for Trusted Publishing attestations; publishing workflows must
  not independently listen to tags. Read [RELEASING.md](RELEASING.md) before releases.
- R/Julia benchmark workflow coverage belongs in tag/manual runs. `task benchmarks:preflight`
  includes Rust smoke and R smoke when R/lme4 is installed. Optional tool skips are not passes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for extended guidance and
[Taskfile.yml](Taskfile.yml) for less common commands.
