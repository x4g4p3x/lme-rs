# Contributor and agent pre-flights

Use the smallest validation tier that proves the change is sound, then run the extra checks required by the files you changed. Do not claim checks that were skipped.

## First: choose the required checks

| Change | Required local validation |
|---|---|
| Rust code | `task lint`, `task test:fast`; use `task rust` for cross-module or public-API changes |
| Python bindings | `task lint:python`, `task python` |
| CI, manifests, release tooling | `task preflight`; use `task ci` before a release or broad refactor |
| R / Julia comparison scripts | `task lint:comparisons`; use `task lint:comparisons:required` when the formatters are installed |
| Documentation or portable examples | `task docs:check` (includes dashboard JSON drift); use `task consumer:smoke` when install or example behavior changes |
| LMM throughput paths (`src/math.rs`, `src/optimizer.rs`) | Read [OPTIMIZATION.md](OPTIMIZATION.md) and run the applicable fair-harness cases |
| Completion score files: `README.md`, `REPO_COMPLETION_BY_AREA.md`, `completion_manifest.json`, or `scripts/ci/check_completion_score.py` | `task completion:check` |

For a change that crosses several rows, run every applicable check. Full Rust integration tests are `task test` (or `cargo test --locked`).

## Completion score policy

The completion headline in [README.md](README.md) and every percentage in [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) are generated claims, not values to edit by hand.

- [completion_manifest.json](completion_manifest.json) is the source of truth: it declares binary scope commitments, their weights, locked `scope` strings, evidence paths, and a `gap` for every incomplete criterion.
- `task completion:check` validates schema version 2 (names, scopes, gaps), evidence-path existence, score arithmetic, report rows, and the README headline.
- A completion item may be marked complete only when its **locked scope** is met and its stated evidence is current. Partial, stale, or substituted evidence earns zero.
- Do not raise a score merely because an API exists, a focused benchmark passes, or the stated scope has been narrowed or replaced. Do not edit `scope` to make an existing artifact pass. Update the manifest and supporting evidence in the same change.

`task ci` runs this check automatically. Run it directly whenever a completion-related file changes.

## Tooling architecture

`scripts/ci/lme_ci.py` is the cross-platform implementation used by `Taskfile.yml`, Lefthook, GitHub Actions, and the legacy `scripts/local_ci.*` wrappers. Keep new checks there rather than duplicating shell logic.

| Component | Role |
|---|---|
| [scripts/ci/lme_ci.py](scripts/ci/lme_ci.py) | Shared Python 3.10+ CI runner |
| [Taskfile.yml](Taskfile.yml) | Thin, user-facing aliases |
| [lefthook.yml](lefthook.yml) | Staged-file pre-commit checks and pre-push preflight |
| [mise.toml](mise.toml) | Pinned Rust, Python, `uv`, Lefthook, and Task |
| `uv` | Locked Python environment and Ruff invocation |

One-time setup:

```powershell
mise install
task setup
```

If tools are already installed, run `task hooks:install`.

## What hooks do

### Commit

Lefthook runs matching staged checks in parallel:

- Rust: format, then Clippy.
- Python: Ruff check and format with auto-staging.
- R / Julia comparisons: formatter when its runtime and formatter package are available.
- Cargo manifests: `cargo check --all-targets`.
- Benchmark inputs: Rust benchmark smoke.
- Repository metadata inputs: metadata dry run, and token verification when `REPO_ADMIN_TOKEN` is set.

The commit hook does not run the full Rust suite, Python bindings suite, or `pip-audit`.

### Push

The pre-push hook runs `task preflight`, which includes linting, `cargo check --workspace --all-targets --locked`, `cargo audit`, legal/provenance checks, and repository-metadata validation. It is not a substitute for `task ci` after broad changes.

Use `--no-verify` only when explicitly necessary; report the bypass and the checks not run.

## CI and release boundaries

GitHub Actions validation runs automatically on pull requests and `v*` tags, and can be manually dispatched. Ordinary non-PR branch pushes do not receive the hosted validation matrix automatically. A lightweight cache-prime workflow runs when Rust dependency inputs change on `master` and weekly so new PRs can reuse trusted dependency artifacts. Pull requests run the full matrix except the four ignored heavy production-load cases. The tag CI calls the crates.io workflow and dispatches the top-level PyPI workflow only after every validation job succeeds; the publishing workflows do not listen to tags independently. The PyPI workflow must remain top-level because Trusted Publishing attestations do not support a reusable publishing workflow.

External GitHub Actions are pinned to full commit SHAs, with the readable release line retained as a comment. Dependabot proposes grouped weekly pin updates; do not replace SHA pins with mutable tags. Dependency audits and libFuzzer smoke tests run weekly in addition to their release/manual entry points.

- `task ci` mirrors the core hosted flow: Rust tests, Python bindings, portable consumer examples, lint, all-targets check, legal checks, documentation/link verification, and the completion-score check.
- Hosted-only coverage includes the multi-OS matrix, Python 3.10–3.13, production-load gates, and `pip-audit`.
- After changing BLAS target tables or release workflows, run `task ci` locally or manually dispatch CI before tagging; macOS Apple Silicon BLAS is not exercised on Windows/Linux.
- Benchmark workflow coverage requiring R or Julia belongs in the tag/manual workflow. `task benchmarks:preflight` runs the Rust smoke plus the R smoke when R/lme4 is available.

For repository-metadata token issues, set `REPO_ADMIN_TOKEN` locally and run `task repo-metadata`; a hosted `401` means the Actions secret must be rotated.

## Command reference

| Command | Purpose |
|---|---|
| `task lint` | Rust format/Clippy plus Python Ruff |
| `task test:fast` / `task test` | Rust unit-only / full Rust suite |
| `task test:consolidated` | Hosted non-Linux single-binary integration suite, doctests, and example checks |
| `task rust` / `task python` | Rust-only CI slice / bindings build and pytest flow |
| `task preflight` | Pre-push checks: lint, check, audit, legal, metadata |
| `task ci` / `task ci:fast` | Core CI mirror / reuse the editable Python environment and skip the isolated-wheel pass |
| `task audit` / `task legal` | Security audit / provenance and license checks |
| `task completion:check` | Verify the manifest-derived completion score and published markers |
| `task docs:check` / `task consumer:smoke` | Validate docs/links/API examples / run clean-install Rust and Python workflows |
| `task benchmarks:site` | Regenerate `docs/benchmarks/data/latest.json` from checked-in reference JSON |
| `task benchmarks:fair-rust-julia` | Fair fit-only Rust vs MixedModels.jl timing when Julia packages are installed |
| `task benchmarks:perf-breakdown` | Rust phase timings against Julia optimizer evaluation counts |
| `task benchmarks:external-timings` | Fair-ish `nlmer`, post-fit inference, and Python FFI timings vs R when available |
| `task explorations` | Standalone native-parser AST, θ-grid, and MCP exploration examples |
| `task lint:comparisons` | Optional R/Julia comparison formatting check |

Run `python scripts/ci/lme_ci.py --help` for the complete command list. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor workflow and [RELEASING.md](RELEASING.md) for release steps.

## Cursor Cloud specific instructions

This repo is a pure Rust library (`lme-rs`) plus a PyO3/maturin Python binding (`lme_python`); there are no servers or long-running services. "Running the app" means building the crate and running examples/tests. See the command reference above for the canonical `task` aliases.

- Toolchain comes from `mise` (per [mise.toml](mise.toml): Rust stable, Python 3.11, `uv`, `task`, `lefthook`). The startup update script runs `mise install`. Interactive shells auto-activate mise via `~/.bashrc`. In a non-interactive shell where `mise`/`task`/`uv` aren't on `PATH`, run `eval "$($HOME/.local/bin/mise activate bash --shims)"` first (or invoke tools with `$HOME/.local/bin/mise exec -- <cmd>`).
- On x86_64 the core crate links Intel MKL statically via `ndarray-linalg`, so no `gfortran`/system BLAS is needed; the first `cargo build`/`task test:fast` compiles MKL and takes a few minutes (subsequent builds are cached). Examples must be run with `--release` (e.g. `cargo run --release --example sleepstudy`) — a debug build is very slow.
- Python bindings gotcha: `uv sync` / `task ci` / `task python` reinstall the venv and **uninstall** the maturin-built `lme_python` before rebuilding it. If you run a bare `uv sync` (or interrupt `task python` before its maturin step), `import lme_python` breaks; restore it with `cd python && uv run --no-sync maturin develop --release` (or re-run `task python`). Run Python examples from `python/` via `uv run --no-sync python examples/<name>.py`.
