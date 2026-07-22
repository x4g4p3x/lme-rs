# Contributor and agent pre-flights

Use the smallest validation tier that proves the change is sound, then run the extra checks required by the files you changed. Do not claim checks that were skipped.

## First: choose the required checks

| Change | Required local validation |
|---|---|
| Rust code | `task lint`, `task test:fast`; use `task rust` for cross-module or public-API changes |
| Python bindings | `task lint:python`, `task python` |
| CI, manifests, release tooling | `task preflight`; use `task ci` before a release or broad refactor |
| R / Julia comparison scripts | `task lint:comparisons`; use `task lint:comparisons:required` when the formatters are installed |
| LMM throughput paths (`src/math.rs`, `src/optimizer.rs`) | Read [OPTIMIZATION.md](OPTIMIZATION.md) and run the applicable fair-harness cases |
| Completion score files: `README.md`, `REPO_COMPLETION_BY_AREA.md`, `completion_manifest.json`, or `scripts/ci/check_completion_score.py` | `task completion:check` |

For a change that crosses several rows, run every applicable check. Full Rust integration tests are `task test` (or `cargo test --locked`).

## Completion score policy

The completion headline in [README.md](README.md) and every percentage in [REPO_COMPLETION_BY_AREA.md](REPO_COMPLETION_BY_AREA.md) are generated claims, not values to edit by hand.

- [completion_manifest.json](completion_manifest.json) is the source of truth: it declares binary scope commitments, their weights, and evidence paths.
- `task completion:check` validates evidence-path existence, score arithmetic, report rows, and the README headline.
- A completion item may be marked complete only when its stated evidence is current. Partial or stale evidence earns zero.
- Do not raise a score merely because an API exists, a focused benchmark passes, or the stated scope has been narrowed. Update the manifest and supporting evidence in the same change.

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

GitHub Actions run automatically on pull requests and `v*` tags, and can be manually dispatched. Ordinary non-PR branch pushes do not receive the hosted matrix automatically. Pull requests run the full matrix except the four ignored heavy production-load cases. The tag CI calls the crates.io and PyPI workflows only after every validation job succeeds; the publishing workflows do not listen to tags independently.

External GitHub Actions are pinned to full commit SHAs, with the readable release line retained as a comment. Dependabot proposes grouped weekly pin updates; do not replace SHA pins with mutable tags. Dependency audits and libFuzzer smoke tests run weekly in addition to their release/manual entry points.

- `task ci` mirrors the core hosted flow: Rust tests, Python bindings, lint, all-targets check, legal checks, doctests, docs, and the completion-score check.
- Hosted-only coverage includes the multi-OS matrix, Python 3.10/3.12/3.13, production-load gates, and `pip-audit`.
- After changing BLAS target tables or release workflows, run `task ci` locally or manually dispatch CI before tagging; macOS Apple Silicon BLAS is not exercised on Windows/Linux.
- Benchmark workflow coverage requiring R or Julia belongs in the tag/manual workflow. `task benchmarks:preflight` runs the Rust smoke plus the R smoke when R/lme4 is available.

For repository-metadata token issues, set `REPO_ADMIN_TOKEN` locally and run `task repo-metadata`; a hosted `401` means the Actions secret must be rotated.

## Command reference

| Command | Purpose |
|---|---|
| `task lint` | Rust format/Clippy plus Python Ruff |
| `task test:fast` / `task test` | Rust unit-only / full Rust suite |
| `task rust` / `task python` | Rust-only CI slice / bindings build and pytest flow |
| `task preflight` | Pre-push checks: lint, check, audit, legal, metadata |
| `task ci` / `task ci:fast` | Core CI mirror / reuse the editable Python environment and skip the isolated-wheel pass |
| `task audit` / `task legal` | Security audit / provenance and license checks |
| `task completion:check` | Verify the manifest-derived completion score and published markers |
| `task benchmarks:fair-rust-julia` | Fair fit-only Rust vs MixedModels.jl timing when Julia packages are installed |
| `task benchmarks:perf-breakdown` | Rust phase timings against Julia optimizer evaluation counts |
| `task lint:comparisons` | Optional R/Julia comparison formatting check |

Run `python scripts/ci/lme_ci.py --help` for the complete command list. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor workflow and [RELEASING.md](RELEASING.md) for release steps.
