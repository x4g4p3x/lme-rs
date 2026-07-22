# Contributing

## Scope

This repository contains:

- the Rust crate in the repository root
- the Python bindings in `python/`
- cross-language comparison scripts in `comparisons/`
- regression fixtures and integration tests in `tests/`

## Local setup

### Recommended toolchain

Install [mise](https://mise.jdx.dev) and run once per clone:

```bash
mise install
task setup    # installs pinned tools + lefthook git hooks
```

[`mise.toml`](mise.toml) pins Rust (stable), Python 3.11, [uv](https://docs.astral.sh/uv/), [lefthook](https://lefthook.dev/), and [go-task](https://taskfile.dev).

See [AGENTS.md](AGENTS.md) for the four-tier pre-flight model (Lefthook commit/push hooks → `task lint` / `task preflight` → `task ci`).

Install **`cargo-audit`** for the pre-push hook: `cargo install cargo-audit` (GitHub Actions pins 0.22.1).

### CI runner

All checks share one implementation: [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py). Task, Lefthook, GitHub Actions, and [`scripts/local_ci.sh`](scripts/local_ci.sh) call into it — no duplicated PowerShell/bash logic. GitHub Actions run automatically for pull requests and `v*` release tags; use local Task/Lefthook checks before pushing, or `workflow_dispatch` for an ad hoc remote run.

```bash
python3 scripts/ci/lme_ci.py ci
python3 scripts/ci/lme_ci.py rust-lint
```

### Rust

Install a stable Rust toolchain via [rustup](https://rustup.rs) (or `mise install`).

Useful commands:

```bash
cargo build --locked
cargo test --locked          # full suite; integration tests use [profile.test] opt-level 2
task test:fast               # unit tests only (~seconds after compile)
cargo check --workspace --all-targets --locked
cargo test --doc --locked
cargo fmt --check
cargo clippy --locked -- -D warnings
cargo doc --no-deps --locked
```

Or via Task:

```powershell
task lint        # fmt --check + clippy + Ruff (python/tests + examples)
task test:fast   # cargo test --lib only (quick unit tests)
task test        # full Rust test suite
task preflight   # pre-push hook: lint + check + cargo audit + repo-metadata dry-run
task audit       # cargo audit + pip-audit (GHA security audit mirror)
task rust        # full Rust slice (no Python)
task             # full core CI mirror
task ci:fast     # reuse python/.venv, skip isolated-wheel pytest
```

To run the same **core** checks as the pull-request/tag [GitHub Actions CI](.github/workflows/ci.yml) locally:

```bash
task ci
# or
python3 scripts/ci/lme_ci.py ci
```

```powershell
task ci
# or
python scripts/ci/lme_ci.py ci
```

Git hooks (parallel pre-commit + pre-push **preflight**) via Lefthook:

```powershell
task hooks:install
```

### Python bindings

If you are changing `python/` or verifying the Python package locally (requires [uv](https://docs.astral.sh/uv/) — pinned in [`mise.toml`](mise.toml)):

```bash
cd python
uv sync --extra dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

[`python/uv.lock`](python/uv.lock) pins CI and local dev dependencies from [`python/pyproject.toml`](python/pyproject.toml) (`[project.optional-dependencies] dev`). After changing those dependencies:

```bash
cd python
uv lock
```

Use `uv run --no-sync pytest tests/` after the explicit `uv sync` so only [`python/tests/`](python/tests/) runs against the extension Maturin just installed; `pytest` alone also collects optional demos under `python/examples/`.

[`task python`](Taskfile.yml) / [`scripts/ci/lme_ci.py python`](scripts/ci/lme_ci.py) assert the editable extension's version and environment path, then build, install, assert, and test the wheel in a separate locked environment. Pull-request/tag CI also exercises Python **3.10**, **3.12**, and **3.13** on Ubuntu (job `python-bindings-versions` in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). If you use **CPython 3.14** locally, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `maturin develop` (see [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)).

## Working on numerical changes

If you change fitting logic, optimizer behavior, variance calculations, or inference code:

- for **LMM θ-search / intercept-only hot paths**, read [OPTIMIZATION.md](OPTIMIZATION.md) and re-run the fair harness cases it lists
- add or update Rust tests in `tests/`
- prefer fixture-backed tests for parity-sensitive behavior; add identities to [`tests/test_statistical_identities.rs`](tests/test_statistical_identities.rs) when a property should always hold (e.g. OLS residual sum, `y = fitted + residual`, valid LRT probabilities)
- update `comparisons/COMPARISONS.md` when the reference output changes materially
- validate hard cases against R `lme4` where practical

Relevant files and directories:

- `src/` for the crate implementation
- `tests/data/` for fixture inputs
- `tests/generate_test_data.R` for R-backed fixture generation
- `comparisons/` for cross-language parity scripts (R, Python/statsmodels, Julia)

### Comparison script formatting (optional locally)

Cross-language R/Julia scripts under `comparisons/` (plus golden-parity generators in `tests/*.R`) use **format-only** checks — not full linters:

- **R:** [styler](https://cran.r-project.org/package=styler) via `scripts/ci/r_format.R`
- **Julia:** [JuliaFormatter](https://github.com/domluna/JuliaFormatter.jl) via `scripts/ci/julia_format.jl` (reads [`.JuliaFormatter.toml`](.JuliaFormatter.toml))

These are **optional** on commit when the runtime is missing (same tier as R/Julia benchmarks). When Rscript/styler or julia/JuliaFormatter are installed, Lefthook auto-formats staged `comparisons/**/*.R`, `tests/*.R`, and `comparisons/**/*.jl`.

```powershell
task lint:comparisons              # skip if R/Julia formatters missing
task lint:comparisons:required     # fail when tools/packages missing
python scripts/ci/lme_ci.py r-format-staged --fix
python scripts/ci/lme_ci.py julia-format-staged --fix
```

The tag/manual [benchmarks workflow](.github/workflows/benchmarks.yml) runs `comparison-format-check --required` before cross-language timing.

## Working on documentation

When documentation changes affect user-visible behavior, keep these files aligned:

- `README.md` for the repository landing page
- `GUIDE.md` for Rust usage
- `python/PYTHON_GUIDE.md` for Python usage
- `USABILITY.md` for workflow traffic lights and adoption posture
- `comparisons/COMPARISONS.md` for parity / golden evidence
- `BENCHMARKS.md` / `OPTIMIZATION.md` when fit timing or amortization guidance changes
- `CHANGELOG.md` for release-facing notes
- `REPO_COMPLETION_BY_AREA.md` when coverage claims change

Do not describe a feature as supported unless it is exposed by the public API and covered by tests or concrete examples.

## Other GitHub Actions workflows

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — CI on pull requests and `v*` tags, plus manual dispatch; ignored heavy production-load cases run only for tags/manual dispatch.
- [`.github/workflows/audit.yml`](.github/workflows/audit.yml) — a required release-CI gate running `cargo audit` on the root and `python/` Rust crates plus `pip-audit` on the [`python/uv.lock`](python/uv.lock) dev environment; manual dispatch is also available.
- [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml) — called by release CI to publish only after the full tag matrix succeeds; manual dispatch runs `cargo publish --dry-run --locked` only.
- [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml) — called by release CI to build and publish only after the full tag matrix succeeds; manual dispatch builds artifacts without publishing.

Pull requests automatically run CI. Ordinary non-PR branch pushes do not start workflows; use Lefthook and Task locally before pushing, and use manual dispatch when a remote check is useful before tagging.

### Manual dispatch (intentional remote runs)

In the GitHub UI: **Actions** → select a workflow → **Run workflow** → choose the branch (defaults to the default branch).

From the CLI (requires [`gh`](https://cli.github.com/) authenticated against this repository):

```powershell
# Core CI matrix (same jobs as a release tag run)
task gha:ci
# or: gh workflow run ci.yml --ref master

task gha:audit
task gha:benchmarks              # optional: WARMUPS=2 REPEATS=5 task gha:benchmarks
task gha:python-release          # build wheels only; no PyPI publish
task gha:crate-publish           # cargo publish --dry-run --locked
task gha:repo-metadata           # sync About box from Cargo.toml
task gha:fuzz                    # short libFuzzer smoke (manual-only workflow)
```

Pass a branch with `REF=my-branch task gha:ci`. Watch runs with `gh run list` or `gh run watch`.

## Repository metadata sync

The GitHub About box is synced from `Cargo.toml` using:

- `.github/workflows/repo-metadata.yml`
- `scripts/sync_github_repo_metadata.py`

If you change `package.description`, `homepage`, `keywords`, or `categories`, the metadata workflow will update the GitHub repository metadata on the next `v*` tag run or manual dispatch.

Preflight locally:

```powershell
task repo-metadata          # dry-run; verifies token if REPO_ADMIN_TOKEN is set
python scripts/sync_github_repo_metadata.py --dry-run
```

The workflow needs a valid **`REPO_ADMIN_TOKEN`** repository secret (fine-grained PAT with **Administration: Read and write** on this repo). If CI fails with `401 Bad credentials`, create a new PAT and update **Settings → Secrets and variables → Actions → REPO_ADMIN_TOKEN**.

## Pull request expectations

Keep changes focused. For non-trivial changes, include:

- the user-visible behavior change
- the tests you ran
- any compatibility or migration note if behavior changed

If a change is provisional or only partially mirrors R behavior, document that explicitly instead of implying full parity.
