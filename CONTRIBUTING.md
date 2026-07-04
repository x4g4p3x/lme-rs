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

All checks share one implementation: [`scripts/ci/lme_ci.py`](scripts/ci/lme_ci.py). Task, Lefthook, GitHub Actions, and [`scripts/local_ci.sh`](scripts/local_ci.sh) call into it — no duplicated PowerShell/bash logic. GitHub Actions run automatically only for `v*` release tags; use local Task/Lefthook checks for ordinary pushes and PR preparation, or `workflow_dispatch` for an ad hoc remote run.

```bash
python3 scripts/ci/lme_ci.py ci
python3 scripts/ci/lme_ci.py rust-lint
```

### Rust

Install a stable Rust toolchain via [rustup](https://rustup.rs) (or `mise install`).

Useful commands:

```bash
cargo build --locked
cargo test --locked
cargo check --workspace --all-targets --locked
cargo test --doc --locked
cargo fmt --check
cargo clippy --locked -- -D warnings
cargo doc --no-deps --locked
```

Or via Task:

```powershell
task lint        # fmt --check + clippy + Ruff (python/tests + examples)
task preflight   # pre-push hook: lint + check + cargo audit + repo-metadata dry-run
task audit       # cargo audit + pip-audit (GHA security audit mirror)
task rust        # full Rust slice (no Python)
task             # full core CI mirror
task ci:fast     # reuse python/.venv, skip wheel-reinstall pytest
```

To run the same **core** checks as the tag-triggered [GitHub Actions CI](.github/workflows/ci.yml) locally:

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

If you are changing `python/` or verifying the Python package locally:

```bash
cd python
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements-ci.txt
maturin develop --release
pytest tests/
```

[`python/requirements-ci.txt`](python/requirements-ci.txt) is a pinned tree for CI and local scripts (generated from [`python/pyproject.toml`](python/pyproject.toml) `[project.optional-dependencies] dev`). Regenerate it after changing those dependencies:

```bash
pip install pip-tools
cd python
pip-compile pyproject.toml --extra dev -o requirements-ci.txt --allow-unsafe --strip-extras
```

Use `pytest tests/` so only [`python/tests/`](python/tests/) runs (same as the tag-triggered [`.github/workflows/ci.yml`](.github/workflows/ci.yml)); `pytest` alone also collects optional demos under `python/examples/`.

The [`scripts/local_ci.sh`](scripts/local_ci.sh) / [`scripts/local_ci.ps1`](scripts/local_ci.ps1) helpers use **uv** to create `python/.venv` with Python **3.11** explicitly, then `pip install -r requirements-ci.txt`, `maturin develop`, and `pytest tests/`. The tag-triggered CI also runs that binding flow on **Ubuntu only** for Python **3.10**, **3.12**, and **3.13** (see job `python-bindings-versions` in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). If you use **CPython 3.14** in your own venv, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `maturin develop` (see [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)).

## Working on numerical changes

If you change fitting logic, optimizer behavior, variance calculations, or inference code:

- add or update Rust tests in `tests/`
- prefer fixture-backed tests for parity-sensitive behavior; add identities to [`tests/test_statistical_identities.rs`](tests/test_statistical_identities.rs) when a property should always hold (e.g. OLS residual sum, `y = fitted + residual`, valid LRT probabilities)
- update `comparisons/COMPARISONS.md` when the reference output changes materially
- validate hard cases against R `lme4` where practical

Relevant files and directories:

- `src/` for the crate implementation
- `tests/data/` for fixture inputs
- `tests/generate_test_data.R` for R-backed fixture generation
- `comparisons/` for cross-language parity scripts (R, Python/statsmodels, Julia)

## Working on documentation

When documentation changes affect user-visible behavior, keep these files aligned:

- `README.md` for the repository landing page
- `GUIDE.md` for Rust usage
- `python/PYTHON_GUIDE.md` for Python usage
- `CHANGELOG.md` for release-facing notes

Do not describe a feature as supported unless it is exposed by the public API and covered by tests or concrete examples.

## Other GitHub Actions workflows

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — release CI on `v*` tags and manual dispatch.
- [`.github/workflows/audit.yml`](.github/workflows/audit.yml) — `cargo audit` on the root and `python/` Rust crates, plus `pip-audit` against [`python/requirements-ci.txt`](python/requirements-ci.txt), on `v*` tags and manual dispatch.
- [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml) — `cargo publish --dry-run --locked` on `v*` tags and manual dispatch.
- [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml) — Python wheel builds and PyPI publish on `v*` tags; manual dispatch builds artifacts without publishing.

No GitHub Actions workflow runs automatically for ordinary branch pushes or pull requests. Use Lefthook and Task locally before pushing; use manual dispatch when a remote check is useful before tagging.

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
