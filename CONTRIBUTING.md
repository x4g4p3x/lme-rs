# Contributing

## Scope

This repository contains:

- the Rust crate in the repository root
- the Python bindings in `python/`
- cross-language comparison scripts in `comparisons/`
- regression fixtures and integration tests in `tests/`

## Local setup

### Rust

Install a stable Rust toolchain via [rustup](https://rustup.rs).

Useful commands:

```bash
cargo build
cargo test
cargo test --doc
cargo fmt --check
cargo clippy -- -D warnings
```

To run the same checks as [GitHub Actions CI](.github/workflows/ci.yml) locally (`build`, `test`, Python `maturin develop` + `pytest`, `fmt`, `clippy`, `doc`):

```bash
./scripts/local_ci.sh
```

```powershell
.\scripts\local_ci.ps1
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

pip install maturin polars pytest
maturin develop --release
pytest tests/
```

Use `pytest tests/` so only [`python/tests/`](python/tests/) runs (same as [`.github/workflows/ci.yml`](.github/workflows/ci.yml)); `pytest` alone also collects optional demos under `python/examples/`.

The same flow runs in CI on every push/PR: a fresh `python/.venv` is created with Python **3.11**, then `maturin develop` and `pytest tests/`. If you use **CPython 3.14** in your own venv, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `maturin develop` (see [python/PYTHON_GUIDE.md](python/PYTHON_GUIDE.md)). The [`scripts/local_ci.sh`](scripts/local_ci.sh) / [`scripts/local_ci.ps1`](scripts/local_ci.ps1) helpers temporarily back up existing `python/.venv` and repo-root `.venv`, run that flow, then restore them.

## Working on numerical changes

If you change fitting logic, optimizer behavior, variance calculations, or inference code:

- add or update Rust tests in `tests/`
- prefer fixture-backed tests for parity-sensitive behavior
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

## Repository metadata sync

The GitHub About box is synced from `Cargo.toml` using:

- `.github/workflows/repo-metadata.yml`
- `scripts/sync_github_repo_metadata.py`

If you change `package.description`, `homepage`, `keywords`, or `categories`, the metadata workflow will update the GitHub repository metadata on the next run.

## Pull request expectations

Keep changes focused. For non-trivial changes, include:

- the user-visible behavior change
- the tests you ran
- any compatibility or migration note if behavior changed

If a change is provisional or only partially mirrors R behavior, document that explicitly instead of implying full parity.
