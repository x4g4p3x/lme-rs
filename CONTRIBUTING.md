# Contributing

## Scope

This repository contains:

- the Rust crate in the repository root
- the Python bindings in `python/`
- cross-language comparison examples in `examples/`
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
pytest
```

## Working on numerical changes

If you change fitting logic, optimizer behavior, variance calculations, or inference code:

- add or update Rust tests in `tests/`
- prefer fixture-backed tests for parity-sensitive behavior
- update `examples/COMPARISONS.md` when the reference output changes materially
- validate hard cases against R `lme4` where practical

Relevant files and directories:

- `src/` for the crate implementation
- `tests/data/` for fixture inputs
- `tests/generate_test_data.R` for R-backed fixture generation
- `examples/` for end-to-end language comparisons

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
