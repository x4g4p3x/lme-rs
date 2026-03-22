# Releasing

This document describes the release flow for the Rust crate, the Python package, and the repository metadata that is synced from source control.

## Release scope

A release in this repository can touch three public surfaces:

- the Rust crate `lme-rs`
- the Python package `lme_python`
- the GitHub repository metadata derived from `Cargo.toml`

## Pre-release checks

Run these from the repository root unless noted otherwise.

### Rust validation

```bash
cargo build
cargo test
cargo test --doc
cargo fmt --check
cargo clippy -- -D warnings
cargo bench --no-run
```

`clippy` should be part of the release checklist. It is already enforced in CI, and releasing with known lint failures is unnecessary self-inflicted risk.

### Python validation

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

## Version bump checklist

Update all user-visible versioned surfaces together.

### Rust crate

- bump `version` in `Cargo.toml`
- run `cargo check` or `cargo build` to refresh `Cargo.lock` if needed

### Python package

- bump `version` in `python/Cargo.toml`
- confirm `python/pyproject.toml` still matches the intended package metadata

### Documentation

- update `CHANGELOG.md`
- update any README or guide snippets that mention a specific version
- review `comparisons/COMPARISONS.md` if release notes depend on changed outputs or behavior

## Benchmarks before release

If the release includes performance-sensitive changes, run the benchmark suite and record the result in the PR or release notes.

```bash
cargo bench
```

Current benchmark coverage is described in [BENCHMARKS.md](BENCHMARKS.md).

For cross-language timing runs, use:

```bash
python scripts/run_cross_language_benchmarks.py
```

## Git workflow

1. Ensure the worktree is clean enough to understand what is being released.
1. Commit the release changes.
1. Create an annotated tag:

```bash
git tag -a v0.1.3 -m "Release v0.1.3"
```

1. Push the branch and the tag:

```bash
git push origin master
git push origin v0.1.3
```

## GitHub Actions behavior

### CI

- `.github/workflows/ci.yml` runs on pushes to `master` and pull requests.
- This validates build, tests, formatting, and `clippy`.

### Python release workflow

- `.github/workflows/python-release.yml` builds wheels on pushes to `master` and on tags matching `v*`.
- The `publish` job only runs on tag pushes.
- On a tag push, the workflow publishes to PyPI and uploads artifacts to the GitHub Release.

### Repository metadata sync

- `.github/workflows/repo-metadata.yml` updates the GitHub About description, topics, and website from `Cargo.toml`.
- It runs when `Cargo.toml`, the metadata sync script, or the workflow file changes.

If a release changes the crate description, homepage, keywords, or categories, verify that the metadata sync workflow has completed successfully after the push.

### Benchmark workflow

- `.github/workflows/benchmarks.yml` runs Criterion benchmarks and cross-language timing runs.
- It uploads benchmark artifacts in CI and attaches them to GitHub Releases on tag pushes.

## Post-release verification

### Rust crate (post-release)

Verify the new crate version appears on crates.io and docs.rs.

Check:

- crate version is correct
- README rendering on crates.io is sane
- docs.rs builds and links resolve

### Python package (post-release)

Verify the PyPI release and GitHub Release artifacts.

Check:

- the expected wheel files exist
- the source distribution exists
- the published version matches `python/Cargo.toml`

### GitHub repository metadata

Verify the repository home page shows the expected:

- description
- topics
- website

## Common failure points

### Version drift

The most common release mistake in this repo is version drift between:

- `Cargo.toml`
- `python/Cargo.toml`
- `CHANGELOG.md`
- versioned examples in `README.md` or `GUIDE.md`

### Python build breakage

The wheel workflow depends on platform-specific build configuration. If the release tag is pushed without validating the Python package locally first, the release may fail after tagging.

### Metadata drift

If `Cargo.toml` changes but the metadata sync workflow fails, the GitHub About box can become stale even when the crate metadata is correct.

## Minimal release checklist

- update versions
- update `CHANGELOG.md`
- run build, tests, docs, fmt, and `clippy`
- run benchmarks if performance-sensitive code changed
- validate Python packaging if `python/` or release plumbing changed
- commit
- tag
- push branch and tag
- verify CI, PyPI, GitHub Release, docs.rs, and metadata sync
- verify benchmark artifacts if the release includes performance-sensitive changes
