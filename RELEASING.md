# Releasing

This document describes the release flow for the Rust crate, the Python package, and the repository metadata that is synced from source control.

## Release scope

A release in this repository can touch three public surfaces:

- the Rust crate `lme-rs`
- the Python package `lme_python`
- the GitHub repository metadata derived from `Cargo.toml`

## Pre-release checks

Run these from the repository root unless noted otherwise.

### Rust validation (same as GitHub CI)

To match [`.github/workflows/ci.yml`](.github/workflows/ci.yml) in one shot (build, test, `fmt --check`, `clippy -D warnings`, `doc`):

```bash
./scripts/local_ci.sh
```

```powershell
.\scripts\local_ci.ps1
```

### Rust validation (manual)

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

### Where releases land (PyPI vs crates.io)

| Surface | Registry | On `v*` tag push | Maintainer action after tag |
|---------|----------|------------------|----------------------------|
| Rust crate `lme-rs` | [crates.io](https://crates.io/crates/lme-rs) | [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml) runs `cargo publish --dry-run` only | **Manual** `cargo publish` (see below) |
| Python `lme_python` | PyPI | [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml) builds wheels and **publishes** | Wait for the workflow; no local publish step |
| API docs | [docs.rs](https://docs.rs/lme-rs) | Builds after the version appears on crates.io | Run `cargo publish` first |

Pushing a tag does **not** upload the Rust crate. If you only tag and push, PyPI will update but crates.io and docs.rs will not until someone runs `cargo publish`.

### Rust crate

- bump `version` in `Cargo.toml`
- run `cargo check` or `cargo build` to refresh `Cargo.lock` if needed
- after the tag is pushed: `cargo publish --dry-run --locked`, then `cargo publish` (requires [crates.io token](https://crates.io/settings/tokens) / `cargo login`)

### Python package

- bump `version` in `python/Cargo.toml`
- confirm `python/pyproject.toml` still matches the intended package metadata
- no separate PyPI publish step — the tag push triggers [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml)

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

1. **Publish the Rust crate to crates.io** (manual; not done by CI):

```bash
cargo publish --dry-run --locked
cargo publish
```

`cargo publish` requires a [crates.io API token](https://crates.io/settings/tokens) (`cargo login` once per machine). Tag pushes already run the same dry-run in [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml); the local step confirms your tree before upload.

1. Wait for GitHub Actions (PyPI publish, dry-run, benchmarks) and verify post-release checks below.

## GitHub Actions behavior

### CI

- `.github/workflows/ci.yml` runs on pushes to `master` and pull requests.
- This validates build, tests, formatting, and `clippy`.

### Python release workflow

- `.github/workflows/python-release.yml` builds wheels on pushes to `master` and on tags matching `v*`.
- The `publish` job only runs on tag pushes.
- On a tag push, the workflow **automatically publishes to PyPI** and uploads artifacts to the GitHub Release.

### Rust crate (crates.io)

- `.github/workflows/crate-publish-dry-run.yml` runs on tags matching `v*` (and on manual dispatch).
- It runs `cargo publish --dry-run --locked` to validate packaging (README, metadata, excluded files).
- It does **not** upload to crates.io — a maintainer must run `cargo publish` locally after the tag push (see [Publishing the Rust crate to crates.io](#publishing-the-rust-crate-to-cratesio)).

### Repository metadata sync

- `.github/workflows/repo-metadata.yml` updates the GitHub About description, topics, and website from `Cargo.toml`.
- It runs when `Cargo.toml`, the metadata sync script, or the workflow file changes.
- The workflow dry-runs the payload, verifies `REPO_ADMIN_TOKEN`, then PATCHes the repository.

Preflight locally: `task repo-metadata` (dry-run; add `REPO_ADMIN_TOKEN` to verify the PAT before push).

If CI fails with **401 Bad credentials**, rotate **Settings → Secrets and variables → Actions → `REPO_ADMIN_TOKEN`** (fine-grained PAT with **Administration: Read and write** on this repository).

If a release changes the crate description, homepage, keywords, or categories, verify that the metadata sync workflow has completed successfully after the push.

### Benchmark workflow

- `.github/workflows/benchmarks.yml` runs Criterion benchmarks and cross-language timing runs.
- It uploads benchmark artifacts in CI and attaches them to GitHub Releases on tag pushes.

## Publishing the Rust crate to crates.io

This step is **required** for every release and is **not** automated. Run from the repository root after pushing the release tag (and after CI / the dry-run workflow look green):

```bash
cargo publish --dry-run --locked
cargo publish
```

`cargo publish` requires a [crates.io API token](https://crates.io/settings/tokens) (`cargo login` once per machine). The `--dry-run` step validates the package without uploading; [docs.rs](https://docs.rs/lme-rs) will not build the new version until the crate is on crates.io.

## Post-release verification

### Rust crate (post-release)

Verify the new crate version appears on crates.io and docs.rs (docs.rs follows crates.io; if docs.rs is stale, confirm `cargo publish` was run).

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

The wheel workflow depends on platform-specific BLAS linking:

- **x86_64** (all OSes): static Intel MKL
- **macOS aarch64**: static OpenBLAS (`OPENBLAS_TARGET=ARMV8` in CI); Homebrew OpenBLAS breaks maturin wheel repair; MKL does not link on native Apple Silicon
- **Linux aarch64**: static OpenBLAS

macOS Apple Silicon is only fully validated on `macos-latest` in GHA. After changing `Cargo.toml` BLAS tables or wheel workflows, run `task ci` or confirm the macOS CI job passes.

If the release tag is pushed without validating the Python package locally first, the release may fail after tagging.

### Metadata drift

If `Cargo.toml` changes but the metadata sync workflow fails, the GitHub About box can become stale even when the crate metadata is correct. Common cause: expired **`REPO_ADMIN_TOKEN`** (401). Rotate the secret and re-run the workflow.

### Forgotten `cargo publish`

Tagging updates PyPI automatically but **not** crates.io. Symptoms: PyPI and GitHub Release show the new version, but [crates.io/crates/lme-rs](https://crates.io/crates/lme-rs) and docs.rs still list the previous version. Fix: run `cargo publish` from the tagged commit.

## Minimal release checklist

- update versions (`Cargo.toml`, `python/Cargo.toml`, `CHANGELOG.md`, version pins in docs)
- run build, tests, docs, fmt, and `clippy` (`task ci` or `python scripts/ci/lme_ci.py ci`)
- run `task preflight` before push (lint, `cargo check --all-targets`, `cargo audit`, metadata dry-run)
- run benchmarks if performance-sensitive code changed
- validate Python packaging if `python/` or release plumbing changed
- commit
- tag (`v*`)
- push branch and tag
- **`cargo publish --dry-run --locked` then `cargo publish`** (crates.io — manual; tag push alone is not enough)
- verify CI, [crate-publish dry-run](.github/workflows/crate-publish-dry-run.yml), PyPI / GitHub Release wheels, [crates.io](https://crates.io/crates/lme-rs), [docs.rs](https://docs.rs/lme-rs), and metadata sync
- verify benchmark artifacts if the release includes performance-sensitive changes
