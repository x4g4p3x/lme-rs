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
python scripts/ci/check_legal_compliance.py
```

`clippy` should be part of the release checklist. It is already enforced in CI, and releasing with known lint failures is unnecessary self-inflicted risk.

### Licensing and notices

Run `task legal` before tagging. Source releases include [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), fixture provenance, and the required GPL/LGPL/Intel license texts. The Python release workflow injects these files into every wheel's `*.dist-info/licenses/` directory.

The x86_64 wheel statically links Intel MKL and all wheels use `sprs-ldl` (LGPL-2.1). Keep [`RELINKING.md`](RELINKING.md) with any redistributed binary and review the target-specific dependency graph and third-party terms before publishing.

### Security-audit exception

`cargo audit` denies all warnings except `RUSTSEC-2024-0436`. That advisory is
informational: the unmaintained `paste` procedural macro is required by the
current `argmin` 0.11 release, and no patched `argmin` release exists. The
unused `statrs` feature path to `paste` is disabled. Remove the exception as
soon as `argmin` publishes a maintained replacement; do not add broader audit
exceptions to make a release pass.

### Python validation

```bash
cd python
uv sync --extra dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

## Version bump checklist

Update all user-visible versioned surfaces together.

### Where releases land (PyPI vs crates.io)

| Surface | Registry | After the `v*` tag CI succeeds | Maintainer action after tag |
|---------|----------|------------------|----------------------------|
| Rust crate `lme-rs` | [crates.io](https://crates.io/crates/lme-rs) | CI calls [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml), which publishes with the `CARGO_REGISTRY_TOKEN` repository secret | Verify the workflow and crates.io listing |
| Python `lme_python` | PyPI | CI calls [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml), which builds wheels and **publishes** | Wait for the workflow; no local publish step |
| API docs | [docs.rs](https://docs.rs/lme-rs) | Builds after the version appears on crates.io | Run `cargo publish` first |

Pushing a `v*` tag starts the full CI matrix. Only after every CI validation job succeeds does CI call the PyPI and crates.io workflows. Publication requires a valid `CARGO_REGISTRY_TOKEN` secret; docs.rs builds after the crate reaches crates.io.

### Rust crate

- bump `version` in `Cargo.toml`
- run `cargo check` or `cargo build` to refresh `Cargo.lock` if needed
- before tagging: run `cargo publish --dry-run --locked`; validated tag CI performs the actual publish with the repository secret

### Python package

- bump `version` in `python/Cargo.toml`
- confirm `python/pyproject.toml` still matches the intended package metadata
- no separate PyPI publish step — successful tag CI calls [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml)

### Documentation

- update `CHANGELOG.md`
- update any README or guide snippets that mention a specific version
- review `comparisons/COMPARISONS.md` if release notes depend on changed outputs or behavior

### Immediately after release

- bump both Cargo manifests to the next unique development version (for example,
  `0.2.1-dev.0` after releasing `0.2.0`)
- refresh the root, Python, and fuzz lockfiles
- do not leave `master` on a version already published to crates.io or PyPI; build
  and editable-install caches use the package version as part of artifact identity

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
git tag -a v0.2.0 -m "Release v0.2.0"
```

1. Push the branch and the tag:

```bash
git push origin master
git push origin v0.2.0
```

1. Wait for the full CI matrix; after it succeeds, verify the PyPI publish, Rust crate publish, benchmarks, and post-release checks below.

## GitHub Actions behavior

**No workflow runs automatically on ordinary branch pushes or pull requests.** Automatic runs are limited to **`v*` release tag pushes**. For pre-release validation on any branch, use **manual dispatch** (GitHub **Actions** tab → pick a workflow → **Run workflow**, or `gh workflow run` — see [CONTRIBUTING.md](CONTRIBUTING.md)).

### CI

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on `v*` tags and manual dispatch.
- It validates build, tests, formatting, `clippy`, security audits, legal/provenance policy, completion-score claims, production-load gates, Python versions, and docs across the release matrix.
- On a `v*` tag, and only after all validation jobs succeed, it calls both publishing workflows with publication enabled.

### Python release workflow

- [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml) is called by successful release-tag CI and can be manually dispatched.
- The `publish` job requires both CI's explicit `publish: true` input and a `v*` tag ref.
- After validated tag CI, the workflow **automatically publishes to PyPI** and uploads artifacts to the GitHub Release.
- Manual dispatch builds and uploads wheel artifacts only (no PyPI publish).

### Rust crate (crates.io)

- [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml) is called by successful release-tag CI and can be manually dispatched.
- **Validated tag CI:** `cargo publish --locked` using the `CARGO_REGISTRY_TOKEN` repository secret.
- **Manual dispatch:** `cargo publish --dry-run --locked` only (validates the package without publishing).
- It fails clearly if the token is missing during a validated tag publish; rotate the token and re-run the workflow rather than publishing from a different tree.

### Repository metadata sync

- [`.github/workflows/repo-metadata.yml`](.github/workflows/repo-metadata.yml) updates the GitHub About description, topics, and website from `Cargo.toml`.
- Runs on `v*` tags and manual dispatch (not on ordinary pushes).
- The workflow dry-runs the payload, verifies `REPO_ADMIN_TOKEN`, then PATCHes the repository.

Preflight locally: `task repo-metadata` (dry-run; add `REPO_ADMIN_TOKEN` to verify the PAT before push).

If CI fails with **401 Bad credentials**, rotate **Settings → Secrets and variables → Actions → `REPO_ADMIN_TOKEN`** (fine-grained PAT with **Administration: Read and write** on this repository).

If a release changes the crate description, homepage, keywords, or categories, verify that the metadata sync workflow has completed successfully after the push.

### Benchmark workflow

- [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) runs Criterion benchmarks and cross-language timing on `v*` tags and manual dispatch.
- Manual dispatch accepts `warmups` and `repeats` inputs.
- It uploads benchmark artifacts in CI and attaches them to GitHub Releases on tag pushes.

## Publishing the Rust crate to crates.io

Publishing is automated after successful `v*` tag CI by [`.github/workflows/crate-publish-dry-run.yml`](.github/workflows/crate-publish-dry-run.yml). The repository must have a valid `CARGO_REGISTRY_TOKEN` Actions secret. Before tagging, maintainers run `cargo publish --dry-run --locked` locally to validate the package; [docs.rs](https://docs.rs/lme-rs) builds after the workflow publishes the crate.

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

### Missing crates.io publish

If PyPI/GitHub Release show the new version but [crates.io/crates/lme-rs](https://crates.io/crates/lme-rs) does not, inspect the **Publish Rust crate** workflow. The usual cause is a missing or expired `CARGO_REGISTRY_TOKEN` repository secret; fix the secret and re-run the workflow from the tagged commit.

## Minimal release checklist

- update versions (`Cargo.toml`, `python/Cargo.toml`, `CHANGELOG.md`, version pins in docs)
- run build, tests, docs, fmt, and `clippy` (`task ci` or `python scripts/ci/lme_ci.py ci`)
- run `task preflight` before push (lint, `cargo check --all-targets`, `cargo audit`, metadata dry-run)
- run benchmarks if performance-sensitive code changed
- validate Python packaging if `python/` or release plumbing changed
- commit
- tag (`v*`)
- push branch and tag
- optionally run **`cargo publish --dry-run --locked`** before tagging; verify the [Publish Rust crate workflow](.github/workflows/crate-publish-dry-run.yml), PyPI / GitHub Release wheels, [crates.io](https://crates.io/crates/lme-rs), [docs.rs](https://docs.rs/lme-rs), and metadata sync
- verify benchmark artifacts if the release includes performance-sensitive changes
