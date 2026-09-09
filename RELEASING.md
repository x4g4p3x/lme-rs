# Release runbook

[Documentation](docs/README.md) · [Contributor setup](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

A release publishes the Rust crate and Python package from one immutable,
validated Git tag. The default branch is `master`.
This runbook documents release actions; ordinary documentation edits do not
require creating a release.

## Release sequence

1. Select a new, unpublished version and align the manifests, lockfiles, and notes.
2. Complete local checks and any required platform/performance validation.
3. Commit the release state and create an annotated `v<version>` tag.
4. Push the branch and that specific tag.
5. Wait for the full tag CI matrix and both publication workflows.
6. Verify the published artifacts, then advance `master` to a development version.

A pushed tag triggers publishing after successful validation. Do not reuse a
historical example tag, move a published tag, or publish a different checkout
to recover a failed upload.

## Align the release state

| File or artifact | Required state |
|:-----------------|:---------------|
| [Cargo.toml](Cargo.toml) | Root package version equals the release version |
| [python/Cargo.toml](python/Cargo.toml) | Binding version equals the root version |
| Root, Python, and fuzz Cargo lockfiles | Reflect the selected package/dependency graph |
| [CHANGELOG.md](CHANGELOG.md) | Versioned section with date and user-facing changes |
| README and guide dependency examples | Match the intended published installation |
| [python/pyproject.toml](python/pyproject.toml) | Intended package metadata; version derives from Cargo |
| [Completion manifest](completion_manifest.json) and report | Current evidence and generated claims remain consistent |

After version/dependency updates, refresh the relevant lockfiles and inspect
their diffs. Include Python environment lock changes only when its dependency
inputs require them.

## Pre-release checks

Run from the repository root:

```bash
task ci
task preflight
cargo publish --dry-run --locked
```

`task ci` includes Rust tests, bindings and consumer examples, docs, legal checks,
and the completion check. `task preflight` also checks Cargo advisories.
Hosted CI supplies the OS/Python matrix, production-load cases, and
`pip-audit`; the local flow does not reproduce every hosted job.

For BLAS target-table or wheel-workflow changes, complete `task ci` locally or
dispatch hosted CI before tagging, and inspect the macOS Apple Silicon result.
Use `task gha:ci` for a deliberate remote validation run.

### Performance-sensitive changes

Read [OPTIMIZATION.md](OPTIMIZATION.md) for affected fitting paths.
Run applicable fair-harness cases and retain dated outputs. Rust-only
microbenchmarks do not establish external speed parity.
[Benchmark coverage](BENCHMARK_COVERAGE.md) maps the available evidence.

### Licensing and notices

Run `task legal`. Preserve [third-party notices](THIRD_PARTY_NOTICES.md),
fixture provenance, [license texts](LICENSES/), and
[relinking instructions](RELINKING.md) in source and binary distributions.
The wheel workflow injects notices into `*.dist-info/licenses/`.

### Audit policy

The shared runner denies Cargo audit warnings except its narrowly configured
`RUSTSEC-2024-0436` exception for the transitive `paste` macro.
Check [lme_ci.py](scripts/ci/lme_ci.py) and the current dependency graph before
changing that exception. Do not broaden exclusions to make a release pass.

## Tag and push

Review `git status` and the exact release commit first. Create an annotated tag
whose name is `v` followed by the selected manifest version. The following is
a **template**; replace `<version>` before running it:

```text
git tag -a v<version> -m "Release v<version>"
git push origin master
git push origin v<version>
```

Push only the intended tag. Wait for validation and publication before calling
the release complete.

## What GitHub Actions does

| Surface | Trigger and publication behavior |
|:--------|:---------------------------------|
| [Core CI](.github/workflows/ci.yml) | Pull requests, `v*` tags, manual runs; publishing is activated only for successful tag CI |
| [Rust crate](.github/workflows/crate-publish-dry-run.yml) | Called after tag validation; publishes with `CARGO_REGISTRY_TOKEN`. Manual dispatch performs a dry run |
| [Python package](.github/workflows/python-release.yml) | Top-level dispatch from successful tag CI; verifies tag, SHA, and package versions before building/publishing |
| [API docs](https://docs.rs/lme-rs) | docs.rs builds after the crate is published |
| [Benchmarks](.github/workflows/benchmarks.yml) | Tag/manual runs produce artifacts; successful publication can update the dashboard overlay |
| [Repository metadata](.github/workflows/repo-metadata.yml) | Tag/manual sync of About fields from Cargo metadata |
| [Pages](.github/workflows/pages.yml) | Relevant `master` changes/manual runs deploy the dashboard |

Ordinary non-PR branch pushes do not run the full validation matrix.
They can still trigger Pages or cache preparation when their path filters match.

### Python publishing boundary

The publishing workflow remains top-level for Trusted Publishing attestations.
The configured publisher names `python-release.yml` and the `pypi` environment.

Normal manual dispatch builds artifacts without publishing. An explicit
`publish=true` recovery requires a semantic release tag, its exact 40-character
validated commit SHA, and matching versions in both Cargo manifests.
The workflow rejects mismatches.

## Verify the release

- Confirm the requested version exists on [crates.io](https://crates.io/crates/lme-rs).
- Confirm [docs.rs](https://docs.rs/lme-rs) built that version; inspect its rendering.
- Confirm [PyPI](https://pypi.org/project/lme-python/#files) contains the expected
  wheel files and source distribution.
- Confirm [GitHub Releases](https://github.com/x4g4p3x/lme-rs/releases) contains
  the intended notes and artifacts.
- Verify clean installation and import for the targeted consumer environment.
- Check About fields and any promised benchmark artifacts.

Wheel files, interpreter metadata, and the CI test matrix are separate checks.
Do not infer a published wheel from a successful source-build job.

## Recover a failure

| Symptom | Action |
|:--------|:-------|
| CI fails | Read the exact job/log; resolve the failure before publication |
| Python tag/SHA/version check fails | Correct the dispatch inputs or prepare a new release; do not move the old tag |
| Validated Python build fails only during upload | Fix the publication configuration, then dispatch from `master` with the original validated tag/SHA and explicit publishing enabled |
| Rust publish lacks credentials | Restore `CARGO_REGISTRY_TOKEN` and rerun the failed publishing job from the release run |
| Metadata returns `401` | Rotate `REPO_ADMIN_TOKEN`, then rerun metadata sync |
| docs.rs has not built | First verify crates.io publication, then inspect the docs.rs build |
| Apple Silicon wheel linking fails | Inspect the target-specific static OpenBLAS configuration and the macOS job |

The ordinary manual Rust publishing workflow is **dry-run only**; rerunning a
failed release job preserves the original publishing context.
Keep recovery tied to the version and commit that passed validation.

## After release

Advance both Cargo manifests to the next unique development version, refresh
the root/Python/fuzz lockfiles, and restore an Unreleased changelog section.
For example, after a `0.2.3` release, development may use `0.2.4-dev.0`.
The actual next version is a maintainer decision.

Keep `master` distinct from already published artifact versions so editable
and wheel caches do not confuse release and development builds.
