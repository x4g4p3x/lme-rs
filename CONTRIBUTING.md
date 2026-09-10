# Contributing

[Documentation](docs/README.md) · [Required checks](AGENTS.md) · [Release guide](RELEASING.md)

The repository contains the Rust crate, Python bindings, numerical fixtures,
cross-language comparisons, and documentation. Start with the setup below,
then run the checks appropriate to your change.

## Local setup

Install [mise](https://mise.jdx.dev), then run from the repository root:

```bash
mise install
task setup
```

[mise.toml](mise.toml) configures Rust stable, Python 3.11, uv, Task, and Lefthook.
If tools are installed already, `task hooks:install` installs the Git hooks.
The pre-push audit also requires `cargo-audit`; install it with
`cargo install cargo-audit`.

All commands below assume the repository root unless a section explicitly
changes directory. Use `mise exec -- task <name>` when the tools are installed
but their shims are not active in your shell.

## Choose the required checks

[AGENTS.md](AGENTS.md) is the authoritative validation policy.

| Change | Minimum checks and escalation |
|:-------|:------------------------------|
| Rust code | `task lint`, `task test:fast`; `task rust` for cross-module or public-API changes |
| Python bindings | `task lint:python`, `task python` |
| CI, manifests, release tooling | `task preflight`; `task ci` for releases or broad refactors |
| R/Julia comparisons | `task lint:comparisons`; use the required variant when formatters are installed |
| Documentation | `task docs:check`; `task consumer:smoke` when install/example behavior changes |
| Completion-related files | `task completion:check` |
| LMM throughput paths | Read [OPTIMIZATION.md](OPTIMIZATION.md) and run applicable fair-harness cases |

For changes spanning rows, run all applicable checks. Report checks that failed,
were skipped, or require a hosted platform.

### What each layer covers

- **Commit hook:** checks matching staged files. It does not run the full test suite.
- **Push hook:** runs `task preflight`: lint, all-target compilation, Cargo audits,
  legal/provenance checks, and metadata validation.
- **`task ci`:** the local core CI flow, including Rust tests, bindings, portable
  consumer examples, documentation, and completion checks.
- **Hosted CI:** adds the OS/Python matrix, production-load gates, and
  `pip-audit`. Local success does not establish macOS Apple Silicon behavior.

Use `--no-verify` only when explicitly necessary and report the bypass.
`task ci:fast` reuses the editable Python environment and skips the isolated
wheel pass; it is not equivalent to full `task ci`.

## Rust development

```bash
task lint
task test:fast
cargo run --release --locked --example sleepstudy
```

`task test` runs the full Rust suite. `task rust` runs the full Rust validation
slice without Python. Use release mode for numerical examples; the first build
can take longer because of native numerical dependencies.

### Numerical changes

Add tests that exercise the changed behavior. Prefer fixture-backed parity
checks and statistical identities over tests that repeat implementation details.

| Area | Where to work |
|:-----|:--------------|
| Public fitting API | [src/lib.rs](src/lib.rs) |
| Regression and identity tests | [tests](tests/) · [statistical identities](tests/test_statistical_identities.rs) |
| Reference data | [tests/data](tests/data/) · [golden manifest](tests/data/golden_parity_manifest.json) |
| R fixture generation | [tests/generate_test_data.R](tests/generate_test_data.R) |
| Independent comparisons | [comparisons](comparisons/) |
| Performance methodology | [BENCHMARKS.md](BENCHMARKS.md) · [OPTIMIZATION.md](OPTIMIZATION.md) |

For reference comparisons, match rows, formula, family, link, weights, and
REML/ML mode. Record tolerances and their reason. Refresh documented evidence
when numerical output changes materially.

## Python bindings

The simplest complete validation command is `task python` from the root.
For an interactive development environment:

```bash
cd python
uv sync --extra dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
uv run --no-sync python examples/lmer_sleepstudy.py
```

`uv sync` can uninstall the editable extension. After synchronizing or an
interrupted build, rerun Maturin. Keep `--no-sync` on subsequent tests and
examples so the extension you just built remains installed.

[python/uv.lock](python/uv.lock) locks the development dependencies.
After changing [python/pyproject.toml](python/pyproject.toml), run `uv lock`
from `python/` and validate the package.

The complete bindings flow checks the extension's version and import path,
runs the editable package tests, then builds and tests an isolated wheel.
`task consumer:smoke` additionally installs the wheel in a dependency-only
environment and runs the portable examples. CI tests source builds on Python
3.10–3.13; the full identity/consumer flow is centered on 3.11.

## Comparison scripts

R and Julia use formatter checks rather than broad lint suites:

- R: `styler`, through [r_format.R](scripts/ci/r_format.R).
- Julia: `JuliaFormatter`, through [julia_format.jl](scripts/ci/julia_format.jl)
  and [.JuliaFormatter.toml](.JuliaFormatter.toml).

```bash
task lint:comparisons
task lint:comparisons:required
```

The optional command skips unavailable runtimes/packages; the required command
fails when they are missing. Installed commit formatters can modify and restage
matching scripts. The tag/manual benchmark workflow requires the formatters.

## Working on documentation

Use [the documentation index](docs/README.md) as the navigation map.

The Rust API landing page comes from [docs/rustdoc.md](docs/rustdoc.md), included
by [src/lib.rs](src/lib.rs). Its theme extension is
[docs/rustdoc.css](docs/rustdoc.css). Run `task doc` to build the styled preview
at `target/doc/lme_rs/index.html`; `task docs:check` also tests its Rust examples
and intra-doc links. Keep the stylesheet arguments in `Cargo.toml` and the
shared CI runner aligned. Check light, dark, and Ayu themes and a narrow viewport
after styling changes. docs.rs builds the published crate, so repository edits
reach that site with a new release, following [RELEASING.md](RELEASING.md).

| Content | Canonical location |
|:--------|:-------------------|
| Introduction and first successful fit | [README.md](README.md) |
| Rust recipes and semantics | [GUIDE.md](GUIDE.md) |
| Python installation and recipes | [python/README.md](python/README.md), [Python guide](python/PYTHON_GUIDE.md) |
| Runnable commands and dependencies | [Example catalog](docs/EXAMPLES.md) |
| Shared error diagnosis | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Workflow scope | [USABILITY.md](USABILITY.md) |
| Numerical and timing evidence | [Comparisons](comparisons/COMPARISONS.md), [benchmarks](BENCHMARKS.md) |
| Release history | [CHANGELOG.md](CHANGELOG.md) |

For every new recipe, state whether it is standalone or a fragment, its working
directory, required packages, and where data comes from. Explain result
semantics before listing options. Link to canonical details instead of copying
long option lists across documents.

`task docs:check` checks local path targets, dashboard JSON drift, Rust
examples/doctests, and generated API docs. It does **not** check heading anchors,
external URLs, or execute all Markdown snippets. Inspect navigation and execute
changed copyable examples separately; use `task consumer:smoke` for installation
and example changes.

Preserve dated measurements and release history as dated evidence.
Completion percentages are generated from locked scopes in
[completion_manifest.json](completion_manifest.json); never edit percentages
or narrow scopes to make a documentation refresh appear more complete.

## CI runner

[scripts/ci/lme_ci.py](scripts/ci/lme_ci.py) is the shared implementation used
by Task, Lefthook, Actions, and the legacy local-CI wrappers. Add new checks
there rather than duplicating shell logic.

```bash
python scripts/ci/lme_ci.py --help
```

Use `python3` if that is your Python launcher on macOS/Linux.
[The CI runner reference](scripts/ci/README.md) maps common commands.

## GitHub Actions

| Workflow | Trigger and purpose |
|:---------|:--------------------|
| [CI](.github/workflows/ci.yml) | Pull requests, `v*` tags, manual dispatch; PRs skip four ignored heavy production-load cases |
| [Cache prime](.github/workflows/cache-prime.yml) | Relevant dependency changes on `master` and weekly trusted cache preparation |
| [Benchmarks](.github/workflows/benchmarks.yml) | Tags/manual runs; timing artifacts and dashboard overlay |
| [Pages](.github/workflows/pages.yml) | Dashboard input changes on `master` or manual dispatch |
| [Audit](.github/workflows/audit.yml) | Release validation, weekly, manual; Cargo and Python dependency audits |
| [Fuzz smoke](.github/workflows/fuzz-smoke.yml) | Weekly/manual formula fuzzing |
| [Rust publishing](.github/workflows/crate-publish-dry-run.yml) | Called after successful tag CI; ordinary manual runs are dry runs |
| [Python publishing](.github/workflows/python-release.yml) | Top-level dispatch after successful tag CI; ordinary manual runs build only |
| [Metadata](.github/workflows/repo-metadata.yml) | Tags/manual synchronization of the GitHub About box |

Ordinary branch pushes do not start the full validation matrix.
[CI_PERFORMANCE.md](CI_PERFORMANCE.md) records historical hosted timings.

### Manual dispatch

Use **Actions → workflow → Run workflow**, or these Task aliases:

```bash
task gha:ci
task gha:audit
task gha:benchmarks
task gha:python-release
task gha:crate-publish
task gha:repo-metadata
task gha:fuzz
task gha:pages
```

The ordinary Python and crate-publish aliases do not publish packages.
Choose a branch with `task gha:ci REF=your-branch`.
Inspect runs with `gh run list` or `gh run watch`.

### Repository metadata sync

The About description, topics, and homepage come from [Cargo.toml](Cargo.toml).
Run `task repo-metadata` to dry-run the payload. If `REPO_ADMIN_TOKEN` is set,
the runner also verifies it.

The hosted workflow needs a fine-grained token with repository administration
write access. A `401 Bad credentials` requires rotating the Actions secret
and rerunning the failed workflow.

## Pull requests

Describe the concrete problem, resulting behavior, validation, and any
compatibility limits. For provisional numerical behavior, state the scope and
evidence. Keep unrelated local changes out of the commit.
