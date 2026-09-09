# CI command reference

[Documentation](../../docs/README.md) · [Contributor setup](../../CONTRIBUTING.md) · [Required checks](../../AGENTS.md)

[lme_ci.py](lme_ci.py) is the shared Python 3.10+ runner for Task, Lefthook,
GitHub Actions, and the legacy wrappers. Run commands from the repository root.
Prefer Task for day-to-day work.

## Common commands

| Task command | Runner command | Coverage |
|:-------------|:---------------|:---------|
| `task lint` | `lint` | Rust fmt/Clippy and Python Ruff |
| `task test:fast` | `test-fast` | Rust unit tests |
| `task test` | `build-test` | Rust build and full test suite |
| `task preflight` | `preflight` | Lint, all-target check, Cargo audit, legal, metadata |
| `task docs:check` | `docs-check` | Local document paths, dashboard drift, Rust examples/doctests, API docs |
| `task completion:check` | `completion-check` | Manifest and published completion markers |
| `task consumer:smoke` | `consumer-smoke` | Rust sleepstudy and isolated Python wheel/example flow |
| `task python` | `python` | Editable extension and isolated-wheel tests |
| `task ci` | `ci` | Full local core validation |
| `task benchmarks:site` | `benchmark-site` | Regenerate checked-in dashboard data |
| `task explorations` | `explorations` | Parser, parameter-grid, and comparison probes |

For example:

```bash
python scripts/ci/lme_ci.py docs-check
python scripts/ci/lme_ci.py --help
```

Use `python3` when that is your platform's launcher.

## Validation boundaries

- Documentation validation checks local **paths**, not heading anchors or external URLs.
  It compiles Rust examples and runs doctests, but not all Markdown code fences.
- Consumer validation verifies an installed wheel's identity and runs portable
  workflows in isolated environments; it is stronger than importing from the checkout.
- `python --reuse-venv --skip-isolated-wheel` is a faster local development
  path that omits the wheel check.
- Metadata validation always dry-runs the payload. Token verification is skipped
  when `REPO_ADMIN_TOKEN` is absent.
- Hosted CI provides additional OS, interpreter, audit, and production-load coverage.
  See [Contributing](../../CONTRIBUTING.md#github-actions).

## Adding a check

Implement it in [lme_ci.py](lme_ci.py), expose a runner subcommand, then add a thin
alias in [Taskfile.yml](../../Taskfile.yml). Wire it into
[lefthook.yml](../../lefthook.yml) or Actions when required.
Keep platform handling in this shared implementation.

The legacy [shell](../local_ci.sh) and [PowerShell](../local_ci.ps1) wrappers
delegate to the full `ci` command; they are not Rust-only runners.
