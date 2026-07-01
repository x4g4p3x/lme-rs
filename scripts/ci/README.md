# CI scripts

All local and GitHub Actions checks flow through **`lme_ci.py`** — a single stdlib Python 3.10+ runner. There are no paired `.ps1` / `.sh` implementations to keep in sync.

## Usage

```bash
python3 scripts/ci/lme_ci.py ci
python3 scripts/ci/lme_ci.py rust-lint
python3 scripts/ci/lme_ci.py python --reuse-venv --skip-wheel-reinstall
```

On Windows, use `python` instead of `python3`.

Prefer [`Taskfile.yml`](../../Taskfile.yml) (`task ci`, `task lint`, …) or [`lefthook.yml`](../../lefthook.yml) for day-to-day work.

## Design

- **Cross-platform** — one code path for Windows, macOS, and Linux.
- **uv** — creates `python/.venv` with Python 3.11 explicitly (avoids maturin picking an unsupported system Python).
- **Ruff** — `uv tool run ruff` for staged Python files; config in [`python/pyproject.toml`](../python/pyproject.toml).
- **GitHub Actions** — calls the same `lme_ci.py` subcommands as local Task/Lefthook.

Legacy wrappers [`scripts/local_ci.sh`](../local_ci.sh) and [`scripts/local_ci.ps1`](../local_ci.ps1) delegate to `lme_ci.py ci`.
