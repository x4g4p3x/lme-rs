#!/usr/bin/env bash
# Mirrors .github/workflows/ci.yml (Rust + Python checks on Ubuntu-style hosts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> cargo build --locked"
cargo build --verbose --locked

echo "==> cargo test --locked"
cargo test --verbose --locked

echo "==> python: maturin develop + pytest (from python/)"
(
  set -e
  cd "$ROOT/python"
  # `maturin develop` requires a virtualenv. Back up any existing python/.venv and repo-root
  # .venv (they may point at CPython 3.14), create a fresh .venv with 3.11, then restore.
  REPO_VENV_BACKUP=""
  if [ -d "$ROOT/.venv" ]; then
    REPO_VENV_BACKUP="$ROOT/.venv_lme_rs_ci_backup_root_$$"
    mv "$ROOT/.venv" "$REPO_VENV_BACKUP"
  fi
  VENV_BACKUP=""
  if [ -d .venv ]; then
    VENV_BACKUP=".venv_lme_rs_ci_backup_$$"
    mv .venv "$VENV_BACKUP"
  fi
  restore_venvs() {
    rm -rf .venv 2>/dev/null || true
    if [ -n "$VENV_BACKUP" ] && [ -d "$VENV_BACKUP" ]; then
      mv "$VENV_BACKUP" .venv
    fi
    if [ -n "$REPO_VENV_BACKUP" ] && [ -d "$REPO_VENV_BACKUP" ]; then
      rm -rf "$ROOT/.venv" 2>/dev/null || true
      mv "$REPO_VENV_BACKUP" "$ROOT/.venv"
    fi
  }
  trap restore_venvs EXIT

  if command -v python3.11 >/dev/null 2>&1; then
    PY=python3.11
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
  "$PY" -m venv .venv
  if [ -f .venv/Scripts/python.exe ]; then
    VPY=.venv/Scripts/python.exe
  else
    VPY=.venv/bin/python
  fi
  export PYO3_PYTHON="$("$VPY" -c "import sys; print(sys.executable)")"
  unset VIRTUAL_ENV
  "$VPY" -m pip install -q -U pip
  "$VPY" -m pip install -q -r requirements-ci.txt
  "$VPY" -m maturin develop --release
  "$VPY" -m pytest tests/ -v
  "$VPY" -m maturin build --release -o dist
  shopt -s nullglob
  wheels=(dist/lme_python-*.whl)
  if [ ${#wheels[@]} -eq 0 ]; then
    echo "No wheel under python/dist"
    exit 1
  fi
  "$VPY" -m pip install -q --force-reinstall "${wheels[0]}"
  "$VPY" -m pytest tests/ -v
)

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo clippy --locked"
cargo clippy --locked -- -D warnings

echo "==> cargo check --workspace --all-targets --locked"
cargo check --workspace --all-targets --locked -v

echo "==> cargo test --doc --locked"
cargo test --doc --locked --verbose

echo "==> cargo doc --locked"
cargo doc --no-deps --verbose --locked

echo "local_ci.sh: OK (matches GitHub CI core jobs; extra Python 3.10/3.12/3.13 matrix is CI-only)"
