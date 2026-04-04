#!/usr/bin/env bash
# Mirrors .github/workflows/ci.yml (Ubuntu job: build, test, fmt, clippy, docs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> cargo build"
cargo build --verbose

echo "==> cargo test"
cargo test --verbose

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
  "$VPY" -m pip install -q maturin polars pytest
  "$VPY" -m maturin develop --release
  "$VPY" -m pytest tests/ -v
)

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo clippy"
cargo clippy -- -D warnings

echo "==> cargo doc"
cargo doc --no-deps --verbose

echo "local_ci.sh: OK (matches GitHub CI)"
