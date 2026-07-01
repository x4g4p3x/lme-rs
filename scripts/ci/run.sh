#!/usr/bin/env bash
# Portable launcher for lme_ci.py (Lefthook + Git Bash on Windows).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Git Bash / lefthook often lack mise on PATH even after winget install.
if [ -n "${LOCALAPPDATA:-}" ] && [ -d "$LOCALAPPDATA/mise/shims" ]; then
  export PATH="$LOCALAPPDATA/mise/shims:$PATH"
fi
if [ -n "${USERPROFILE:-}" ] && [ -d "$USERPROFILE/AppData/Local/mise/shims" ]; then
  export PATH="$USERPROFILE/AppData/Local/mise/shims:$PATH"
fi
if [ -n "${HOME:-}" ] && [ -d "$HOME/.local/share/mise/shims" ]; then
  export PATH="$HOME/.local/share/mise/shims:$PATH"
fi

if command -v mise >/dev/null 2>&1; then
  exec mise exec -- python scripts/ci/lme_ci.py "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/ci/lme_ci.py "$@"
fi
if command -v python >/dev/null 2>&1; then
  exec python scripts/ci/lme_ci.py "$@"
fi

echo "error: install mise (mise install) or ensure python3 is on PATH" >&2
exit 1
