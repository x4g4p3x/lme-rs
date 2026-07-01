#!/usr/bin/env bash
# Backward-compatible entry point. Prefer: task ci  or  python3 scripts/ci/lme_ci.py ci
set -euo pipefail
exec python3 "$(dirname "$0")/ci/lme_ci.py" ci "$@"
