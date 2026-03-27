#!/usr/bin/env bash
# Mirrors .github/workflows/ci.yml (Ubuntu job: build, test, fmt, clippy, docs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> cargo build"
cargo build --verbose

echo "==> cargo test"
cargo test --verbose

echo "==> cargo fmt --check"
cargo fmt --check

echo "==> cargo clippy"
cargo clippy -- -D warnings

echo "==> cargo doc"
cargo doc --no-deps --verbose

echo "local_ci.sh: OK (matches GitHub CI)"
