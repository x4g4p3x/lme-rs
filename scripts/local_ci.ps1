# Mirrors .github/workflows/ci.yml (Ubuntu job: build, test, fmt, clippy, docs).
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

Write-Host '==> cargo build'
cargo build --verbose

Write-Host '==> cargo test'
cargo test --verbose

Write-Host '==> cargo fmt --check'
cargo fmt --check

Write-Host '==> cargo clippy'
cargo clippy -- -D warnings

Write-Host '==> cargo doc'
cargo doc --no-deps --verbose

Write-Host 'local_ci.ps1: OK (matches GitHub CI)'
