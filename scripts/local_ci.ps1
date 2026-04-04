# Mirrors .github/workflows/ci.yml (Rust + Python checks).
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

Write-Host '==> cargo build --locked'
cargo build --verbose --locked

Write-Host '==> cargo test --locked'
cargo test --verbose --locked

Write-Host '==> python: maturin develop + pytest (from python/)'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location (Join-Path $root 'python')
# Maturin can pick up a repo-root or python/.venv with an unsupported CPython (e.g. 3.14).
$repoVenvBackup = $null
if (Test-Path (Join-Path $root '.venv')) {
    $repoVenvBackup = Join-Path $root '.venv_lme_rs_ci_backup_root'
    if (Test-Path $repoVenvBackup) { Remove-Item -Recurse -Force $repoVenvBackup }
    Rename-Item (Join-Path $root '.venv') $repoVenvBackup
}
$venvBackup = $null
if (Test-Path .venv) {
    $venvBackup = '.venv_lme_rs_ci_backup'
    if (Test-Path $venvBackup) { Remove-Item -Recurse -Force $venvBackup }
    Rename-Item .venv $venvBackup
}
try {
    $pyForVenv = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $pyForVenv = (& py -3.11 -c 'import sys; print(sys.executable)').Trim()
        } catch { }
    }
    if (-not $pyForVenv) {
        $pyForVenv = (Get-Command python -ErrorAction Stop).Source
    }
    & $pyForVenv -m venv .venv
    $vpy = Join-Path (Join-Path (Get-Location) '.venv') 'Scripts\python.exe'
    if (-not (Test-Path $vpy)) {
        $vpy = Join-Path (Join-Path (Get-Location) '.venv') 'bin\python'
    }
    $env:PYO3_PYTHON = (& $vpy -c 'import sys; print(sys.executable)').Trim()
    Remove-Item Env:\VIRTUAL_ENV -ErrorAction SilentlyContinue
    & $vpy -m pip install -q -U pip
    & $vpy -m pip install -q -r requirements-ci.txt
    & $vpy -m maturin develop --release
    if ($LASTEXITCODE -ne 0) { throw "maturin develop failed" }
    & $vpy -m pytest tests/ -v
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
    & $vpy -m maturin build --release -o dist
    if ($LASTEXITCODE -ne 0) { throw "maturin build failed" }
    $whl = Get-ChildItem -Path dist -Filter 'lme_python-*.whl' | Select-Object -First 1
    if (-not $whl) { throw "No wheel under python/dist" }
    & $vpy -m pip install -q --force-reinstall $whl.FullName
    & $vpy -m pytest tests/ -v
    if ($LASTEXITCODE -ne 0) { throw "pytest after wheel failed" }
} finally {
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
    if ($venvBackup -and (Test-Path $venvBackup)) {
        Rename-Item $venvBackup .venv
    }
    if ($repoVenvBackup -and (Test-Path $repoVenvBackup)) {
        $rv = Join-Path $root '.venv'
        if (Test-Path $rv) { Remove-Item -Recurse -Force $rv }
        Rename-Item $repoVenvBackup $rv
    }
}
Set-Location $root

Write-Host '==> cargo fmt --check'
cargo fmt --check

Write-Host '==> cargo clippy --locked'
cargo clippy --locked -- -D warnings

Write-Host '==> cargo check --workspace --all-targets --locked'
cargo check --workspace --all-targets --locked -v

Write-Host '==> cargo test --doc --locked'
cargo test --doc --locked --verbose

Write-Host '==> cargo doc --locked'
cargo doc --no-deps --verbose --locked

Write-Host 'local_ci.ps1: OK (matches GitHub CI core jobs; extra Python 3.10/3.12/3.13 matrix is CI-only)'
