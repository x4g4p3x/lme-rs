# Backward-compatible entry point. Prefer: task ci  or  python scripts/ci/lme_ci.py ci
$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'ci\lme_ci.py') ci @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
