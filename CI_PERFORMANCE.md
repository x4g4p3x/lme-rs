# CI performance

This document records hosted-CI timing evidence and the optimizations that are
safe to depend on. Times are GitHub Actions job timestamps, not local estimates.

## 2026-08-06 baseline and first optimization

The baseline was a successful pull-request run. Compilation and cache transfer,
not test execution, dominated its critical path. Windows spent 21m58s compiling
the Rust test suite, the first Python release build took 10m30s, and uploading
the raw Windows Cargo target cache took 13m58s. Most individual test binaries
then ran in seconds.

| Measurement | Wall time | Runner-minutes | Wall reduction | Runner reduction |
|---|---:|---:|---:|---:|
| [Baseline](https://github.com/x4g4p3x/lme-rs/actions/runs/30989006722) | 48.72 min | 129.22 min | - | - |
| [Optimized, cold cache](https://github.com/x4g4p3x/lme-rs/actions/runs/31095144153/attempts/1) | 17.70 min | 94.30 min | 63.7% | 27.0% |
| [Optimized, warm cache](https://github.com/x4g4p3x/lme-rs/actions/runs/31095144153/attempts/2) | 6.27 min | 27.70 min | 87.1% | 78.6% |

The warm workflow was 7.8 times faster while retaining the complete validation
matrix. The improvement came from Rust-aware caches for both Cargo workspaces,
CI-only non-incremental/no-debug profiles, controlled cache writers plus a
default-branch primer, independent Rust/Python OS jobs, one canonical full
Python flow with wheel-only compatibility jobs, and consolidated Ubuntu Rust
validation that reports every failing check together.

## Remaining bottlenecks after the first optimization

The warm run left two compilation bottlenecks:

- Windows `cargo test`: 5.43 minutes. The log contained 70 compiler/linker
  invocations, including one executable for each of 48 integration-test files
  and every example; the tests themselves took seconds.
- Ubuntu production-load validation: 5.60 minutes. Its cache restored the debug
  dependency set, so the release dependency graph rebuilt for 5m07s.

The follow-up optimization therefore uses a single integration-test harness on
Windows and macOS while executing the same test bodies, retains doctests, and
type-checks every example. The shared CI runner rejects a harness that omits or
stale-references any integration-test file. Ubuntu keeps the canonical per-file
`cargo test` layout. Release production-load artifacts use a distinct cache
populated by the default-branch primer so a debug-cache hit cannot mask a
release-cache miss.

When changing CI performance, record both cold and warm hosted runs here. Do not
trade away assertions, supported operating systems, Python versions, doctests,
or failure reporting for a lower timing number.
