# Toolchain follow-up, 10 October 2026

Baseline: `0557890c72dc7f3718465ccbc51acc66e501ea26`. This report concerns
packaging, linking, validation, and environment maintenance. Model algorithms
and inference tolerances are unchanged.

## Adopted maintenance changes

- Dependabot now covers both Cargo manifests and both uv projects. Minor and
  patch updates are grouped; major updates remain separate proposals.
  GitHub documents these ecosystems and multi-directory updates in its
  [options reference](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
- Ruff checks the repository tooling as well as binding tests/examples.
  actionlint 1.7.12 checks workflows in hooks, preflight, and CI. Optional
  ShellCheck/Pyflakes are disabled consistently across platforms.
- A dedicated locked Python comparison environment supplies all dependencies
  for the statistical harness tests. Missing imports fail before pytest can
  skip a module. Full CI executes these tests and the tooling unit tests.
- R 4.6.1 uses a generated renv 1.3.1 lock, explicit package declarations,
  and package caching. Fresh restore checks include imports and synchronization
  with the lock. Caller environment overrides remain available.
- Both Rust manifests declare a minimum compiler of 1.88; Python metadata
  declares 3.10, matching the existing interpreter matrix. The minimum Rust
  job checks the library, optional features, examples, and bindings. Both
  minimum and latest compiler jobs gate releases.

Enabling comparison regressions exposed statsmodels 0.15's move from
`data.design_info` to `data.model_spec`. The harness now accepts both fitted
Patsy layouts and rejects missing/unknown specifications. Independent tests
preserve training categories and column order on a reduced prediction grid.
Existing numerical tolerances and hypothesis families are unchanged.

## Measurement rules

Windows x64, 12 logical processors, Rust 1.99.0. Every timing comparison has
two warmups and ten measured samples per variant, alternating pair order.
BLAS/OpenMP, Rayon, and Polars thread limits are one; test-runner parallelism
is two for both runners. Preparation/imports are outside Python call timings.
Test timings include the Cargo/Nextest command overhead with compiled artifacts
already available. Link timings replay identical object/archive inputs and
library search paths, excluding compilation and model execution.

The [raw evidence](toolchain-followup-2026-10-10.json) retains every warmup and
measurement, command, native artifact/input hash, and the candidate source hashes.
The driver gained SDK-path and consolidated-suite options during the investigation;
each report retains the driver hash used for its own measurements.

The [experiment driver](../scripts/run_toolchain_experiments.py) records raw
samples, commands, compiler inputs or native-module hashes, and environment
metadata. Run its `--help` for the three probe modes. Build artifacts before
timing; keep probes sequential and free of competing builds.

## Shared Python wheels

The candidate `cp310-abi3-win_amd64` wheel passed all 88 binding tests on each
of CPython 3.10, 3.11, 3.12, 3.13, and 3.14. The optional workflow module was
skipped in these standard development environments and is covered separately
by the dedicated comparison check.

Fresh-process Python 3.11 comparisons used matching locked dependencies and
checked each fit against independently computed OLS coefficients and predictions.

| Complete measured batch | Native median | abi3 median |
|---|---:|---:|
| 200 ordinary least-squares fits | 39.005 ms | 38.710 ms |
| 50,000 coefficient getters | 3.389 ms | 3.429 ms |
| 1,000 prediction calls | 180.550 ms | 179.752 ms |

The exploratory paired-bootstrap 95% candidate/baseline ratio intervals are
0.975–1.007 for fits, 0.862–1.034 for getters, and 0.991–1.007 for predictions
(2,000 resamples, fixed seed). These small differences overlap observed
variability and show no material slowdown in these measured Windows workloads.
They do not establish general
cross-platform performance equivalence or cover every binding operation.

Decision: use the optional `abi3` feature for release wheels. The build matrix
shrinks from 25 interpreter/platform combinations to five platform/architecture
builds. The 15 native installation jobs still cover Python 3.10 through 3.14
on Linux x86_64, Windows x64, and macOS Apple Silicon. Linux aarch64 and macOS
x86_64 remain build-only targets. Source builds retain native interpreter APIs.
This follows [Maturin's stable-ABI packaging guidance](https://www.maturin.rs/tutorial).
The tested interpreters are standard GIL-enabled CPython; this does not establish
free-threaded interpreter compatibility.

## Windows linking

The release sleepstudy example was compiled once with saved temporary objects.
The probe replayed the captured linker arguments with Microsoft's linker and
the pinned compiler's `rust-lld -flavor link`. Both received the same explicit
MSVC 14.50.35717 and Windows SDK 10.0.26100.0 library paths.

| Linker | Median | Range |
|---|---:|---:|
| Microsoft link.exe | 1,927.950 ms | 1,873.726–1,970.585 ms |
| Rust LLD | 712.361 ms | 693.682–758.375 ms |

LLD reduced this linking phase by 63.1%. Both linked executables ran successfully
and produced identical sleepstudy model summaries and predictions. This is
neither a complete-build speed claim nor evidence about debugger usability.
Its exploratory 95% median-time ratio interval is 0.362–0.388 versus Microsoft.

Decision: expose `task ci:windows:lld` as an explicit opt-in, preserving the
normal platform linker. Full validation results are recorded below.

## Test runner

Nextest 0.9.148 was downloaded from the official release and checked against
its SHA-256 manifest. The first comparison selected the same 132 unit tests.

| Runner, unit suite | Median command time | Range |
|---|---:|---:|
| Cargo | 542.175 ms | 508.139–576.530 ms |
| Nextest | 2,169.724 ms | 2,109.242–2,327.932 ms |

The consolidated comparison selects the same two binaries: 132 unit tests and
380 integration tests, with the same four ignored heavy cases. Doctests remain
the separate unchanged Cargo check.

| Runner, consolidated suite | Median command time | Range |
|---|---:|---:|
| Cargo | 10,634.129 ms | 10,128.746–11,794.511 ms |
| Nextest | 15,696.158 ms | 15,054.590–16,092.704 ms |

Every invocation ran the same 512 tests successfully and retained the four
heavy ignored cases. Nextest was 47.6% slower in this measured Windows
consolidated workload, with an exploratory 95% median-time ratio interval of
1.413–1.533 versus Cargo. Its process isolation and reporting remain useful
features, but this evidence does not justify a migration here. Keep Cargo
and the existing consolidated harness; no Nextest dependency is added to
normal development or CI. Other operating systems were not timed.

## Setup failures retained

- An initial ABI-wheel test invocation used the repository root rather than
  the binding tests' required `python/` working directory. Missing fixture paths
  caused 49 failures; corrected runs passed on all five interpreters.
- A source-only R package lacked Windows build tools. Lock generation used
  available CRAN Windows binaries, then verified a fresh independent restore.
- The first standalone Microsoft relink lacked Cargo's implicit SDK search
  environment. The fair comparison supplied identical explicit library paths
  to both linkers. The initial failure is retained in the raw evidence.
- The first LLD validation attempt was stopped to serialize Python environment
  changes with normal CI. A later attempt caught an outdated compatibility-test
  fixture after the shared-ABI guard was strengthened; the fixture was corrected
  and the complete check restarted. Neither was a linker correctness failure.

## Validation

- Full `task ci` passed: normal and Basin Rust suites, doctests, native editable
  bindings, isolated native/shared wheels, consumer examples, 16 comparison
  regressions, tooling tests, lint, legal notices, documentation, and completion.
- `task preflight` passed, including both Cargo security audits and repository
  metadata validation. The extended `task audit` passed both Cargo graphs and
  both locked Python dependency environments. Earlier audits with the local
  extension installed explicitly skipped it because the unpublished version
  cannot be resolved on PyPI. Development bindings were restored after auditing.
- Rust 1.88.0 passed all-target compilation with Basin/Polars lazy features and
  both native/shared-ABI binding compilation. The same shared Windows wheel
  passed 88 tests on each of the five supported Python versions.
- R 4.6.1 restored all 86 locked packages into an independent project, verified
  synchronization/imports, and ran sleepstudy, cbpp GLMM, and categorical ANOVA
  reference examples. All four R golden-reference cases passed without skips.
  Required R/Julia formatting checks passed with both runtimes installed.
- Full `task ci:windows:lld` passed the same Rust/Basin, binding, native/shared
  wheel, consumer, comparison, tooling, documentation, and completion checks
  using the pinned compiler's LLD in an isolated target directory. This check
  retained the normal build profiles and numerical tolerances. Standard CI
  and release jobs continue to use the platform linker.

Hosted Linux/macOS runtime and packaging checks require a future workflow run;
local Windows validation does not establish their results.
