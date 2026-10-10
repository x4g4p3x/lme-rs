# Toolchain changes, 10 October 2026

Baseline: `5f5765b366aa18c6f36eaea878561fe2fc17b745`.
The [raw evidence](toolchain-2026-10-10.json) records dependency sets, compiler
identity, source checksum, commands, individual samples, and cache statistics.

## Dependency footprint

`cargo tree --locked --edges normal,build --prefix none --format '{p}'` was run
against both revisions' root and Python manifests/lockfiles. ANSI escapes and
duplicate package markers were removed before counting unique packages.

| Normal/build dependencies | Before | After |
| --- | ---: | ---: |
| Rust library | 246 | 239 |
| Python extension | 260 | 253 |

The removed packages are `polars-lazy`, `polars-plan`, `polars-expr`,
`polars-mem-engine`, `recursive`, `recursive-proc-macro-impl`, and `rand_distr 0.4.3`.
Lazy expressions remain in development dependencies and the opt-in `polars-lazy`
feature. CSV input, IPC bindings, statistical APIs, and numerical algorithms
are unchanged. Removing the unused Polars object feature also removes an internal
optional dependency edge; neither Cargo lockfile changes package versions.

This is a dependency-graph reduction, not a measured whole-build speedup.
Development tests/examples still compile the lazy engine. The initial check
timings used different warm build caches and are not comparable evidence.

## Compiler-cache mechanism measurement

Windows x86_64, AMD64 Family 25 Model 117 Stepping 2, 12 logical processors,
Rust 1.99.0, sccache 0.16.0; the actual `smallvec 1.15.1` dependency source
was compiled as an optimized Rust library in a dedicated output directory. Both
variants used the same compiler, input, options, and output path. The cached variant
used an isolated local sccache directory. Each variant had two warmups followed by
ten measured samples, alternating execution order on successive pairs. No other
build or numerical benchmark ran during sampling.

| Compilation | Median elapsed | Observed range |
| --- | ---: | ---: |
| Direct compiler | 183.1 ms | 178.1–197.8 ms |
| Warm sccache | 108.5 ms | 101.5–118.3 ms |

All 24 compilations succeeded. The 12 cached requests recorded one miss and 11
hits, with no failed compilations or cache errors. An initial attempt to pass the
mise compiler shim to sccache failed before compilation; the measured runs used
the real compiler path returned by `rustup which rustc`.

These measurements demonstrate local cache reuse for one dependency. They do not
estimate complete Cargo build time, link time, remote-cache transfer overhead, or
hosted-CI improvement. CI retains Cargo artifact caches, disables incremental
compilation as before, and reports sccache statistics. Final linking remains
outside sccache's Rust cache.

## Reproducible environments

The Python benchmark environment is separate from binding development, installs
from its checked-in uv lockfile, and receives the built extension wheel without
resolving dependencies again. Python, Rust, uv, Task, Lefthook, and Ruff validation
versions are pinned; preflight verifies local/CI version alignment. A separate
hosted job checks the latest stable Rust compiler.

Julia 1.10.11 generated the checked-in project/manifest. CI restores that environment
and caches its package depot and precompiled modules using the runtime and manifest
identity. Benchmark drivers select the project unless `JULIA_PROJECT` is supplied.
Package restoration and precompilation succeeded locally.

Full hosted matrix execution and GitHub remote-cache performance require a hosted
run. Local validation results are reported with delivery; this report makes no
cross-library or numerical-throughput claim.

## Validation

- `task ci` passed: default Rust suite (132 unit tests, 380 integration tests,
  six doctests, four pre-existing heavy tests ignored), optional Basin validation,
  portable examples, legal/completion/documentation checks, and both editable and
  isolated-wheel Python tests (88 passed, one optional workflow module skipped
  in each environment).
- `task preflight` passed, retaining the existing narrow informational RustSec
  exception. No new audit exemptions were added.
- `task benchmarks:preflight` passed its release Rust sleepstudy smoke; R was
  skipped because Rscript is unavailable. Smoke timing is not speed evidence.
- The benchmark/toolchain test suite passed: 28 tests passed and two R-dependent
  tests were skipped.
- The new Python environment restored from its lock, built and installed a wheel,
  ran `comparisons/sleepstudy.py`, and passed third-party dependency audits before
  and after wheel installation. The unpublished local `lme-python` development
  package cannot itself be looked up by pip-audit.
- Julia restoration/precompilation, sleepstudy and CBPP examples, and Julia
  comparison formatting passed. R comparison formatting was skipped.
- Bare-library compilation and `cargo check --locked --features polars-lazy --lib`
  passed. Workflow actionlint, pinned Ruff checks for the changed scripts, Markdown
  links (48 documents, 947 targets), and diff whitespace checks passed.

The hosted OS/Python matrix, production heavy cases, and remote cache transfer
performance were not run locally. No statistical throughput claim is made.
