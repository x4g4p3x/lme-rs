# Dependency security remediation

## Status and context

- Created prospectively on 26 October 2026 (Europe/Berlin).
- Starting revision: `6130e26`, clean isolated delivery branch.
- Status: implemented and validated locally; authorized publication pending.
- The user authorized a separate dependency repair after the skill commit's normal
  pre-push audit failed. Preserve the separate primary checkout's unrelated work.

## Scope and acceptance criteria

Resolve the root, Python and fuzz dependency audit findings without new advisory
ignores or hook bypasses. Preserve model input behavior and Rust 1.95 support.
Include the locked Python development and comparison environments when their
additional package audit reveals vulnerable dependencies.
Keep this repair separate from the preceding skill/evidence commit, then publish
both on the delivery branch. No release or merge is included.

## Intended implementation and verification

Upgrade vulnerable TLS/Tokio packages within compatible ranges. Polars 0.55.2
requires zstd 0.13, while the security fixes require 0.14.3; avoid vendoring or
misrepresenting a patched package version. Disable unused default CSV
decompression. Use polars-arrow's uncompressed IPC reader for the Python bridge,
explicitly requesting uncompressed output from Python Polars. Retain projection,
categorical normalization, schema and all record batches. Rust callers needing
compressed CSV can explicitly enable Polars IO's decompression feature.

Add regression coverage for forced uncompressed IPC and multiple record batches;
run focused binding tests, all three audits, `task ci`, `task preflight`, and the
normal commit/push hooks. Inspect feature resolution and final remote identities.

## Material findings and decisions

Initial audit: rustls-webpki 0.103.15, webpki-roots 1.0.6, Tokio 1.50.0 and
zstd 0.13.3 / zstd-safe 7.2.4 / zstd-sys 2.0.16 have denied advisories.
Polars IO's default `decompress` feature and its `ipc` feature both enable zstd;
disabling only defaults is insufficient for Python. The public binding accepts
frames/tables, not serialized IPC files, so internal uncompressed interchange
preserves its supported inputs. No project API advertises compressed CSV reading.

An additional `pip-audit` found vulnerable certifi and filelock packages in the
Python development environment (five reports, including a duplicate certifi
advisory ID). The comparison environment also has older versions of those
packages. Update those packages in both UV locks and rerun the affected Python
checks and audits after synchronization; retain the initial failed audit.
The first UV update retained vulnerable filelock 4.1.0 on Python 3.10 because
patched releases require 3.11. Restrict the optional audit tool to Python 3.11+
(the existing canonical audit environment); retain Python 3.10 runtime and
source-test support. Remove that vulnerable resolution branch instead of hiding
it behind the local interpreter's audit result.
The comparison environment's own audit additionally reported 37 Pillow 12.3.0
advisories with fixes in 13.0.0. Update that comparison-only dependency and rerun
the affected workflow tests and its audit; the main binding environment already
audits clean after the certifi/filelock update.

## Implementation and observed validation

Root, Python and fuzz locks now select rustls-webpki 0.103.16, webpki-roots
1.0.11 and Tokio 1.54.1. Tokio's compatible Mio dependency updates to 1.2.4
where needed. The unused zstd family is absent from all three graphs; the Python
graph also drops LZ4. Existing advisory policy is unchanged.

Observed so far:

- All three `cargo audit --file <lock> --deny warnings --ignore
  RUSTSEC-2024-0436` invocations passed (411/350/344 dependencies).
- `cargo check --locked --manifest-path python/Cargo.toml` passed.
- First `task ci` failed before test execution with missing rlib-format
  dependencies in the reused build cache. A fresh, isolated target directory is
  used for the rerun; the failure is retained here, not counted as a test pass.
- The isolated ordinary `task ci` reached linking but hit Windows LNK1318/PDB
  limits while separate test executables consumed most available disk. It was
  stopped and only this task's package build artifacts were cleaned with
  `cargo clean -p lme-rs`. Validation now uses the documented consolidated Rust
  harness for default and Basin configurations, followed by all remaining full CI
  stages unchanged. This is a CI mirror with that substitution, not a claim that
  the ordinary `task ci` invocation passed. An untracked local driver imports the
  canonical CI runner; no repository CI policy or build profiles were changed.
- Regression tests cover explicit uncompressed transfer, multiple IPC record
  batches with coefficient/prediction comparisons, and malformed IPC errors.
  Both new regressions passed in the native Python test run.
- Consolidated default Rust tests passed: 145 unit tests, 408 integration tests,
  six doctests; four existing heavy integration tests remain ignored.
- Consolidated Basin tests passed: 159 unit tests, 411 integration tests,
  six doctests; the same four heavy tests remain ignored. Optional-feature
  Clippy, all-target compilation and Rust documentation also passed.
- Native Python tests passed: 93 passed, three skipped. The optional workflow
  benchmark module requires the separate comparison environment; two wheel-notice
  tests require a wheel in `python/dist`, while CI builds wheels in temporary
  directories. Two harness tests also skip because R is unavailable.
- Rust 1.95 checks passed for all root targets with `basin,polars-lazy`, native
  Python bindings, and `abi3` bindings. `cargo +1.95.0 check --locked
  --manifest-path fuzz/Cargo.toml` passed; this is compilation, not a fuzz run.
- The full CI mirror passed with the documented consolidated default/Basin test
  substitution. All other canonical CI stages ran unchanged: numerical examples,
  bindings, isolated native and stable-ABI wheels, dependency-only consumer
  examples, comparison tests, lint, all-target compilation, legal, documentation
  and completion checks.
- After refreshing the Python locks, reran native bindings, isolated native and
  stable-ABI wheels and consumer examples: each test environment passed 93 tests
  with the same three skips. All 16 comparison tests passed. After the separate
  comparison-only Pillow update, reran those 16 tests; all passed again.
- Final Python locks select certifi 2026.10.23 and filelock 4.2.4; the comparison
  lock additionally selects Pillow 13.0.0. No other Python package versions changed.
  Both configured Python 3.11 environments passed `pip-audit`. The local unpublished
  `lme-python` development package is not in PyPI and is excluded by the service;
  this is not an audit of the project's own source code.
- `uv tree --locked --python-version 3.10` confirms the compatible development/test
  graph excludes the audit tool/filelock; the 3.11 tree selects the patched packages.
  This resolves markers, rather than executing a Python 3.10 test suite locally.
- `task preflight` passed, including Cargo audits, legal/provenance, metadata
  dry-run and live token verification. Normal Git hooks remain enabled.

## Limitations and next action

Local validation cannot establish the hosted OS/Python matrix, four heavy ignored
Rust tests, unavailable R checks, or production-load gates. No advisory ignores
were added. The original checkout's 28 pending paths remain byte-for-byte unchanged.
Publish the separate dependency commit after `6130e26` using normal hooks, then
verify local, tracking and live branch SHA equality and a clean delivery worktree.
