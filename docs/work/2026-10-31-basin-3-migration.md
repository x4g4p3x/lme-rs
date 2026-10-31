# Basin 3 dependency migration

## Status and context

Created prospectively on 31 October 2026 (Europe/Berlin) from `c4f7d7c` in an
isolated worktree. The primary checkout contains unrelated pending work. PR #51
proposes Basin 3.0.0 but fails feature-enabled compilation on the old interface.
No release is included.

## Scope and acceptance criteria

Integrate PR #51's exact head and migrate the optional optimizer while preserving
bound projection, stopping tolerance, convergence reporting, invalid-cost
handling, recovery probes, shared iteration budget, and Rust 1.95 support.
Keep original PR history and deliver only after local and hosted validation.

## Intended implementation and verification

Inspect Basin 3 interfaces and defaults, add independent regression coverage for
any changed stopping or projection behavior, and adapt the backend without
weakening existing tests. Run focused Basin tests, `task lint`, `task rust`,
`task basin:check`, documentation and `task preflight`, plus applicable fair
harness workloads as required by `OPTIMIZATION.md`. Re-audit all affected locks
and require fresh hosted success on the immutable integration head. Record
failures, skips and the precise final state; do not claim unmeasured speedups.

## Material findings and decisions

The old backend imports removed constraint, simplex-state and termination
interfaces. Basin 3 also enables its own simplex convergence by default; the
existing library's strict sample-standard-deviation rule must stay authoritative.
Its projected solver wraps callback errors in a constraint-solve enum; preserve
the original objective error rather than converting it to an opaque string.

## Implementation and observed validation

`cargo check --locked --features basin --lib` reproduced the three import/type
errors on the proposed dependency. Added independent objective tests for a
strict tolerance equal to the initial simplex standard deviation and a tight
quadratic tolerance that must not be preempted by simplex-collapse defaults.
The Basin 1 baseline passed 14 backend tests, including both stopping-rule
regressions, in a separate detached worktree. A third test preserving the
concrete objective error type also passed after a forced rebuild; an initial
zero-test selection was discarded. The initial Basin 3 interface adaptation
compiled, but two of 15 backend tests failed: default simplex stopping preempted a tight tolerance and prevented
boundary recovery. The four default simplex tolerances were disabled to preserve
the existing strict stopping rule. All 15 focused Basin 3 backend tests then
passed, including both previously failing cases. `task basin:check` passed
feature-enabled Clippy, all-target compilation, the full test suite, and docs.
Release golden parity passed both without the feature and with Basin 3 (two
tests each). A Rust-only fair-harness run completed sleepstudy REML, weighted
sleepstudy REML, and the 100,000-row random-slopes case with three warmups and
11 repeats: all recorded fits converged and the report recorded no failures.
The untracked, locally available report is `benchmark-results/basin3-migration.json`;
it identifies the dirty pre-integration source by SHA-256
`a73e30f9dd69b12c8cdbb1ca2533199194548cd407bbd9324d5c4652cba242f7`.
This is a single Rust-only run, not an equivalent before/after performance
claim. Existing PR checks also fail a separate Python security audit; the final
integration has not yet been validated.

## Limitations and next action

The migration commit passed normal pre-commit hooks. The updated default
branch was integrated, retaining both work-record index entries. On this
combined tree, `task preflight` passed, including both Cargo audits; both locked
Python environments also passed `pip-audit`. The Basin feature suite was repeated against the changed dependency locks:
`task basin:check` passed again. Documentation links and staged whitespace
checks passed on the combined tree.
`task rust`, minimum-Rust, and hosted validation remain pending on the final
integration; preflight included `task lint`. The prior base's Python lock has a
known audit finding; do not merge until the repaired lock and fresh checks pass.
Preserve unrelated work in the primary checkout.
