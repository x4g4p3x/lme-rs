# Integrate and merge all open PRs

## Status and context

- Created prospectively on 26 October 2026 (Europe/Berlin).
- Starting revision: `29811d4`, clean delivery worktree; default `master` is
  `93dc06acec14671348432a1155d3efc80b62261c`.
- Status: in progress; user authorizes integration and merge into the default branch.
- Preserve the primary checkout's 28 unrelated pending files.

## Scope and acceptance criteria

Integrate the published skill/security work and all three open dependency PRs,
retaining each original head in history. The default branch is named `master`,
not `main`; do not rename it. Validate the combined head, merge through a PR with
an expected-head guard, verify every intended PR is merged and local/tracking/live
default-branch identities agree. No tag or package release is included.

| PR | Original head | Change |
|---|---|---|
| #45 | `ba83f5ec71e55a534bb2741af95fb3164a1c89d0` | sprs 0.11.6 |
| #46 | `b75af4465d319e5b09839f4179b19fe490ec5775` | PyO3 0.30.1 |
| #47 | `08545cd0581bc2e355da5e73ab5611333f3ad6f6` | pinned install-action 2.87.29 |

## Intended implementation and verification

Combine these heads on an integration branch based on `29811d4`, preserving the
security repair rather than replacing lockfiles with vulnerable base versions.
Coordinate sprs in root, Python and fuzz locks. Adapt binding APIs if PyO3 requires
it and retain Rust 1.95 / Python 3.10 support. Inspect every failed hosted check;
fix actionable blockers without weakening tests or adding advisory ignores.

Use the documented Windows consolidated CI equivalent, all three Cargo audits,
both locked Python audits, affected bindings/wheels and MSRV checks, plus normal
preflight and Git hooks. Reuse unaffected checks from the immediately preceding
milestone where justified. Require fresh hosted success on the exact integration
head before merging; retain merge commits so original PR heads remain ancestors.

## Material findings and decisions

All three original PRs have failed release-policy and Cargo/Python audit checks.
The security repair in the starting branch addresses the known audits. Inspection
of the release-policy job log shows two mandatory harness tests fail because the
bare hosted Python lacks NumPy/SciPy. Run that harness in the existing locked
comparison environment rather than skipping the tests or relying on workstation
packages. Other hosted jobs on the original heads succeeded, but those results
are historical and do not validate the new integration head.

## Implementation and observed validation

The three heads combined without conflicts in an uncommitted octopus merge.
Root/Python sprs resolve to 0.11.6; the fuzz lock was updated from 0.11.4 to the
same version. Existing security fixes remain intact; all three Cargo audits pass.
PyO3 0.30.1 compiles but deprecates 13 `skip_from_py_object` flags because that
behavior is now the default. Remove those flags while retaining explicit
`from_py_object` opt-ins on cloned Python classes.

The exact replacement release-policy command passed all 161 harness tests, with
two existing unavailable-R skips. The initial PyO3 compilation passed with the
deprecation warnings; a warning-free rerun, full combined validation and fresh
hosted matrix remain pending. No merge has been made into the default branch.

## Limitations and next action

Preserve the primary dirty checkout. After remote delivery, synchronize its
default branch only if the pending files can be preserved exactly; otherwise use
a clean checkout for final default-branch verification and state the limitation.
