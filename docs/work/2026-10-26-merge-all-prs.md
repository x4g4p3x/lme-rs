# Integrate and merge all open PRs

## Status and context

- Created prospectively on 26 October 2026 (Europe/Berlin).
- Starting revision: `29811d4`, clean delivery worktree; default `master` is
  `93dc06acec14671348432a1155d3efc80b62261c`.
- Status: integration implemented; validation and delivery tracked in
  [PR #48](https://github.com/x4g4p3x/lme-rs/pull/48). The user authorizes merge
  into the default branch after verification.
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

The three heads combined without conflicts in octopus merge commit `393644b`.
Root/Python sprs resolve to 0.11.6; the fuzz lock was updated from 0.11.4 to the
same version. Existing security fixes remain intact; all three Cargo audits pass.
PyO3 0.30.1 compiles but deprecates 13 `skip_from_py_object` flags because that
behavior is now the default. Remove those flags while retaining explicit
`from_py_object` opt-ins on cloned Python classes.

The exact replacement release-policy command passed all 161 harness tests, with
two existing unavailable-R skips. After removing the deprecated flags, PyO3
compilation passed without warnings. Workflow lint and normal pre-commit/pre-push
hooks passed. Every original PR head and `29811d4` is an ancestor of `393644b`.

Rust 1.95 checks passed for all root targets with `basin,polars-lazy`, native and
stable-ABI Python bindings, and fuzz-target compilation. This is not a fuzz run.
At this checkpoint, 13 of 19 hosted validation jobs passed on `393644b`, including
release policy, all security audits, both compiler compatibility jobs, Python
3.10–3.14 on Ubuntu, and initial Windows/macOS binding jobs. Remaining hosted jobs
and the full local consolidated CI mirror were running without observed failures.

This is a commit-time checkpoint. Final immutable check results and the merge
state are recorded in PR #48; publication into `master` is gated on fresh success
for the final head, including this documentation update. No release is authorized.

## Limitations and next action

Preserve the primary dirty checkout. After remote delivery, synchronize its
default branch only if the pending files can be preserved exactly; otherwise use
a clean checkout for final default-branch verification and state the limitation.
