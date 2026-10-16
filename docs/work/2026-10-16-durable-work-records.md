# Durable work record workflow

Created and updated 16 October 2026 (Europe/Berlin). This is a retrospective setup
record, written after the workflow documents were drafted. The user approved the
approach before editing. Status: **implemented and validated for delivery**.
The user authorized committing and pushing this work on 16 October 2026;
this record is reviewed before that delivery. Git history identifies its commit.
Baseline: `615ea2528f90b148229b77491ade6f2c9f668c40`, with the scientific-audit
implementation and documents already present as uncommitted changes, preserved.

## Scope and acceptance criteria

Require one living repository record per substantial task or milestone: establish
scope and checks before implementation, capture material decisions as work proceeds,
review/update before each commit, and read relevant records on later tasks. Provide
a discoverable index and template, and reuse the existing scientific milestone plan.
No personal-memory writes or release is part of this setup. Commit/push delivery
was authorized separately after implementation and validation.

## Implementation and decisions

- Added the lifecycle and discovery rules to [AGENTS.md](../../AGENTS.md#durable-work-records).
- Added the [index](README.md), [template](TEMPLATE.md) and a link from the
  [documentation landing page](../README.md#contribute-and-maintain).
- Registered the [existing scientific plan](../SCIENTIFIC_VALIDATION_NEXT_STEP_PLAN.md)
  and added its retrospective findings/delivery context instead of duplicating it.
- Chose one record per meaningful task, reused across commits, to avoid document
  churn. Trivial edits need no new record; changed facts still need correction.
- Kept hooks unchanged: agent instructions require record maintenance, while a
  hook cannot establish the truth or completeness of a scientific/task narrative.
  Historical records must be rechecked, and sensitive context must stay out.

## Verification and limitations

Selected `task docs:check` and `git diff --check` for this instruction/documentation
change. Both passed including the final index/setup-record additions:
66 Markdown documents/1,081 local targets, 62 script tests, six doctests,
examples and API documentation. No new Rust/Python implementation checks are required for this
documentation-only setup; earlier scientific checks belong to their own record.

## Next action

The record and final diff/check results were reviewed for authorized delivery;
commit and push hooks must remain enabled. Apply this workflow to later substantive
tasks. The initial index is not an exhaustive archive; register older plans when
they become relevant. Personal AI memory remains a separate mechanism.
