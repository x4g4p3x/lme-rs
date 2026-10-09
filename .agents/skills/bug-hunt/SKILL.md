---
name: bug-hunt
description: Find and fix reproducible correctness bugs in an existing codebase using focused exploration, independent regression evidence, and repository-required validation. Use for a bug hunt or correctness investigation, especially numerical libraries and post-fit APIs; respect review-only requests.
---

# Bug Hunt

Deliver confirmed defects with minimal root-cause fixes, durable regression coverage,
and an honest account of what was examined. A passing suite alone does not establish
correctness; use contracts and independent identities to find gaps in existing tests.
Run this workflow directly. Invoking it does not invoke Grok or a delegation skill.

## Establish the scope

- Inspect the current branch, working tree, and relevant diff before editing. Preserve
  unrelated work; do not reset, stash, update dependencies, or switch branches as setup.
- Read applicable `AGENTS.md` instructions and the repository's validation map. Select
  required checks before editing. For `lme-rs`, read [references/lme-rs.md](references/lme-rs.md).
- Follow the requested mode: implement confirmed fixes for a fix-oriented hunt; for
  review-only requests, reproduce and report without changing production code.
- Prioritize consequential public behavior, recently changed areas, and boundaries
  between modules. Choose a bounded initial area and broaden when evidence warrants it.
  Do not spend the hunt scanning the entire repository or impose a quota of findings.

## Investigate contracts, not just suspicious code

Use targeted searches to connect public entry points, implementation, and existing tests.
Write down the expected behavior and a plausible violation before constructing a probe.
Distinguish documented limitations from defects.

Useful probes include:

- Equivalent inputs: change units, row ordering, data types, or irrelevant columns
  where the contract says the result should be preserved or transform predictably.
- Cross-path agreement: compare direct and reusable fits, Rust and Python APIs,
  single and batched operations, or prediction and post-fit inference.
- Independent identities: compare nested models, analytic cases, or trusted fixtures
  rather than copying the implementation into the expected result.
- Boundary controls: exercise missing values, empty inputs, singular designs, unused
  fields, and numerical extremes alongside nearby valid cases. Expect a clear error
  for unsupported or unrepresentable inputs rather than imposing arbitrary success.

## Confirm each finding before fixing it

1. Create the smallest realistic regression that exercises the public behavior.
   Prefer an existing test target; use a scratch probe if a hypothesis is still weak.
2. Run it against the unfixed implementation. Inspect the exit status and actual test
   count. Compilation errors, missing tools, timeouts, and zero selected tests are not
   reproductions. Record the observed failure and independent expected behavior.
3. Trace the failure to its cause. Include a negative control where a permissive fix
   could accidentally accept invalid input, such as retaining collinearity rejection
   while fixing sensitivity to predictor units.
4. Make the smallest coherent fix that preserves the surrounding contract. Avoid
   unrelated refactors, arbitrary tolerance relaxation, swallowed errors, or special
   cases tied to the fixture. Explain numerical thresholds relative to their scale.
5. Run the regression again and test adjacent affected behavior. Promote confirmed
   probes into durable tests; do not commit exploratory failures or unsupported claims.

For reference-library comparisons, match inputs and model settings first. Explain
tolerances from numerical precision or statistical approximation; do not loosen them
merely to obtain a pass. If a finding remains uncertain, label it as a hypothesis and
state the missing evidence.

## Validate and finish

- Register new tests in any consolidated test harness. Add the repository's required
  changelog entry for user-visible fixes.
- Run focused checks first, then the required tier for the final scope. Cross-module
  changes require broader checks. Reuse passing results for unchanged files and
  dependencies in the same task; rerun affected checks after further edits.
- Serialize builds sharing an output directory. Capture verbose logs and inspect
  summaries and exit codes. Diagnose infrastructure failures before changing code.
- Review the final diff for scope, error behavior, test independence, and accidental
  formatting or generated-file changes.
- Report confirmed bugs and their impact, fixes, validation, ignored/skipped checks,
  and remaining uncertainty. Link files or a PR when available. Do not claim an
  exhaustive audit, performance parity, or completion from a bounded hunt.

## Git delivery, only within the user's authorization

A bug hunt does not by itself authorize committing, pushing, opening a PR, merging,
releasing, or deleting branches. Preserve authorization already given in the current
task and complete authorized delivery without asking again.

When authorized, use the repository's branch conventions, retain hooks, and inspect
any changes hooks stage. Verify the pushed SHA against the tracking ref and live remote.
For an authorized merge, review the final PR and require fresh checks for its current
head, including security audits. Inspect exact failed logs, fix actionable failures,
and merge with the expected head SHA. Verify the merged state and synchronize the
local base branch only when ancestry and local changes permit it.

If cleanup is requested, check for later commits and active worktrees first. A squash
merge needs content-equivalence evidence because commit ancestry alone will differ.
Delete the remote branch with a lease against its inspected SHA and verify absence;
do not automatically delete other branches or publish a release.
