---
name: bug-hunt
description: Find and fix reproducible correctness bugs with independent regression evidence. Use for bug hunts or correctness investigations; preserve review-only scope.
---

# Bug Hunt

Find consequential contract failures in a bounded area. A passing suite alone is not evidence that the contract is correct. Run directly; this skill does not request delegation.

1. Inspect Git state and relevant changes; preserve unrelated work. Read applicable AGENTS.md and select the repository-required checks. In lme-rs, use [its correctness guide](references/lme-rs.md). For review-only work, reproduce and report without editing production files.
2. Trace a public behavior through implementation and tests. State the expected contract and a plausible violation. Prefer independent identities, analytic fixtures, equivalent-input transformations, or cross-path agreement. Match rows, weights, offsets and model settings in comparisons. Distinguish unsupported inputs from defects; do not impose a finding quota.
3. Reproduce with the smallest realistic regression before fixing. Verify the command exit and a nonzero test count. Build failures, missing tools and timeouts are not reproductions. Include relevant invalid-input or boundary controls so the fix cannot simply accept everything.
4. Fix the cause with a coherent scoped change. Preserve precision, error behavior and statistical contracts. Do not loosen tolerances, swallow errors, or special-case a fixture to get a pass. Rerun the regression and adjacent checks; retain confirmed probes as independent tests.
5. Complete the repository-required validation tier, test registration, and any required changelog or records. Do not create process files that the task or repository does not require. Serialize builds sharing outputs; reuse passing checks only while their inputs remain unchanged. Review the final diff and report actual evidence, skipped checks, limits and remaining uncertainty. A bounded hunt is not an exhaustive audit.

Git publication and cleanup require existing user authorization; do not ask again for an authorized action. Keep hooks enabled, inspect hook changes, and verify local/tracking/live equality after a push. An authorized merge requires fresh checks, including security audits, on the current head. Enforce the inspected expected head SHA in the merge operation, then verify the merged state. Branch deletion needs worktree/later-commit checks, ancestry or content-equivalence evidence, and a lease against the inspected remote SHA. Do not infer release authority.
