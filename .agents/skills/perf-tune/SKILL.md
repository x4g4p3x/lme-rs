---
name: perf-tune
description: Identify measured bottlenecks and implement correctness-preserving performance improvements in an existing codebase, using profiling, comparable before-and-after benchmarks, and repository-required validation. Use for optimization, latency, throughput, allocation, or memory work; respect analysis-only requests.
---

# Performance Tuning

Deliver a measured improvement to the requested workload, with reproducible
evidence and preserved behavior. A cheaper kernel or fewer allocations does not
by itself establish a faster complete operation.
Run this workflow directly; invoking it does not invoke a delegation skill.

## Establish the scope and baseline

- Inspect the branch, working tree, and relevant diff. Preserve unrelated work;
  do not reset, stash, switch branches, or update dependencies as routine setup.
- Read applicable `AGENTS.md` and performance guides. Select the required checks
  before editing. In `lme-rs`, read [references/lme-rs.md](references/lme-rs.md).
- Follow the requested mode. Analysis-only work produces measurements and
  recommendations without changing production code. For implementation work,
  carry supported changes through correctness and performance validation.
- Define the metric, workload, timing boundary, and any supplied target. Without
  a specific target, choose a bounded, consequential path using current evidence.
  Distinguish elapsed time, throughput, peak memory, and allocation counts.
- Establish a baseline before editing. Record the revision and relevant dirty
  diff, inputs or seed, compiler, build profile/features, runtime versions,
  hardware, thread settings, and benchmark command. Preserve raw samples and
  results so the original workload can be rerun against the candidate.
- Prefer existing harnesses. Profile to distinguish preparation, repeated work,
  the core operation, and result construction. Collect instrumentation separately
  from throughput measurements when it changes execution cost.

## Change the measured cause

State the bottleneck, proposed mechanism, and behavior that must remain invariant
before changing code. Focus changes on costs the measurements actually expose.

Useful approaches include removing repeated work, reducing allocation or copying,
reusing valid intermediates, improving data layout, and selecting algorithms that
fit the workload. Choose from evidence rather than prescribing a cache, parallel
implementation, or new dependency for every task.

- Preserve public results, error behavior, precision, convergence, and supported
  input shapes. Do not gain speed by silently skipping required work, loosening
  tolerances, shrinking the problem, or relaxing iteration budgets.
- For caches and reusable workspaces, identify the complete validity key and
  lifetime. Test changing inputs and controls, invalidation, ownership, and any
  concurrent use affected by the change; a repeated identical input is insufficient.
- For numerical changes, use independent identities, trusted fixtures, or the
  general path to check objective values, estimates, uncertainty, and diagnostics.
  Exercise relevant boundary, singular, weighted, and offset cases.
- Keep existing memory/shape gates and fallback behavior for specialized paths.
  Test the optimized and fallback paths where their behavior changes.
- Add meaningful regression coverage when the change alters a correctness-sensitive
  path. Avoid wall-clock assertions in ordinary unit tests; use benchmark evidence
  for speed and deterministic checks for behavior.
- Keep attempts separable. If a candidate fails correctness or shows a material
  regression, revise it or remove only that candidate's edits, preserving existing
  user changes. Do not leave a speculative optimization as a verified improvement.

## Measure equivalent work

- Compare baseline and candidate with identical inputs, settings, build modes,
  features, and thread limits except for the intended change. Track source and
  binary identity; do not accidentally compare a stale executable or another backend.
- Separate startup, preparation, cold execution, and reuse. Compare complete
  operations as well as the changed phase. Do not compare a cached/prepared path
  with fresh construction as if their timing boundaries were equivalent.
- Serialize timing runs and avoid competing builds or CPU-intensive checks.
  Warm up as appropriate, retain repeated samples, and use independent sessions
  with alternating execution order when drift could change the conclusion.
- Follow the repository's sampling and acceptance rules. Inspect distributions
  and uncertainty, not just the fastest sample or a single median. Tiny gains
  within noise are inconclusive; do not select favorable runs or discard failures.
- Check representative adjacent workloads and adverse sizes or structures that
  the changed path serves. Explain runtime/memory tradeoffs and limit the claim
  to measured cases. Missing reference runtimes are skips, not parity evidence.
- Report baseline and candidate values, units, sample counts, relative change,
  variability, correctness checks, and regressions. Retain commands and raw
  evidence. A focused win does not establish a universal speedup or completion.

## Validate and deliver

- Run focused checks, then the repository-required tier for the final scope.
  Register new tests in any consolidated harness. Reuse passing results only for
  unchanged code and dependencies; rerun affected checks after further edits.
- Add the repository's required changelog entry for user-visible performance
  improvements, with measured scope. Update benchmark evidence or documentation
  when making maintained performance claims; do not adjust generated completion
  percentages or narrow commitments to earn credit.
- Review the final diff and report the improvement, mechanism, validation,
  skipped/failed checks, and limits. If measurements remain inconclusive, say so
  and deliver the evidence rather than inventing a speedup.
- Skill invocation does not authorize commits, pushes, PRs, merges, releases, or
  benchmark publication to an external service. Honor explicit authorization
  already given for the current work without asking again. When authorized, keep
  hooks enabled, inspect hook edits, and verify local, tracking, and live remote
  SHAs after pushing.
