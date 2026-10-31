# Dependency pull requests

## Status and context

Created prospectively on 30 October 2026 (Europe/Berlin). Integration is in
progress in an isolated worktree based on `c4f7d7c`; the primary checkout has
unrelated work. No integration commit or PR has been created.

## Scope and acceptance criteria

Evaluate open dependency PRs #49–#52, repair actionable blockers, preserve the
primary checkout, and deliver only validated changes. Retain original PR heads in
integration history and verify fresh hosted checks and the exact head before merge.
No release is included.

## Intended implementation and verification

Combine compatible PRs #49, #50 and #52 and repair the shared Python security
audit failure. Inspect Basin 3's breaking changes separately; preserve optimizer
semantics and the supported Rust minimum if migration proves feasible. Run the
applicable preflight, CI, audits and affected tests, and normal Git hooks. Inspect
and document any failures or skipped checks. Do not merge with a failed audit.

## Material findings and decisions

All four original heads fail the Python lock audit for msgpack 1.2.1
(`CVE-2026-114613`, fixed in 1.2.3). Basin #51 also fails compilation because
its 3.0 API removes imports used by the optimizer. Its minimum-Rust job fails
for the same compile errors; it does not establish an MSRV incompatibility.

## Implementation and observed validation

The heads of #49, #50 and #52 were combined without conflicts. The Python lock
was updated to msgpack 1.2.4; the benchmark lock already had 1.2.3. The checked
fuzz lock's smallvec was also updated from 1.15.1 to 1.16.4 to match the other
locks, then re-audited. Workflow pins retain full SHAs. The current rust-cache v2 ref has since advanced;
the proposed pinned commit remains a real upstream commit, rather than a claim
that it is still the moving ref's tip.

On 31 October, after space was recovered, the remote default branch and the
three PR heads were rechecked and had not moved. `task preflight` passed again
on this combined working tree, including the root and Python Cargo audits. The
fuzz Cargo audit and both locked Python audits also passed again. The restarted
`task docs:check` passed. `task test:comparisons` passed all 16 tests after rebuilding its wheel.
The full local CI and isolated-wheel pass were not run; normal hooks and fresh
hosted checks on the final integration head remain pending.

On this combined working tree, `task preflight` passed, including lint,
all-target compilation, root and Python Cargo audits, legal and metadata checks.
The fuzz Cargo lock also passed the same audit policy. Both locked Python
environments passed `pip-audit`. `task test:comparisons` was interrupted during
wheel compilation to avoid competing builds; it did not pass. `task docs:check`
was stopped during doctest compilation when free disk fell below 3 GB; it did
not pass. An attempt to clean the interrupted comparison's isolated build target
was rejected by the execution environment. Git hooks, full CI, and final hosted
checks remain pending. Original hosted checks do not validate the integrated head.

## Limitations and next action

Run the normal hooks, publish the compatible integration, and obtain fresh
hosted validation before any guarded merge. Investigate Basin independently; its new constraint,
solver state and termination interfaces require a behavioral migration. Do not
weaken convergence, bounds, or iteration contracts. No Basin migration is
included in this head.
