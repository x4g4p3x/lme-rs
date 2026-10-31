# Local CI build-artifact retention

## Status and context

Created prospectively on 30 October 2026 (Europe/Berlin). Started on 31 October from `93dc06a`; the primary checkout contains unrelated
pending work. A cleanup-only delivery checkout is based on `4bfb737`.

## Scope and acceptance criteria

The user requested post-local-CI cleanup for the inventoried projects, with a
seven-day age threshold. Preserve source, scientific and benchmark evidence,
active builds, other worktrees, and user-configured shared targets. A CI result
must not be represented as passing when validation failed.

## Intended implementation and verification

Identify each project's actual local CI entry points and ownership of target
directories. Design and test a mechanism that cannot race concurrent Cargo
builds, follow links outside its owned target, or interpret an unchanged file
timestamp as proof that a Cargo artifact is unused. Validate with isolated
fixtures, including active-build, links, shared-target, seven-day boundary,
failure, and hosted CI cases. Apply each repository's required CI and docs
checks. Do not run deletion against existing project targets while testing.

## Material findings and decisions

The user approved using CI-owned dated directories and retaining existing shared
targets. The inventory includes several independent repositories, with different CI
entry points and persistent-cache expectations. Rust's build artifacts can be
reused without changing their modification times, so age alone is not evidence
that a file is unused. Cargo's `clean --profile` removes a whole profile rather
than selecting stale artifacts, and its whole-profile path does not obtain a
build lock. The benchmark-results tree includes preserved evidence and must not
be swept by a broad recursive cleanup. CI-owned generations require a fresh build
when rotated; their last CI use, not per-artifact timestamps, determines age.
The initial managed directory was inside `target`, where ordinary `cargo clean`
could remove an active CI build. It was moved beside `target` before delivery.
Earlier validation used the original location; those generated directories are
retained and are not swept by the final policy.
The standard-library helper and its tests are copied into each independent
repository so local CI has no network or sibling-checkout dependency. Keep the
copies synchronized when maintaining this behavior.

## Implementation and observed validation

Local CI now runs in owned weekly target generations, guarded by an exclusive
cross-process lock. Completed generations whose last CI use was at least seven
days ago are removed after a run; interrupted and unowned generations remain.
Configured targets, unavailable offline metadata and hosted CI bypass the
policy. The standalone runner and Task use the same path. Existing targets and
benchmark evidence remain outside the cleanup root. Parallel managed CI in one
checkout is rejected, and each new generation incurs a cold build.

The original cross-project implementation was tested in the working checkouts. Focused tests exercise the
seven-day boundary, ownership, interruption, a real Windows junction, process
locking, configured targets, resolved-path containment and runner delegation.
Script lint and the resumed documentation check passed. `task preflight` reached
Cargo audit but failed on the old baseline's existing advisories: six
vulnerabilities and 11 denied warnings; no security ignore was added. An initial
docs run and BioBit-rs CI were interrupted and do not count as passes; their
reruns passed. BibleQuoteBench's full coverage gate passed (98.50% lines).
Finding Logos `npm run verify`, formatting, and an isolated-copy WebAssembly
build passed. BioBit `task ci`, `task ci:tools` and preflight passed; final-location
CI passed again after checking all four Rust workspaces. The agent, MCP and SCP
CI checks passed at the earlier cache location. SCP's
separately required `cargo audit` found ten vulnerabilities in its pre-existing
lockfile; dependency updates are outside this change. All
eight projects passed a metadata-and-command smoke
check using the final sibling cache location, and the final isolated retention
tests passed in each project. The Finding Logos verification and an isolated
WebAssembly build also passed using the final location.

The managed target is shared by the root Rust workspace and Python binding
builds during local CI. The local fast CI path at the final location reached Python tests but failed
`test_lmer_sleepstudy`: the output contained intercept variance 612.0911 while
the test expects the string 611.9033 (92 passed, one skipped, one failed). The
primary checkout contains unrelated numerical changes; the same focused test
passed in a clean, newer integration checkout, but that comparison does not
isolate the cause. The run omitted the isolated-wheel pass. The separate preflight security audit remains failing on the old lock.

The cleanup-only delivery checkout based on `4bfb737` passed all 20 focused
retention and runner tests, `task preflight` (including security audits),
`task docs:check`, and `git diff --check`. Its preflight used an explicitly
configured existing Cargo target and therefore bypassed retention during that
run. Full `task ci` and its isolated-wheel pass were not rerun there. The five
standalone adapter, benchmark and site repositories were committed and pushed;
BioBit-rs passed its cleanup-only CI and was pushed separately.

## Limitations and next action

Finish delivery with normal hooks and verify the remote branch. The cleanup lock only
coordinates participating CI wrappers: unrelated commands must not use managed
target generations. Offline metadata failure skips retention rather than changing
the configured build. Existing shared build targets need separate manual cleanup.
