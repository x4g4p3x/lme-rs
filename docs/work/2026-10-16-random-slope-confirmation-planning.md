# Random-slope confirmation design and resource planning

## Status and context

Created prospectively 16 October 2026 (Europe/Berlin). Status: **planning milestone
implemented, validated and reviewed for authorized delivery**.
Baseline `2d85393a06484e9fda6f776ddb26e7cf5da0187d`, pushed with hooks/preflight;
HEAD, tracking and live remote agree and the checkout was clean. The user
authorized this planning step after delivery of the completed
[numerical-controls milestone](2026-10-16-random-slope-numerical-controls.md).
No full confirmatory campaign, package release or external publication is launched
by this task. Preserve all completed v1/v2 and 0.2.7 evidence.

## Scope and acceptance criteria

Produce a concrete, reviewable confirmation design: 72 proposed primary claims
(36 balanced interior null cases times Satterthwaite/KR), familywise uncertainty,
exact count acceptance intervals, global decision-power planning, failure and
near-band-edge sensitivity, reference-optimizer policy and bounded resource costs.
Null coverage is the paired complement, not another independent primary claim.
Do not estimate a supposedly calibrated true rate from the 20-replicate pilot.

Planning assumptions declared before calculation: familywise alpha .05,
rejection band [.04,.06], proposed global decision power >=.90 at ideal true rate
.05 with no failures. Show conditional true-rate/failure sensitivity explicitly.
Use a grid of candidate replication budgets; select the smallest passing TESTED
budget, not a claimed global minimum. No independence assumption between claims.

Investigate the nine retained nonoptimal R fits only as separately receipted
post-study reference diagnostics. Evaluate a prospective comparator policy based
on likelihood/numerical checks, never p-value agreement. No changes to v2 records.

## Intended implementation and verification

Implement a separate planner and inputs with 72-claim Bonferroni Clopper–Pearson
rules and union-bound decision power. Validate boundary counts and power using
independent small exact/rational binomial identities plus R's beta/binomial
functions. Test zero/all counts, failure envelopes, empty acceptance ranges and
malformed inputs. Include power near band edges and missing outcomes; do not
silently assume ideal rates or missing-at-random validity in actual decisions.

Use matched R optimizer diagnostics on the selected nine v2 cases with fixed
variants/starts saved before refits; retain all attempts/failures. Distinguish
empirical success on selected cases from a future guaranteed comparator.

Use current pilot wall/disk measurements to propose primary-only and broader
budgets with clear extrapolation limits. Specify streaming storage/analysis,
source/native/environment freezes, independent held-out seeds, all-attempt failure
rules and independent review requirements before a full campaign.

Run focused planner/control tests, the scientific/script suite, script lint,
required comparison formatting and documentation checks. No Rust/binding changes
are planned; reuse unchanged engine checks. Do not modify frozen study sources.

## Findings and observed validation

The [proposal](../LMM_RANDOM_SLOPE_CONFIRMATION_PROPOSAL.md),
[planning report](../../scientific/RANDOM_SLOPE_CONFIRMATION_PLANNING.md),
[calculation](../../scientific/RANDOM_SLOPE_CONFIRMATION_PLANNING.json) and
[internal AI review](../LMM_RANDOM_SLOPE_CONFIRMATION_PLANNING_REVIEW.md) now
provide the concrete design and next implementation contract.

- 22,000 is the smallest tested ideal-case budget meeting the declared >=90%
  global decision-power target. After sensitivity inspection, propose 30,000
  per primary case: joint bound .982000016 under fixed .1% missingness and
  conditional evaluable rejection probability .05. This is an assumption, not
  an observed pilot calibration rate or uniform power over [.04,.06]. The old
  48-claim/20,000-replicate design remains historical.
- The primary decision bounds observable `evaluable AND reject` and `unevaluable
  OR reject` endpoints with 144 one-sided CP bounds. No MAR or between-method
  independence assumption is needed for envelope coverage. Exact rational and
  independent R calculations verify 432 rows; maximum probability difference
  2.21e-13, all count intervals match.
- Four fixed R variants were receipted before refits on exactly the nine retained
  above-oracle references. All 36 attempts remain; objective-only selection
  recovers 9/9 numerical gates, maximum objective gap 1.71e-13. Tight-default alone
  recovers 3/9. No pilot outcome changed. The final source-matching diagnostic is
  separate from an initial identical numerical run; the initial executed source
  is archived to account for post-receipt formatting. Explicit fixed-order ties
  and inference-blind selection have regressions.
- Independent review identified an optimistic pilot-mixture resource proxy.
  Final estimates use case-specific costs and include a stated fourfold baseline
  R proxy for the proposed fixed bank. Primary 30,000 plus secondary 2,000 and
  R subset: 1,196,000 candidate/94,000 reference datasets, 11.73 phase-wall proxy
  hours; twofold provision 23.46 hours/8.42 GB decimal raw storage. Analysis,
  numerical audit, environments and archives are additional. Native RAM remains
  unmeasured; a fresh resource rehearsal is required.
- The local 140,185-byte, 25-entry raw archive passes SHA/CRC checks, with matching
  source/input/environment identities; its public descriptor explicitly says
  not uploaded. Preserve all drafts and completed study evidence.

Validation completed in this task: nine focused planner/reference-policy tests;
all 98 script tests including both R-dependent tests (no skips with R on PATH);
`task lint:scripts`; `task lint:comparisons:required` with R and Julia available;
`task docs:check` (portable examples, six doctests and rustdoc); final diff review
and raw/source/archive checks. The first script-suite invocation skipped two
R-dependent tests because PATH lacked R; the complete rerun passed them. No Rust
or binding sources/dependencies changed; engine checks from the delivered
milestone remain applicable. No full CI, full confirmation, release or public
archive upload ran. The user authorized commit/push of this planning milestone
and continuation into the streaming implementation on 16 October 2026. Reviewed
the complete pending scope and receipts before delivery; keep hooks enabled.

## Next action

Implement the proposal's new streaming confirmation harness with durable
input/start/outcome chunks, all-attempt counts, failure-preserving resume,
bounded-memory analysis and generic four-variant R eligibility. Test crash,
truncation, duplicate-key and worker-order behavior using controlled fixtures.
Run a fresh small rehearsal in a separate namespace, measure actual native RAM,
time and disk overhead, and complete an independent implementation review.
Freeze actual sources/protocol and obtain authorization only once the concrete
campaign and resource caps are ready. Do not scale the v2 in-memory runner
unchanged or start full confirmation from this planning proposal alone.
