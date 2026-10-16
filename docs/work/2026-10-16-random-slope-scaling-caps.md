# Random-slope scaling, resource caps and native interruption

Created before implementation, 16 October 2026 (Europe/Berlin). Status: engineering
milestone completed and independently reviewed; two numerical blockers retained.
Baseline `bda59f02216a6c47f5e53d5421b26d8b73da935c` is pushed with hooks/preflight;
HEAD, tracking and live remote match; checkout initially clean. The user authorized
this next step after delivery of the [streaming milestone](2026-10-16-random-slope-streaming-harness.md).

## Scope and prospective acceptance

Keep every earlier frozen source and study unchanged. Add a separate scaling
protocol/runner, reusing pinned durable-store, generator and R/statistical helpers
without runtime monkey-patching. Full confirmation remains disabled. No engine,
production dependency, release or full campaign is part of this task.

Before observations specify two fresh, source-frozen runs: all 94 cases at 10 and
20 replicates (940/1,880 candidate fits and 3,760/7,520 R attempts). Use four workers,
one library thread each, unique namespaces, all four reference variants and the
same numerical/method criteria. Do not overlap measurements with tests/builds.
Require independently checked inventories/counts/72 software decision rules and
transparent failures. These samples cannot confirm calibration.

Implement explicit per-phase time, disk and process-memory limits with measurable
abort receipts. Windows hard job committed-memory limits differ from resident
working set; identify the precise metric rather than calling either a whole-host
RAM bound. Disk is sampled with a declared interval/overshoot, fail closed when
unavailable, and keep a free-space reserve. Preserve started attempts on abort.

Freeze separate fault controls before execution: load the exact native candidate,
commit a deterministic attempt start, interrupt its worker/process tree, resume
only never-started keys, prove prior outcomes unchanged and no duplicate start.
Keep fault inputs/namespaces separate from the two scaling runs. Exercise low
time/disk/memory cap aborts with controlled processes/data before scientific runs.

## Intended validation and decisions

Choose caps and declare their semantics before observing scaling results. Verify
the underlying worker/environment/source identity and outcome inventory on resume.
Use independent exact count/R checks and AI source/result review. Compare fixed
vs per-attempt storage, fit/durable-I/O and analysis costs across the two budgets;
do not turn two points into a guaranteed full-campaign bound or a speed claim.
Secondary means remain descriptive: defer optional bootstrap intervals rather
than introduce an unreviewed bootstrap estimator into this resource milestone.

Run focused cap/native-fault regressions, full script tests, script lint, required
R/Julia formatting and documentation checks before the freeze; leave all earlier
frozen files byte-identical. Preserve unsuccessful executions and any subsequent
protocol/source changes with a fresh freeze/namespace. After results, specify
what remains before final confirmation freeze and campaign authorization.

## Findings, validation and next action

Implemented the separate runner, Windows job guard, three-manifest bank freeze,
native-loaded interruption control and command-bound capped analyses. Pre-freeze
AI review found and prompted fixes for an announcement publication race, live
descendants after successful root exit, missing command provenance, contradictory
resource receipts and namespace/bank identity. Regressions cover these cases.

The pre-freeze synthetic memory probe showed that this host's raw OS peak-job
charge can include a denied allocation: it is not proof of successful committed
or resident memory. Retain that counter, current committed-byte samples and limit
events separately; conservatively reject an above-limit raw charge. An initial
test assumed the raw peak stayed below the limit and was corrected to test actual
allocation denial. Explicit job termination also fixes Windows kill-on-close
returning a misleading zero exit status in the time-control test. These are
engineering probe observations, not frozen scientific data.

All 134 script tests passed with R available, including 16 new scaling/cap tests;
affected cap/scaling tests passed again after the final process-inventory changes.
Script lint, required R/Julia formatting and documentation checks passed. No
Rust/binding code changed. Source review resolved all freeze blockers.

Bank A froze all three manifests before generation. Its negative cap controls
passed and the native interruption worked, but the new marker audit rejected
the run: Windows virtual-environment launcher PID differs from the actual loaded
interpreter. Retain this failed control and its original source snapshots in
`benchmark-results/scientific/random-slope-pilot/20261016-scaling-a`; no n10/n20
scaling inputs were generated and no resume occurred. A read-only diagnostic
confirmed both durable rep1 hashes matched the announcement. Do not reinterpret
that failed attempt as accepted or silently resume it under changed sources.

Before replacement bank B, query the complete Windows job process list (including
nested children), require native PID membership and fold refreshed memory/events
into the interrupt-time guard. AI review verified the structure and scope; focused
tests passed, including a fixture with different root/native PIDs. The amended
protocol records this correction before new observations. Freeze all three bank B
roles under fresh `scaling-v1-20261016b-...` namespaces and the same 106 source
hashes, retaining earlier frozen sources unchanged.

Bank B negative controls and the actual native interruption/resume passed: three
inputs/starts/outcomes, two completed fits and one retained interruption, no R
attempts. Replicate 0 bytes/start and rep1 durable input/start were preserved;
only rep2 was newly fitted.

Both bank B measurements completed sequentially without concurrent tests/builds:
940/1,880 candidate starts/outcomes and 3,760/7,520 R starts/outcomes. Candidate
phases took 28.178/51.388 seconds, R phases 127.516/281.401 seconds. Separately
capped analysis and R verification passed with exact command/input/output hashes.
Both independent R checks reproduced binary counts in 564 rows and 72 primary rules;
source/journal identities match and original fits were not rerun or replaced.

Resource acceptance passed within the prospectively declared 900-second, 4-GiB
Windows committed-memory, 512-MiB sampled disk and 1-GiB free-volume limits. Maximum
sampled reference-job commit was 1,860.24 MiB; maximum sampled study disk 22.915 MiB.
Largest actual sampling gap was 0.2130 seconds. Physical journals grew from
9,080,832 to 14,745,600 bytes; compressed payload, other database storage and
analysis costs are separated in the [resource receipt](../../scientific/RANDOM_SLOPE_SCALING_V1_RESOURCES.json).
Two-point independent-input secants are descriptive, retain a negative R-time
intercept, and do not establish a full-campaign bound or speed claim.

Numerical readiness **did not pass**. Each run contains one converged candidate
rank-zero fit with a meaningfully inferior REML objective to the rank-one analytic
and selected-R solution. All methods of those fits remain failed; 939/1,879 other
pairs have no numerical or binary inference differences. All selected R fits
passed, while 11/23 unselected variant numerical misses remain recorded. Both
candidate counterexamples are outside the primary family; primary failure counts
are zero, but all 72 calibration rules fail at these small sample sizes. No
confirmation/calibration claim follows.

The [diagnostic fixtures](../../scientific/RANDOM_SLOPE_SCALING_V1_COUNTEREXAMPLES.json)
contain both exact synthetic input/designs, original candidate/reference fits,
analytic optima and provenance. Independent AI review confirmed the objective
gaps and identified a plausible near-zero-column recovery-guard mechanism; it
has not been regression-reproduced or repaired in this milestone. Keep the engine
and old frozen sources unchanged through publication.

Post-run AI audit independently regenerated all 2,820 responses, checked all
14,100 starts and 14,100 outcomes against immutable-read journals/audits, and
recomputed balanced active-set optima, every count/bound and secondary moment.
It confirmed exactly the two candidate misses and 11/23 unselected R misses.
Review corrected draft attribution: the R checker verifies binary counts and
bounds, while the separate AI audit verified secondary bias/RMSE/SE/width moments.

The 28,801,948-byte, 1,508-entry local archive includes failed bank A and completed
bank B, source freezes, journals, logs, raw audits and independent checks. SHA/CRC
verification passed; public original-file copies match bytes. The descriptor
explicitly says not uploaded. Installed environments and private context are
excluded. See the [report](../../scientific/RANDOM_SLOPE_SCALING_V1.md),
[review](../LMM_RANDOM_SLOPE_SCALING_REVIEW.md) and
[archive descriptor](../../scientific/RANDOM_SLOPE_SCALING_V1_BUNDLE.json).

## Next action

Follow [the concrete optimizer milestone plan](../LMM_RANDOM_SLOPE_OPTIMIZER_NEXT_STEP.md):
add failing public-API regressions for both archived cases, establish the actual
recovery mechanism, repair it with scale/budget/boundary coverage and required
Rust/performance validation, then build a newly identified candidate and run a
fresh independent numerical pilot. Preserve all original failures. Only after
numerical repair and production orchestration/analysis/storage provisioning may
the final confirmatory freeze and explicit campaign authorization proceed. No
new engine fix, full campaign, release or public archive upload occurred here.
The user authorized commit/push of this milestone and continuation into the
optimizer repair on 16 October 2026. The initial delivery is the pushed
`bda59f0` baseline. Final independent review verified all eight
capped jobs, native control/resume, public fixture/copy identities, archive SHA/
CRC/member bytes and all nine descriptive secants, with no delivery/provenance
blocker. Required documentation checks and final link checks passed. The initial
Windows help query needed UTF-8 output to print a theta symbol; rerunning with
`PYTHONUTF8=1` succeeded without changing frozen tooling. HEAD, tracking and live
remote matched `bda59f0` before this delivery; this commit covers only the new
milestone's sources/docs/receipts. Keep hooks enabled and verify the live remote
after pushing. No hooks were bypassed for the initial delivery.
