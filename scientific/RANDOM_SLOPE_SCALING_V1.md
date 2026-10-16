# Random-slope capped scaling rehearsal v1

16 October 2026. **Resource execution passed; numerical readiness did not.** Two
candidate fits report convergence at an inferior rank-zero solution. Preserve
them as failures and fix/revalidate the optimizer before confirmation. These
engineering samples establish neither nominal calibration nor release readiness.

The [prospective protocol](../docs/LMM_RANDOM_SLOPE_SCALING_PROTOCOL.md) specifies
all 94 designs at 10 and 20 replicates, four workers and the unchanged four-variant
R bank. Bank B froze all three roles and **106 source hashes** before generating
any inputs, under distinct `scaling-v1-20261016b-n10-a`, `-n20-a` and `-native-a`
namespaces. The earlier 98-source streaming freeze remains unchanged. Candidate
engine/wheel/native identity is the previously validated v2 engine from
`580b270edfcfd8f544818f3ff8171aacbd30d099`, version `0.2.8.dev0`, with native SHA
`2ed6b683889945bf25e772ac6d7feaddf60a5592cad7ec25053328fff4999daa`.

## Numerical observations and retained failures

| Inventory | 10 replicates | 20 replicates |
|:---|---:|---:|
| Candidate starts/outcomes | 940 | 1,880 |
| R starts/outcomes, four variants | 3,760 | 7,520 |
| Candidate balanced numerical misses | 1 | 1 |
| Selected-reference numerical misses | 0 | 0 |
| Eligible paired comparisons, each method | 939 | 1,879 |
| Paired numerical/rejection/coverage differences | 0 | 0 |
| Unselected R variant numerical misses | 11 | 23 |
| Primary failed inferences across 72 claims | 0 | 0 |

Every candidate/R attempt established the runner's convergence flag, which alone
did **not** establish a correct optimum. All three methods of each flagged
candidate remain failed in all-attempt reporting; no alternate candidate fit was
substituted. The fixed criterion-only reference selector found an eligible
numerical solution for all 2,820 datasets. The 34 unselected R misses remain in
variant counters and the raw flag logs; convergence messages are insufficient
for the reference software as well. No R exceptions, nonconvergence or interrupted
attempts occurred in these two scaling runs.

There are 920/1,840 balanced datasets with independent spectral numerical
controls; the 20/40 imbalanced datasets have no balanced oracle and rely on the
other declared checks. Paired results exclude the two flagged candidate fits;
zero paired differences must not hide these exclusions. Both counterexamples
are stress cases outside the 72-claim primary family. All 72 primary software
rules fail their narrow calibration band at these small sample sizes, as
expected; no calibration pass or pooled new primary family is claimed.

The [complete diagnostic fixtures](RANDOM_SLOPE_SCALING_V1_COUNTEREXAMPLES.json)
retain synthetic responses, designs, candidate/selected-R fits, analytic optima,
namespace/key and audit hashes:

| Case and replicate | Candidate REML criterion | Analytic optimum | Gap |
|:---|---:|---:|---:|
| n10 near correlation, G8/m5/v0.25/r0.95/null, rep0 | 112.65301256008023 | 112.59237819950228 | 0.06063436058 |
| n20 zero slope, G8/m5/null, rep5 | 126.11097215994276 | 126.10562920489640 | 0.00534295505 |

Both candidate covariance estimates have numerical rank zero; their constrained
sample optima have rank one and agree with selected R criteria to rounding.
A zero-slope generating distribution does not require a zero sample estimate.
The likely mechanism is a whole-column recovery guard in `src/optimizer.rs`:
off-diagonals around `2.3e-7`/`2.5e-7` exceed `sqrt(epsilon)` and skip the joint
zero-column escape check. This is a source-review hypothesis, not a completed
regression or repair. Follow the [next-step plan](../docs/LMM_RANDOM_SLOPE_OPTIMIZER_NEXT_STEP.md).

Independent unchanged R verification checked raw eligibility, reference
selection, numerical records, p/CI consistency, binary counts in all 564 rows and
all 72 primary bounds in **each** run. A separate AI audit regenerated all 2,820
responses, checked all 14,100 starts and 14,100 outcomes against journals/audits,
independently recomputed the balanced spectral optima and reproduced all
secondary bias/RMSE/SE/width moments. The R checker itself does not independently
validate those secondary moments. Maximum CP-bound errors were
`1.31e-18`/`1.12e-16`; maximum known-variance GLS errors `1.12e-15`/`1.34e-15`.
The exact original summaries/checks and freezes are retained:
[n10 summary](RANDOM_SLOPE_SCALING_V1_N10.json),
[n20 summary](RANDOM_SLOPE_SCALING_V1_N20.json),
[n10 R check](RANDOM_SLOPE_SCALING_V1_N10_R_CHECK.json),
[n20 R check](RANDOM_SLOPE_SCALING_V1_N20_R_CHECK.json),
[n10 freeze](RANDOM_SLOPE_SCALING_V1_N10_FREEZE.json),
[n20 freeze](RANDOM_SLOPE_SCALING_V1_N20_FREEZE.json).

## Resource measurements

All phases were sequential, without concurrent tests/builds/rehearsals. Each
phase used the prospective **900 s / 4 GiB committed memory / 512 MiB sampled
study disk / 1 GiB free-volume reserve** limits. All workers and descendants
completed, no limit event/abort occurred and measurements remained within bounds.
The parent, other applications and archive creation are outside these job limits.

| Measurement | 10 replicates | 20 replicates |
|:---|---:|---:|
| Candidate phase, cleanup included | 28.178 s | 51.388 s |
| Four-variant R phase, cleanup included | 127.516 s | 281.401 s |
| Separately capped Python analysis | 6.358 s | 11.557 s |
| Python analysis loop alone | 5.017 s | 9.822 s |
| Separately capped independent R verification | 2.684 s | 5.392 s |
| Candidate maximum sampled job commit | 445.05 MiB | 413.10 MiB |
| Reference maximum sampled job commit | 1,857.49 MiB | 1,860.24 MiB |
| Reference raw OS peak job charge | 1,861.83 MiB | 1,860.50 MiB |
| Analysis maximum sampled job commit | 75.08 MiB | 74.33 MiB |
| R verification maximum sampled job commit | 129.11 MiB | 149.94 MiB |
| Largest actual sampling gap, all phases | 0.1793 s | 0.2130 s |
| Maximum sampled study disk, all phases | 14.322 MiB | 22.915 MiB |
| Physical journals | 9,080,832 bytes | 14,745,600 bytes |
| Compressed input payload | 2,109,585 bytes | 4,200,314 bytes |
| Compressed outcome payload | 2,327,418 bytes | 4,649,973 bytes |
| Other journal pages/indexes/metadata/space | 4,643,829 bytes | 5,895,313 bytes |
| Analysis files, including raw audit and R check | 3,515,694 bytes | 6,861,615 bytes |

The [resource receipt](RANDOM_SLOPE_SCALING_V1_RESOURCES.json) includes original
job/worker receipts, per-native-process resident high-water values and each
worker's generation, start/outcome persistence, callback and recovery timers.
Summed worker callback times were 81.466/157.359 s for candidates and
472.458/1,058.702 s for R; these overlapping wall timers include callback/pipeline
work and cannot be added to phase wall time as CPU attribution. Start/outcome
commit sums were 7.231/13.901 s for candidates and 20.573/39.555 s for R.

Committed memory, the raw OS peak charge, and per-process resident high-water
are distinct metrics. On this host a denied allocation can appear in the raw
peak charge, so an above-limit raw charge conservatively rejects a run rather
than being relabeled successful allocated memory. Time/disk are sampled, with
0.1 s sleeps plus handling cost; they are not kernel filesystem quotas. The
active-phase cap clock excludes cleanup; the end-to-end table includes it. The
minimum sampled free volume across all runs was over 57.23 GiB.

The descriptive secant per additional dataset was **0.02469 s candidate**,
**0.16371 s four-variant R**, **0.00511 s analysis loop**, **6,026.35 journal
bytes** and **3,559.49 analysis bytes**. The fitted R intercept is **negative
26.37 s**, retained as evidence of noisy/nonlinear behavior. Physical storage
minus payload is not pure fixed overhead. These two independent input banks do
not identify a reliable production cost model. Both rehearse R for every dataset
and use uniform case budgets; the proposed full campaign has different case
weights and an R subset. Do not replace the earlier provisional 23.46-hour/
8.42-GB allowance with a claimed measured bound. Analysis, archive duplication,
environment and scratch provisioning still need a production plan.

## Controls, reproducibility and next action

The [control receipt](RANDOM_SLOPE_SCALING_V1_CONTROLS.json) preserves successful
time/disk aborts and nested-child allocation denial under a 128 MiB hard limit.
The actual native-loaded interruption control preserves completed rep0 and both
earlier durable start timestamps, retains rep1 as interrupted, fits only rep2,
and verifies marker PID membership in the whole owned job plus input hashes.
Its three starts/outcomes include two completed fits and one interruption, zero
R attempts, and are excluded from scaling data.

The first frozen bank A's native control was **rejected**, because its audit
mistakenly required equality between native interpreter and Windows venv launcher
PIDs. Its input hashes matched and interruption worked; it completed one fit and
durably started one interrupted fit, but was never resumed. Bank A n10/n20 generated
no inputs. Preserve its original sources, negative controls and failed receipt.
Before replacement bank B, amend the protocol/source, use fresh namespaces and
verify owned-job membership. This correction is disclosed, not a replacement of
a scientific outcome or retrospective acceptance of bank A.

The 134-test script suite passed with R available; all 16 affected cap/scaling
tests passed again after the process-inventory correction. Script lint, required
R/Julia formatting and documentation checks passed. No Rust/binding sources or
dependencies changed; no new full engine/bindings suite, full CI, confirmation,
release or public raw-data upload was run. The separate
[AI review](../docs/LMM_RANDOM_SLOPE_SCALING_REVIEW.md) is independent internal
scrutiny, not human peer review.

The [local archive descriptor](RANDOM_SLOPE_SCALING_V1_BUNDLE.json) records a
**28,801,948-byte, 1,508-entry** SHA/CRC-verified archive containing both banks,
their source snapshots, journals, logs, audits and checks. Installed environments
are excluded; the previously hashed candidate wheel is separate. The archive is
local and has not been uploaded. Public summary/check/freeze/acceptance/control
copies are byte-identical to originals. The
[bank index](RANDOM_SLOPE_SCALING_V1_BANK.json) binds the three bank B manifests;
[n10 acceptance](RANDOM_SLOPE_SCALING_V1_N10_ACCEPTANCE.json) and
[n20 acceptance](RANDOM_SLOPE_SCALING_V1_N20_ACCEPTANCE.json) bind capped analysis/R
commands, executable/script/input/output hashes and resource acceptance.

Next: reproduce and repair both optimizer counterexamples with regression-first
tests, then independently revalidate a new binary in a fresh frozen pilot. Keep
all original failures unchanged. Finish production orchestration/storage/analysis
provisioning and final protocol/source review before seeking explicit campaign
authorization. Full confirmation remains disabled.
