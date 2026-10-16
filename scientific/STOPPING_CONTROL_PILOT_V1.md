# Fresh stopping-control pilot v1 and retained-fit diagnostic analysis

16 October 2026. **Numerical readiness is false:** three candidates report
convergence but fail the unchanged independent numerical gates. The original
analysis also aborted on a disk-sampling race. Separate diagnostic analysis
completed successfully; it does not grant retrospective engineering acceptance.
No nominal-calibration, general-correctness, release or confirmation claim.

## Prospective design and identities

The [protocol](../docs/LMM_RANDOM_SLOPE_STOPPING_CONTROL_PROTOCOL.md) and independent
source review preceded generation of 1,880 fresh inputs, 20 per unchanged case,
under `stopping-control-pilot-v1-20261016-a`. Freeze 122 sources, all 64 unchanged
engine inputs, exact original pre-build receipt and 19 prior manifest identities
(11 distinct namespaces). Fix REML, default starts, absolute cost-SD tolerance
`1e-10`, maximum 5,000 iterations and the unchanged four-variant R bank.

The candidate deliberately reuses the repaired native, SHA
`34edf7fa6bdf11a5fc0581c14d4ca11a18c017781a1e97eb1e3b0469f44f59cb`, and wheel SHA
`b8e3b970c6b9a15d644abb50b87bf2d5d6f6a5714abe803c2faffb1a4d827fa5`.
Python 3.11.15/Polars 2.0.0/platform and original engine sources match exactly.
The original [build receipt](RANDOM_SLOPE_REPAIR_PILOT_V1_BUILD_INPUTS.json) remains
separate from the new study's actual Git base `b4cbf07` plus frozen working-tree
execution sources. No new binary, production math or convergence contract.
See the [original freeze](STOPPING_CONTROL_PILOT_V1_FREEZE.json).

## Completed fits and numerical failures

| Inventory or check | Observed |
|:---|---:|
| Candidate durable starts/outcomes | 1,880 |
| Candidate reported convergence | 1,880 |
| Candidate fit or method exceptions | 0 |
| R durable starts/outcomes | 7,520 |
| Candidate balanced numerical misses | **3** |
| Selected-reference numerical/method failures | 0 |
| Eligible pairs per method | 1,877 |
| Eligible paired numerical/rejection/coverage disagreements | 0 |
| Unselected R balanced numerical misses | 37 |
| Unselected R nonconvergence | 1 |

Of 1,840 balanced candidates, 1,837 pass the analytic controls; the other 40
imbalanced datasets have no balanced oracle and pass the declared comparison/
inference checks. The original raw methods remain retained even when numerical
failure makes them ineligible. All R variants remain present; objective-only
selection gives a valid reference for every input. Unselected failures are
prospectively nongating; they never erase candidate failures.

The [three immutable counterexamples](STOPPING_CONTROL_PILOT_V1_COUNTEREXAMPLES.json)
preserve exact inputs, original candidates, numerical checks and all R variants:

- `interior-G8-m5-v0.05-r0-beta0`, replicate 3: effectively rank-zero fit versus
  rank-one optimum; criterion `118.3728812168659` versus `118.37133605909438`, gap
  **`0.00154515777153`**, despite convergence after 53 iterations. The first
  Cholesky diagonal is about `3.05e-6`, above the outer recovery guard; a skipped
  zero-column check is a mechanism to investigate, not a completed new repair.
- `interior-G8-m15-v0.25-r0.5-beta0`, replicate 15: covariance `01` difference
  **`2.7078722e-6`** exceeds `1.7733984e-6`, despite criterion gap `1.67e-10`.
- `interior-G50-m15-v0.25-r0-beta0.5`, replicate 13: covariance `01` difference
  **`-1.5900777e-6`** exceeds `1.5339017e-6`, despite criterion gap `6.57e-10`.

The first two failures are primary-null datasets, giving four failed primary
method outcomes. The 72 narrow calibration rules do not pass at n=20 and do not
establish nominal rates. The successful reused-input sensitivity was control-
selection evidence; this separate fresh pilot shows why it was not a correctness
guarantee. No tolerance, gate, reference selection or outcome was altered.

## Original abort and separate diagnostic repair

The original analysis aborted after 2.371 seconds with
`resource_measurement_unavailable:FileNotFoundError`; its partial files and
[abort receipt](STOPPING_CONTROL_PILOT_V1_ORIGINAL_ABORT.json) remain unchanged.
The old sampler checks is-file and then stats again: SQLite can delete WAL/SHM
between those operations. A deterministic regression reproduces the failure.
Independent review also identified that Python's pathlib traversal can suppress
directory-permission errors; a denied-subtree regression reproduces that omission.

A separate controller module repairs both without modifying the original frozen
helper: explicit scandir traversal, one stat per enumerated child, narrow handling
of vanished children, root checks before/after, and propagation of permission/
I/O errors. Five tests pass, including actual SQLite WAL churn under an owned
job. Directory size is still sampled, not an atomic snapshot or whole-host quota.

The [diagnostic protocol](../docs/LMM_RANDOM_SLOPE_STOPPING_DIAGNOSTIC_REANALYSIS.md)
was reviewed and frozen before reanalysis, capturing 127 sources and original
journal/partial-output/abort identities. It runs the exact old statistical
reporter and R checker into a separate destination, under the same caps/gates.
All 94 journal bytes and original failed-analysis evidence match before/after
both phases. No fits were rerun or replaced. The
[assessment](STOPPING_CONTROL_PILOT_V1_ASSESSMENT.json) forces original engineering
acceptance **false**, numerical readiness **false**, confirmation claim **false**;
diagnostic execution acceptance alone is **true**. The
[diagnostic freeze](STOPPING_CONTROL_PILOT_V1_DIAGNOSTIC_FREEZE.json) and
[full summary](STOPPING_CONTROL_PILOT_V1.json) retain that separation.

## Independent checks, resources and validation

The [independent R check](STOPPING_CONTROL_PILOT_V1_R_CHECK.json) verifies raw
eligibility/reference selection, inference/p/CI, all 564 count rows and 72 exact
bounds. Maximum CP discrepancy is `1.11e-16`, known-GLS discrepancy `1.33e-15`.
It does not itself validate every secondary moment. Separate AI source/result
review is internal scrutiny, not human peer review or statistical certification.
That review regenerated all 1,880 inputs, matched every one of 9,400 starts and
9,400 outcomes, independently enumerated all 1,840 balanced optima and recomputed
all 564 count/moment rows and 72 bounds. Stored-oracle criterion differences
were at most `4.55e-13`. It confirmed the three failures, all source/binary/input
identities, reference selection and command/resource receipts.

Candidate/reference/analysis phases ran sequentially without overlapping builds/
tests. Original analysis failure is retained; the later diagnostic jobs have a
separate freeze and receipts. All jobs use 900-second/4-GiB aggregate job-commit/
512-MiB sampled-study-disk/1-GiB-free-reserve caps. Parent/environment/archive and
tests are outside these measured jobs; no whole-milestone resource cap is claimed.

| Phase | Wall time | Maximum sampled job commit | Outcome |
|:---|---:|---:|:---|
| Candidate | 55.385 s | 445.52 MiB | Completed |
| Four-variant R | 253.108 s | 1,854.77 MiB | Completed |
| Original analysis | 2.371 s | 72.28 MiB | Measurement abort retained |
| Separate diagnostic analysis | 10.643 s | 76.87 MiB | Completed |
| Separate diagnostic R check | 4.778 s | 152.35 MiB | Completed |

Maximum sampled disk is 27.824 MiB and largest sample gap 0.2071 seconds. Successful
jobs end with zero owned descendants and no observed limit events. The aborted
job's last snapshot has two processes before forced cleanup; it is not a clean
completion receipt. See [all resource receipts](STOPPING_CONTROL_PILOT_V1_RESOURCES.json).

Script lint passes. The final full suite passes 161 script tests with R available;
all eight sampler/diagnostic-identity tests pass again after final formatting.
The pre-input documentation tier passed, including all 153 then-existing tests,
examples, six doctests and documentation generation. Final links and publication
checks are recorded in the [work record](../docs/work/2026-10-16-stopping-control-pilot.md).
No production/bindings sources changed; prior engine checks remain applicable.
No new Rust/performance/bindings suite, hosted CI, release or confirmation ran.

## Preservation and next action

The [raw bundle descriptor](STOPPING_CONTROL_PILOT_V1_BUNDLE.json) records local
SHA/CRC-verified evidence, including original source/build freezes, fits/starts,
partial failed analysis/abort, separately frozen diagnostic analysis and exact
reused wheel/environment metadata. It is **not uploaded**; installed environments
are excluded. Original public copies match retained bytes.

Follow the [boundary/covariance repair plan](../docs/LMM_RANDOM_SLOPE_PRECISION_RECOVERY_NEXT_STEP.md).
Reproduce these failures before editing the engine; investigate boundary
classification and covariance precision separately, retain negative controls,
budgets and both backends. A newly frozen fresh pilot after repair must meet its
own gates before production preparation. The earlier failed studies stay failed.
