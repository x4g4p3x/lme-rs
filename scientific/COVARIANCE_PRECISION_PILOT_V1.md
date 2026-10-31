# Covariance precision repair and fresh numerical pilot v1

16 October 2026. **The fresh pilot's engineering and numerical-readiness gates
pass.** All 1,880 candidates are eligible and agree with the selected independent
reference on the declared numerical/inference checks. This is exploratory,
bounded evidence: no nominal-calibration, general-correctness, release or full
confirmation claim. The 72 calibration rules do not pass at this small n.

## Repair and independent diagnostic

Three exact-bit public regressions failed on the preceding engine before edits.
The repair checks zero-column objective equivalence independently of a tiny
diagonal cutoff. When a centered precision stencil would cross a diagonal bound,
it also proposes feasible face directions; it never declares those points
equivalent merely because they are nearby. Interior coordinate interpolation
uses resolved centered curvature, stays inside the sampled interval and accepts
only evaluated objective improvements. Preserve improving sampled endpoints and
boundary differentiation samples. Coordinate-specific restart steps and shared
outer-iteration accounting preserve feasibility, the caller's absolute cost-SD
tolerance and honest nonconvergence on budget exhaustion.

These are local search heuristics, not joint stationarity or global-optimum
certificates. The rounding guard is practical resolution protection, not a
universal factorization-error bound. See the
[work record](../docs/work/2026-10-16-covariance-precision-recovery.md) and
[independent AI review](../docs/LMM_COVARIANCE_PRECISION_REVIEW.md).

All three retained inputs pass the original criterion, variance, covariance-entry
and eigenvalue-rank gates under seven response/predictor-unit, reversed-row and
alternative-start variants. The exact-bit derivative is hash-bound to the
[original failures](STOPPING_CONTROL_PILOT_V1_COUNTEREXAMPLES.json). It is
diagnostic replay, not fresh validation. The independent 60/80-digit balanced
likelihood agrees with cached/full Rust fixed-point costs to maximum errors
`5.31e-14`, `1.12e-12`, `3.14e-12`. Verified improvements are `0.00154515777`,
`1.6657e-10`, `6.3725e-10`. The first input's exact origin penalty is about
`1.0219e-10`, illustrating why objective equivalence alone is insufficient.
See [raw points](COVARIANCE_PRECISION_REPAIR_RAW_POINTS.json) and
[independent diagnostic](COVARIANCE_PRECISION_REPAIR_DIAGNOSTIC.json).

## Prospective design and exact identity

The separate [protocol](../docs/LMM_COVARIANCE_PRECISION_PILOT_PROTOCOL.md) and
independent source review preceded response generation. Freeze 136 sources and
all 64 pre-build engine inputs under fresh namespace
`covariance-precision-v1-20261016-a`. The unchanged 94-case design uses 20 inputs
per case, REML, default starts, absolute cost-SD tolerance `1e-10`, maximum 5,000
shared outer iterations, and all four fixed-order R variants on every input.
No retries, outcome replacement, adapted controls or reference-start adaptation.
Prior namespaces and manifest hashes are retained in the
[freeze](COVARIANCE_PRECISION_PILOT_V1_FREEZE.json).

New native SHA:
`77211eac27abdd1be2c27243b5a60d295e47a52f839664998a78222fe6ff6cbc`.
Wheel SHA:
`f37776f76d8d7cc3ee83cf68291408f67d422612411987bbfa5f4ed43b72f09c`.
Python 3.11.15, Polars/runtime 2.0.0, package 0.2.8.dev0; Windows/AMD64,
Rust 1.99.0. The [build receipt](COVARIANCE_PRECISION_PILOT_V1_BUILD_INPUTS.json)
identifies actual working-tree source on base `93dc06a`; it does not attribute
the repair to that commit. Source hashes match before/after building; installed
native bytes match the wheel. Historical archived sources and failed outcomes
remain unchanged. Explicit current-source evolution is limited to engine files
and the regression target, not historical execution or numerical helpers.

## Fresh outcomes

| Inventory or check | Observed |
|:---|---:|
| Candidate durable starts/outcomes and convergence | 1,880 |
| R durable starts/outcomes and convergence | 7,520 |
| Balanced candidate numerical misses | **0 / 1,840** |
| Candidate/selected-reference method failures | **0** |
| Eligible pairs per method | **1,880** |
| Paired numerical/rejection/coverage disagreements | **0** |
| Maximum candidate balanced criterion difference | `8.23e-11` |
| Maximum candidate outer iterations | 174 |
| Unselected R balanced numerical misses | 30 |
| Unselected R nonconvergence | 0 |

The other 40 imbalanced inputs have no balanced oracle; their declared comparison
and inference checks pass. All R variants remain retained. Objective-only
selection chooses baseline/tight-default/tight-positive/tight-negative variants
163/890/494/333 times. The 30 unselected numerical misses remain visible and are
prospectively nongating; selection never uses oracle agreement. All-attempt
denominators and failed-outcome handling are unchanged.

All 72 narrow simultaneous calibration rules fail at n=20. Zero numerical
failures does not certify nominal error/coverage rates. Secondary moments remain
descriptive, with no bootstrap intervals. The
[full analysis](COVARIANCE_PRECISION_PILOT_V1.json),
[independent R check](COVARIANCE_PRECISION_PILOT_V1_R_CHECK.json) and
[acceptance](COVARIANCE_PRECISION_PILOT_V1_ACCEPTANCE.json) preserve this separation.
R independently checks 564 binary-count rows, 72 bounds, selection, raw-fit
inference/p/CI and known-variance GLS; maximum bound/GLS discrepancies are
`1.11e-16` / `1.55e-15`.

## Resource and performance limits

Before freeze, separately retained negative controls enforce time, sampled disk
and hard Windows job memory. The memory control emits allocation denial with
exit 91 and a native limit event; OS peak charge alone is insufficient. Source
closure, original interpreter/program/disk-target command identity and raw
plan/start/end/log bytes are revalidated and hash-bound into the study.

Candidate/reference/analysis/R-check jobs ran sequentially without concurrent
builds/tests/benchmarks in 52.823/242.964/9.976/4.637 seconds. Maximum sampled
reference-job commit is 1,853.96 MiB; maximum sampled study disk is 25.950 MiB.
Limits are 900 seconds per phase, 4 GiB aggregate job-committed memory, 512 MiB
sampled study disk and 1 GiB free reserve. Sampling is 0.1 seconds plus scan
cost; actual gaps and OS charges/events are in the
[resource receipts](COVARIANCE_PRECISION_PILOT_V1_RESOURCES.json). Parent, build,
environment and archive preparation are outside measured jobs. No whole-host
quota or cross-platform hard-limit claim.

Correctness has a measured cost. Final serial local medians show roughly
38--45% overhead on small default slope fits and 13--20% on the 100k slope case;
Basin's corresponding ranges are 39--61% and 10--26%. Scalar/crossed cases stay
close to baseline. These Rust-only, baseline-first samples are not a randomized
order-balanced campaign or Julia comparison. Retain earlier prototypes and the
overlapping-verification measurements, excluded from the final comparison.
See [performance identities/results](COVARIANCE_PRECISION_REPAIR_PERFORMANCE.json).

## Validation, availability and next gate

Final Rust validation passes 575 tests, with four existing heavy cases ignored.
Affected optimizer tests pass in default/Basin configurations (22/35), all 31
contract tests pass in both, and release golden parity passes in both. All 172
script tests pass with R available; Rust/Python/script lint and documentation
checks pass. Independent AI result and publication review are recorded separately
and do not replace human peer review.

The raw evidence is retained locally; the
[archive descriptor](COVARIANCE_PRECISION_PILOT_V1_BUNDLE.json) states its exact
availability, contents and checksums. No upload or external archival availability
is implied. Preserve all earlier failed studies. The next gate is tested,
reviewable [production campaign preparation](../docs/LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md),
with explicit scheduling/storage/stop budgets and a separately frozen manifest.
The full confirmatory campaign remains disabled until explicitly authorized.
