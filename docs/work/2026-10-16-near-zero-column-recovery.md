# Near-zero covariance column recovery

Created prospectively on 16 October 2026 (Europe/Berlin). Status: implemented
locally and independently reviewed; fresh pilot retains one iteration-limit
failure, so numerical readiness remains blocked.
Baseline `10f7b7817acd2ec5f2f3fc535587603cc6d5e106` is committed and pushed with
hooks/preflight enabled; HEAD, tracking and live remote match, checkout clean.
The user authorized this next milestone after the scaling evidence delivery.
Follow the [bounded repair plan](../LMM_RANDOM_SLOPE_OPTIMIZER_NEXT_STEP.md).

## Scope and acceptance

Reproduce both archived rank-zero false-convergence cases through the public Rust
API before production edits. Preserve the original data, fits, scientific sources,
manifests and local archive. Identify the recovery mechanism, repair it without
loosening numerical gates or arbitrarily enlarging coordinate thresholds, and
retain true-boundary and shared-budget behavior. No release/full confirmation.

First test a candidate principle: when a diagonal is on its boundary, compare the
cost of zeroing its whole Cholesky column with the current objective, rather than
classifying a tiny off-diagonal by coordinate size alone. A numerically equivalent
origin can supply an even quadratic curvature probe; accept only an actual
feasible improvement against the original best objective. Confirm the mechanism
with failing regressions before choosing the implementation.

## Planned validation

Use existing model-contract and optimizer unit targets, with independent analytic
criteria/covariance/residual-variance expectations from the immutable fixtures.
Exercise response/predictor unit changes, genuinely nonzero columns, true boundary
optima and caller iteration budgets. Run focused failures/fixed tests, `task rust`
and `task lint:python` (covering the Rust/lint minimum), release golden parity and
applicable fair-harness cases. Read OPTIMIZATION.md/BENCHMARKS.md; record comparable
baseline/fixed complete fits and avoid a general performance claim.

After numerical repair, build a separately identified isolated wheel and use a new
source-frozen, reviewed fresh-seed pilot with independent balanced/GLS controls
and fixed-bank R checks. Earlier scientific sources remain immutable; historical
freezes are verified against their archived copies when the engine changes.
Maintain this record and Unreleased changelog; report actual checks and limits.

## Findings and next action

Both new public-API regressions failed before production edits, reproducing the
exact archived objective gaps. A synthetic quadratic with a residual `-2.5e-7`
off-diagonal also failed the old whole-column guard. The hypothesis is confirmed:
that residual exceeds coordinate resolution although its cost effect is invisible.

The repair compares the exactly zero column's objective with the original best.
If zeroing genuinely improves cost, retain that feasible improvement; if it is
equivalent at the existing objective tolerance, estimate the even quadratic at
that exact origin. Curvature uses the origin cost, but every returned probe must
improve the original best by the existing gate. No coordinate cutoff was enlarged.
Both public regressions now pass across five unit transformations and default/
archived starts (20 fits total); five covariance-recovery unit tests pass, including
a genuinely nonzero optimum. Existing shared-budget/true-boundary checks are part
of the broader passing `task rust` run. No historical raw fits were modified.

The fair-harness baseline for five affected/control cases was retained before
production edits. Release golden parity passed (two tests), as did the five fixed
default fair cases. Measured objective/coefficient/convergence/iteration checks
are identical before/after for cold and prepared fits. Cold medians are slightly
slower, including +6.25% in the unchanged scalar control; retain those timings
without a speed or general non-regression claim. This is one ordered pair of
sessions, three warmups/11 measurements/one thread. Because the shared helper
also serves Basin, its five slope contract tests, three boundary tests, nine
recovery unit tests and all five fair cases were exercised and passed. No
Basin-before timing or Julia speed comparison is claimed.

The full Rust tier passed, with four documented heavy production cases ignored.
Python Ruff and script lint passed. All 138 script tests passed with R on PATH;
four affected receipt/source tests passed after final source edits. A supplementary
documentation benchmark-test invocation skipped two optional R tests because its
PATH omitted R; they already passed in the complete invocation. Documentation
checks are satisfied by the passing Rust tier's all-target/examples/doctest/doc
components plus Markdown links, benchmark-tests and benchmark-site checks; the
last link check covered 95 documents and 1,225 local targets. No new complete
bindings suite, full CI/hosted matrix or heavy production campaign ran.

Before inputs, the independent AI source reviewer found no optimizer blocker and
requested two harness corrections: preserve exact pre-build receipt bytes and
verify SHA/parsed identity; state candidate/selected-reference readiness scope
explicitly while retaining unselected variant failures. Both were corrected and
tested before freeze. The statistical report is an explicit separate copy of
the immutable scaling reporter, with only docstring/runner import changed.

Built a new isolated CPython 3.11 wheel after recording all 64 engine inputs;
the source hashes remained identical afterward. It records the actual dirty
working-tree engine on base `10f7b78`, not an invented committed revision.
The initial `--no-deps` environment install omitted Polars' runtime binary; install
the exact matching `polars-runtime-32==2.0.0` before any study inputs, then verify
package compatibility and Python/Polars/platform against the previous candidate.
New native SHA is `34edf7fa6bdf11a5fc0581c14d4ca11a18c017781a1e97eb1e3b0469f44f59cb`;
wheel SHA is `b8e3b970c6b9a15d644abb50b87bf2d5d6f6a5714abe803c2faffb1a4d827fa5`.

Freeze all 114 source hashes and the exact build receipt, then run all 94 cases
at 20 fresh replicates under `repair-pilot-v1-20261016-a`, unchanged controls/
gates/caps and four-variant R bank. All four measured jobs completed sequentially
without concurrent tests/builds/rehearsals. Engineering acceptance passed; the
[original acceptance record](../../scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_ACCEPTANCE.json)
explicitly keeps numerical readiness false and confirmation claim false.

## Observed fresh pilot and remaining gate

The 1,880 candidate and 7,520 R starts/outcomes are complete. Among 1,840 balanced
inputs, no converged candidate failed the independent oracle. One candidate in
`interior-G50-m15-v0.25-r0-beta0.5`, replicate 15, honestly exhausted 5,000
iterations. Its objective matches the oracle to displayed precision and covariance
is accurate, but all methods remain ineligible. The 1,879 eligible pairs per
method have zero numerical or reject/cover disagreements. Selected R fits all
pass; 20 unselected R numerical misses and one unselected nonconvergence remain
retained. The candidate failure is outside the primary family; primary failure
counts are zero but all 72 calibration rules fail at this small n. No calibration
pass or outcome replacement is claimed.

Independent R checked all 564 binary-count rows and 72 bounds. Independent AI
audit regenerated 1,880 inputs, matched all 9,400 starts and 9,400 outcomes,
recomputed all 1,840 balanced active-set oracles and every count/moment/bound, and
verified source/build/wheel/native, variant selection, PID/resource and capped
command provenance. Max converged objective gap is `1.87e-11`. The budget case's
diagnostic objective and final criterion differ by `3.18e-12`, versus requested
`1e-12`; this is a precision clue, not proof of a new defect or justification for
loosening the frozen gate. Source review and final data review were read-only,
without fits/builds/edits. No additional accounting/provenance defect was found.

Measured candidate/R/analysis/R-check times are 55.068/282.450/10.157/5.086 seconds;
maximum sampled reference-job commit is 1,841.99 MiB, study disk 23.524 MiB, sample
gap 0.1672 seconds. Limits were 900 seconds/4 GiB committed/512 MiB sampled disk/
1 GiB free volume; parent, environment and archive preparation are outside them.
The [report](../../scientific/RANDOM_SLOPE_REPAIR_PILOT_V1.md),
[review](../LMM_RANDOM_SLOPE_REPAIR_REVIEW.md),
[resource receipt](../../scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_RESOURCES.json) and
[diagnostic fixture](../../scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_BUDGET_DIAGNOSTIC.json)
preserve the precise evidence and limits.

The local 39,888,649-byte/543-entry raw archive includes source/build freezes,
wheel/package metadata, journals/logs, analysis/checks, failed/fixed regression
evidence and raw performance reports. SHA/CRC verified; original public copies
match raw bytes. Independent publication review also verified all archive entries
against local evidence; public JSON and privacy checks passed. It is not uploaded;
installed environments are excluded. See
the [descriptor](../../scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_BUNDLE.json).
Old archived sources and outcomes remain unchanged. Reviewed for the user's
authorized commit/push: implementation and frozen execution inputs are unchanged
since validation; final Markdown links and whitespace checks pass. Deliver with
normal hooks, then verify local/tracking/live remote identity before beginning
the separate convergence diagnostic milestone.

## Next action

Follow the [convergence diagnostic plan](../LMM_RANDOM_SLOPE_CONVERGENCE_NEXT_STEP.md)
before changing stopping semantics or proposing a new prospective control/policy.
Preserve this honest failure and failed readiness gate. A separately reviewed,
newly frozen pilot must support any later eligibility claim. Only afterward
follow the conditional [production preparation plan](../LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md),
final freeze and explicit campaign authorization. No full confirmation or release.
