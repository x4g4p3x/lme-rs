# Fresh stopping-control pilot

Created prospectively, 16 October 2026 (Europe/Berlin). Status: completed locally;
original analysis abort and three numerical misses retained, readiness blocked.
Starting revision `b4cbf075bcaaa985d212c792772871fca480da1f` was pushed with
hooks/preflight, local/tracking/live remote equal and clean checkout. The user
authorized this next milestone after diagnostic delivery. Follow the existing
[handoff plan](../LMM_RANDOM_SLOPE_STOPPING_CONTROL_NEXT_STEP.md).

## Scope and acceptance

Implement a separate source-frozen protocol/runner/report/tests and conduct the
fresh exploratory numerical pilot: 94 cases times 20 replicates, all four R
variants, absolute tolerance 1e-10, maximum 5,000 iterations, default starts.
Reuse the exact repaired engine only after checking original pre-build receipt,
64 engine sources, installed-native/wheel/environment identities. Freeze a new
namespace after independent source review, before any response generation.
Keep all numerical/inference/reference/readiness gates and resource caps. Retain
all failures and previous outcomes. No release, confirmation or production run.

## Intended implementation and validation

Use explicit separate execution modules and immutable legacy helpers; do not
mutate historical globals or execute an old candidate helper at the wrong control.
Keep the statistical reporter identical except its runner import/docstring.
Register this record before editing. Tests must exercise intentional same-engine
reuse and reject mismatched identities, changed controls/sources/inputs, old seed
namespaces and output overlap. Run focused tests, script lint/full script suite,
and documentation checks. Engine/bindings sources remain unchanged; reuse their
passing validation. Run candidate, R, analysis and independent R checks as separate
capped sequential jobs without concurrent tests/builds. Require independent AI
source and result scrutiny; distinguish that from human peer review.

## Findings and decisions

The tolerance is fixed prospectively from the delivered diagnostic. Do not alter
it or any gate based on this run. A failed pilot stays failed and blocks readiness.
If it passes, publish numerical readiness independently of the 72 calibration
rules, which cannot be confirmed at this small n. Production preparation remains
a subsequent engineering milestone with a separate campaign-authorization gate.

## Observed validation and next action

Separate runner/report/completer implemented without changing production engine
sources. Nine focused integrity/control tests pass. Script lint and the full
documentation tier pass, including all 153 script tests with R available, example
compilation, doctests and documentation generation. Earlier full-script validation
passed 151 tests before the last two integrity tests were added; final tier covers
all nine new tests. A mistyped runner subcommand was corrected to `ruff-scripts`.

Independent read-only review requested binding the published repair anchor and
checking its predecessor hash; both are now frozen/verified. Namespace total
length is limited to the generator's 64 characters. Injected method exceptions
retain the legacy failed-eligibility behavior, and corrupted saved inputs fail
before any callback. Reviewer rechecked the corrections and found no remaining
pre-freeze blocker. Reporter differences are only docstring/import ordering and
runner import; statistical calculations and gates are identical.

Frozen `stopping-control-pilot-v1-20261016-a` after review and all tests/builds
finished: 122 sources, 64 unchanged engine inputs, and 19 prior manifest identities
(11 distinct namespaces). The original native SHA remains
`34edf7fa6bdf11a5fc0581c14d4ca11a18c017781a1e97eb1e3b0469f44f59cb`, wheel SHA
`b8e3b970c6b9a15d644abb50b87bf2d5d6f6a5714abe803c2faffb1a4d827fa5`.
The exact pre-build receipt still identifies its original build; study revision
is actual base `b4cbf07` plus frozen working-tree execution sources. No rebuild or
production-source change. A separately recorded source-review receipt binds the
frozen manifest before input generation.

Candidate phase completed with 1,880 starts/outcomes, all converged and no fit or
method errors; cleanup-inclusive 55.385 seconds, all four workers exit zero and
no resource abort. Reference phase started only afterward. Accuracy/readiness
decisions wait for completed paired analyses and independent checks. No tests/
builds overlap measured phases. No new commit for this milestone.

## Retained failure and separate diagnostic analysis

All 7,520 reference outcomes completed. Original capped analysis then aborted on
`resource_measurement_unavailable:FileNotFoundError` after 2.371 seconds. Its
partial files and abort receipts remain untouched. A partial candidate flag also
shows a genuine incorrect-convergence case, so numerical readiness is blocked
independently of the resource failure. No fit will be rerun or replaced.

Before a separately labelled retained-fit analysis, reproduce the old sampler's
is-file/stat deletion race with a failing test and implement a separate hardened
controller; the frozen original remains unchanged. Independent review also
identified pathlib's suppressed directory-permission errors. Explicit scandir,
single-stat child sampling, narrow vanished-child handling and root checks fix
both issues. The same two regressions fail against the immutable old controller.
Five sampler tests pass, including denied subtree/root/permission controls and
actual SQLite WAL churn under an owned job. Three driver tests verify fit/partial/
abort preservation, nonempty WAL/source rejection and original-abort identity.
All 161 script tests pass with R available; final script lint passes after import
formatting. Recheck the affected tests after formatting before diagnostic freeze.

The [separate diagnostic protocol](../LMM_RANDOM_SLOPE_STOPPING_DIAGNOSTIC_REANALYSIS.md)
keeps all gates/caps and executes the identical old reporter/R checker in a new
destination. Freeze original 122 plus five new source identities, every original
journal, partial output and abort receipt. Verify them before/after both phases.
Independent reviewer rechecked the repaired enumeration and driver; no remaining
pre-freeze blocker. Original engineering acceptance/readiness stay false even if
this diagnostic execution completes. No new fitting or source changes to v1.

## Completed diagnostic results

The separately reviewed 127-source diagnostic freeze completed exact old
reporter/R-check execution. All original journals, partial files and abort
receipts remain byte-identical before/after both phases. Assessment explicitly
keeps original engineering acceptance/readiness/confirmation false. Only the
diagnostic execution is accepted. All jobs and tests have finished.

Three of 1,840 balanced candidates fail unchanged gates: one incorrect rank-zero
optimum (criterion gap 0.00154515777), two covariance-01 precision misses despite
small criterion gaps. All 1,880 candidates reported convergence, no fit/method
exceptions; 1,837 balanced candidates pass. Forty imbalanced inputs lack a
balanced oracle. Selected R has no failure; 1,877 eligible pairs per method
agree numerically and in reject/cover decisions. Retain 37 unselected R numerical
misses and one nonconvergence. Two primary datasets fail, producing four primary
method failures; none of 72 narrow calibration rules passes at this small n.
This is no nominal-rate conclusion or general correctness certificate.

Independent R verifies all 564 count rows and 72 bounds (max CP difference
1.11e-16, known-GLS 1.33e-15). Full source/result/publication review and raw bundle
publication preparation are tracked below. The [report](../../scientific/STOPPING_CONTROL_PILOT_V1.md),
[assessment](../../scientific/STOPPING_CONTROL_PILOT_V1_ASSESSMENT.json) and
[counterexamples](../../scientific/STOPPING_CONTROL_PILOT_V1_COUNTEREXAMPLES.json)
preserve exact observations. No new core/bindings/performance/hosted/release or
confirmatory run occurred. Reused engine validation remains applicable; earlier
five optional all-target Clippy test-code findings remain unchanged.

Next follow the [boundary/covariance repair plan](../LMM_RANDOM_SLOPE_PRECISION_RECOVERY_NEXT_STEP.md).
Reproduce the three exact failures, investigate skipped objective-equivalent
columns separately from interior covariance precision, and preserve gates,
budgets and backend controls. A new repaired engine needs separately frozen
fresh validation. Reviewed for the user's authorized commit/push: implementation,
frozen sources, original failures and diagnostic evidence remain unchanged since
validation and independent publication review. Deliver with normal hooks, then
verify local/tracking/live remote identity and remaining checkout changes.

Independent result audit regenerated all 1,880 inputs, matched all 9,400 starts
and outcomes, independently enumerated 1,840 balanced optima (1,294 interior,
545 one-binding, one two-binding) and recomputed all 564 count/moment rows and
72 bounds. Maximum oracle-criterion discrepancy is 4.55e-13. All 122 original/
127 diagnostic sources, 100 original input/partial/abort bindings, exact reused
engine/wheel/native/build identities, all eight Python/four R PID/environment
receipts and analysis command/resource bindings agree. No fits, builds or edits
were performed by the reviewer. Publication review follows the raw archive.

Final publication review verified all 502 archive entries against retained bytes,
unique inventory/CRC/SHA, eight exact public original-file copies, all three full
counterexamples and resource/worker receipts. No remaining publication blocker.
The local bundle is 41,187,682 bytes, SHA
`e3592d83d1f27101ab4b64fda91e69e4c8500872341ae8c8e612fa59f56e3f2e`;
it is not uploaded and excludes installed environments. Console-log local paths
are sanitized with original/sanitized hashes recorded; scientific payloads and
sources are unchanged. Original 127 source and 100 input/partial/abort bindings,
public hashes/JSON/privacy and whitespace checks pass. Final Markdown links cover
102 documents and 1,266 local targets. Required documentation components reuse
the passing pre-input tier and post-change full script/link validation;
no unnecessary Rust rebuild or unrun CI is claimed.
