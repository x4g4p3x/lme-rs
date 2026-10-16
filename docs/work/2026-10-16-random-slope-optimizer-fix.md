# Random-slope optimizer boundary recovery

Created prospectively 16 October 2026 (Europe/Berlin). Status: **implemented and
validated for authorized delivery; fresh-pilot numerical controls remain to be settled**.
Baseline: `922e4600e8635c3a03cd7f3b74a9db1186b0dbe3`, pushed and verified against
tracking/live remote, clean tree. The user authorized proceeding with the numerical
fix after committing the [v1 pilot](../../scientific/RANDOM_SLOPE_PILOT_V1.md).
No release or full confirmatory campaign is authorized by this record.

## Scope and acceptance

Reproduce nonoptimal reported convergence through the public API using the retained
v1 counterexamples and independent constrained optimum. All 119 affected balanced
candidate fits had an exactly zero Cholesky diagonal. Investigate projected-simplex
collapse, preserve true boundary solutions, respect caller iteration/tolerance
controls and test default/custom paths and optional Basin separately. Preserve all
v1 sources, inputs, outcomes and decisions; fixes apply to a distinct candidate.

## Intended implementation and checks

Add a regression first in the existing model-contract target. Inspect the existing
Basin covariance recovery before choosing a minimal shared fix for Argmin LMMs;
do not substitute the balanced diagnostic oracle into production fitting.
Select Rust lint, all-target compilation, consolidated unit/integration/doctests,
API documentation and Python lint (the broad Rust slice in AGENTS.md), focused
Basin tests, release golden parity and applicable fair-harness workloads. Update
the changelog and run documentation checks. Serialize Cargo work and keep timing
runs separate from other heavy work. Bindings need their tier only if edited.

Recheck retained v1 records only as explicitly labelled post-study diagnostics,
never replacements. If proceeding to a new pilot after the fix, freeze a distinct
binary/protocol with disjoint seeds and independent review before observations.

## Findings, decisions and observed validation

The public regression reproduced false convergence with criterion 800.05309016
against the independent global 796.06355447. It failed under default Argmin and
passed after covariance-face recovery. Basin already passed that regression before
the fix. The regression checks default and explicit study controls, and default
and retained failed starting points. Only the explicit study controls impose the
pilot's covariance-parameter gate: default cost tolerance is coarser. The objective
gate is unchanged for both. No frozen study gate was loosened.

The default LMM path now reuses the covariance-column/orientation probes from
Basin, restarts only for an actually evaluated improvement and shares the overall
iteration budget. A regression discovers the initial collapsed search's iteration
count and gives recovery exactly one remaining iteration: the improved probe is
retained with nonconvergence. Generic GLMM/NLMM searches get no covariance recovery;
their Argmin convergence flag now explicitly requires SolverConverged termination.

Independent AI review identified a further helper limitation: an entirely zero
column can require a joint inward direction. A rank-three quadratic regression
failed before the extension. Both backends now estimate the local quadratic form
at two scales and verify feasible negative-curvature probes by actual cost. The
review also led to retrying the smaller scale after unusable large-scale samples;
the finite-domain regression failed before that hardening. These are local
safeguards, not global convergence guarantees, and add O(k²) evaluations only for
a wholly zero covariance column. The existing orientation policy is preserved;
Basin gains the additional whole-zero-column recovery too.

A separate real balanced LMM helper control has B=[[1,1],[1,4]], within RSS16,
face criterion55.44632023 and independent global55.12635891. Diagonal-only probes
raise its cost; a joint column direction improves it. The public face-start fit
already reaches the optimum in the old binary, so that public example is a passing
baseline control, not a reproduced public failure. Rational rotated cases with
zero, one and two active covariance bounds preserve the exact REML solutions.

Focused regression/boundary tests, the budget control and default/Basin
optimizer/contract checks pass. Three diagnostic integrity tests reject
overlapping/existing outputs, old/mismatched binaries and changed audit inputs.
The new runner saves separate candidate/source/wheel identities and checks all
original inputs/outcomes before and after refits. It cannot replace the v1 study.

## Completed validation and diagnostic findings

Rust lint, all-target compilation, 143 unit tests, 406 consolidated integration
tests, six doctests, examples and API docs passed. Four heavy production tests
remain intentionally ignored. Basin Clippy, 27 optimizer tests and 26 model
contracts passed. Release golden parity passed both tests in each backend.
The 77-test script suite passed its available tests; its two R-dependent controls
then passed separately with the installed R on PATH. Python/script lint and
`task docs:check` passed. Final Markdown-only additions use a fresh link check,
reusing unchanged passing doctests/examples/docs generation.

The fair harness completed sleepstudy REML, weighted sleepstudy REML and correlated
slopes with 100,000 rows, three warmups/11 repeats/phase reports. Argmin ran both
orders; Basin ran one order. Unweighted/large-slope fit gates passed. The known
weighted objective-normalization comparison remains unverified. No general speed
or equivalent prepared/full-fit claim is made. No new binding code, full binding
matrix, hosted platform matrix, package release or confirmation was performed.

The [diagnostic report](../../scientific/RANDOM_SLOPE_BOUNDARY_FIX_DIAGNOSTIC.md)
and [machine-readable evidence](../../scientific/RANDOM_SLOPE_BOUNDARY_FIX_DIAGNOSTIC.json)
own the counts/hashes. Separate refits of all 1,880 retained inputs converged with
all methods evaluable. All 1,840 balanced objectives meet the independent global
gate: 119 prior failures become zero; maximum gap is 2.047e-7. Original v1 hashes
remain unchanged. Independent AI review rederived feasible active-set optima
from raw inputs and confirmed the numbers. The diagnostic wheel/native/source
hashes identify this uncommitted-source candidate separately from v1.

The same 1e-8 controls still miss covariance-entry gates in 42 balanced fits
(39 off-diagonal, three slope-variance entries). A separately receipted, selected
precision experiment at 1e-12 with the same native binary and maximum 5,000
iterations makes all 42 evaluable with zero parameter/objective misses. The
reviewer verified these independently too. These are selected post-study records,
not pooled calibration estimates. Against retained R, 51 pairs remain flagged
(formerly 166), including seven R fits above the global optimum. One KR rejection
disagreement involves such a nonoptimal R fit. One balanced boundary classification
miss and one imbalanced Satterthwaite p discrepancy lie outside the selected 42;
the report identifies them. No claim that all numerical issues are resolved.

Delivery: the user-requested pilot commit 922e460 was pushed with hooks/preflight
and local/tracking/live remote SHA equality. The user subsequently authorized
committing and pushing this numerical fix and evidence, then proceeding with the
next diagnostic step. Review and reuse the unchanged passing checks above; retain
commit/push hooks and verify local/tracking/live SHA equality after delivery.

## Next action

Review and settle prospective tighter search controls without relaxing numerical
gates. Recheck boundary classification for zero_slope-G8-m5-v0-rundefined-beta0.5,
replicate 17, and inference/derivative precision for imbalanced-G8-mcycle3-5-10-v0.25-r0-beta0,
replicate 18. Use new separately named diagnostic outputs; preserve every completed
run. Then freeze an exact reviewed engine/protocol with disjoint seeds for a new
pilot. A full confirmatory family/budget remains a later decision. Preserve all
v1 and 0.2.7 outcomes; do not describe selected-input refits as confirmation.
