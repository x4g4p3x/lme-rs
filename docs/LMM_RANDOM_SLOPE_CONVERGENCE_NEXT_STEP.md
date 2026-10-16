# Next milestone: diagnose the strict-control iteration limit

## Execution record

Started prospectively on 16 October 2026 after verified delivery of `c87b8e0`
(local, tracking and live remote equal; clean checkout). Status: completed locally;
see the dated results below.
This existing plan owns the milestone; no duplicate work record is needed.

Before production changes, reproduce the fixture with the public fit API, inspect
the dependency's actual stopping arithmetic, and retain a diagnostic-only search
trace. Compare repeated cached evaluations and an independently assembled dense
or balanced likelihood at the retained theta and feasible neighboring points.
Use separate output files; preserve the completed pilot and gates. First run
focused diagnostics with one numerical thread. Choose Rust/feature/performance
validation only if a confirmed production defect warrants a repair; otherwise
publish the diagnostic evidence and prospective policy with script/docs checks.
No convergence-semantic change or fresh pilot is implied by a good objective.

After exact replay and before the broader sensitivity run, declare one additional
post-study diagnostic: refit all 1,880 retained inputs at absolute cost-SD tolerance
`1e-10`, with the same installed engine, default starts and maximum 5,000
iterations. Use four workers, one numerical thread and the existing resource caps.
Check all 1,840 balanced likelihoods against the unchanged independent oracle;
retain every start, exception and nonconvergence separately. This tests whether a
prospective control is plausible over the declared design registry, not whether
calibration holds. It does not supply new seeds, inference/reference parity or a
passing replacement pilot. Freeze the diagnostic script/control/input/native
identity before launching; do not tune further based on results in this milestone.

Created after the repaired-engine pilot, 16 October 2026. The near-zero covariance
repair fixes its two regression cases. The fresh 1,880-data pilot has no balanced
numerical miss among converged candidates, but one honest iteration-limit failure
keeps its prospectively declared readiness gate false. Preserve that result and
all earlier failures. Do not change the completed protocol, mark its fit converged,
rerun/replace its outcome, or quietly loosen tolerance to get a passing pilot.

## Establish the behavior and contract

Start with the [diagnostic fixture](../scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_BUDGET_DIAGNOSTIC.json),
the [repair record](work/2026-10-16-near-zero-column-recovery.md) and its report.
The case is `interior-G50-m15-v0.25-r0-beta0.5`, replicate 15 in namespace
`repair-pilot-v1-20261016-a`. Its returned REML criterion is
`2361.163568871662`, equal to the independent analytic optimum to displayed
precision; covariance/residual variance also closely agree. It returns
`converged=false`, `IterationLimit`, 5,000 iterations and no eligible inference.
That honest flag is not evidence of an incorrect numerical optimum.

Before production edits, inspect the public FitControl convergence contract,
Argmin sample-cost-standard-deviation termination in `src/optimizer.rs`, and the
matching Basin adapter. Reproduce the preserved behavior through a focused public
API diagnostic. Distinguish a stagnating search, a defect in state/convergence
accounting, and an absolute objective tolerance below attainable arithmetic
precision. A criterion around 2,361 with tolerance 1e-12 motivates this question;
it does not prove the mechanism.
The retained diagnostic objective is `2361.163568871659`, about `3.18e-12`
different from the final re-evaluated criterion (roughly seven ULPs); the
requested tolerance is roughly 2.2 ULPs here. Independent review treats this
as a clue about stopping/evaluation arithmetic, not causal proof.

First evaluate the exact retained theta and nearby feasible points repeatedly,
comparing the cached objective with an independently assembled dense calculation
or balanced constrained likelihood. Record objective variability, parameter/
covariance spread, improvement and iteration counts. Separately labeled
diagnostic refits may vary controls or starts to investigate the mechanism, but
must never replace the frozen pilot's failed record or count as fresh evidence.
Use actual new seeds only after a new protocol is fixed.

## Decide without weakening the evidence

If an implementation defect is confirmed, add a failing regression, fix its cause
and exercise hard iteration budgets, false convergence, true boundaries, units
and both optimizer backends. Do not substitute a data-specific criterion/tolerance
exception or silently change the meaning of an explicitly requested absolute
FitControl tolerance. Any public convergence-semantic change needs a documented
contract and independent checks, including nonstationary negative controls.

If the optimizer is honestly exhausting an unrealistic control, propose a
principled prospective control/readiness policy based on numerical resolution
and objective/stationarity/parameter evidence across the full declared domain.
Alternatively justify how genuine nonconvergence is handled by the pre-existing
failure-envelope design. Separate fatal incorrect-convergence/numerical misses
from honest failures without relabeling this already failed study. Review that
proposal independently before freezing a new binary/control/protocol/namespace.
No full campaign is authorized by this diagnostic work.

Run required Rust/script/docs/feature/performance checks for actual changes;
maintain a prospective living record and Unreleased changelog as applicable.
Retain source/native/wheel identities and original diagnostics. A new reviewed
fresh pilot must meet its own predeclared gates before proceeding to the
[production preparation plan](LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md). Production
orchestration, analysis/storage provision, final freeze and explicit campaign
authorization remain separate gates. AI scrutiny is not human peer review.

## Completed diagnostic and decision, 16 October 2026

Status: completed locally; no production engine or convergence-contract change.
Exact public API replay reproduces the original failure. A diagnostic Argmin
delegate reports cost SD `1.818989403546e-12` from iteration 200 through 5,000;
final evaluations cycle among adjacent floating-point points. Repeating the exact
retained theta is bit-stable, including after neighboring probes. Two production
evaluation paths differ by about seven ULPs. Independent 60/80-digit balanced
arithmetic puts the true optimum gap at `7.406685840455e-13`; separate AI review's
110-digit uncentered Gram calculation confirms it. No accounting defect was
demonstrated. Do not mislabel this as random objective noise or impossible
convergence for every path: retained-start refits and Basin can converge.

The declared uniform `1e-10` reused-input sensitivity has 1,880 exact durable
starts/outcomes, all converged, maximum 295 iterations, and all 1,840 balanced
numerical checks passing. Maximum objective gap is about `1.30e-9`; weakest gate
fraction is 0.8091. Forty imbalanced inputs lack a balanced oracle. No inference/
R/nominal calibration claim follows. The original failed pilot remains unchanged.

Review strengthened helper bindings and published-journal/WAL identity guards
after an exploratory first sensitivity. Preserve that run and rerun the identical
control in a separate final `b` directory. Final evidence binds 32 helpers, 64
engine sources, exact diagnostic script/example/test sources, input bits, original
94-journal inventory and distinct default/Basin executables. The Python
sensitivity uses the previously identified native. Independent AI source/result
audit found no remaining numerical, inventory, identity or resource blocker.

Six new diagnostic tests and all 144 script tests pass with R available. Required
lint, 145 Rust unit tests, five affected slope contract tests and focused example
Clippy pass. Both release diagnostic examples ran. An extra all-target Clippy
check fails on five unchanged test-code lints (one `items_after_test_module` in
`src/mcp.rs`, four `single_range_in_vec_init` in optimizer tests). This broader
failure is retained separately; ordinary required lint and new-example Clippy
pass. No production fix means no new throughput claim; prior unchanged-engine
golden/Basin/performance validation is retained. No full CI, bindings suite, heavy
production cases, fresh statistical pilot, release or confirmation ran here.

`docs-check` also passes: 97 Markdown documents/1,237 local targets, all 144
script tests with no skips, benchmark-site freshness, all-example compilation,
six doctests and documentation generation. Final whitespace, engine identity,
public-copy hashes, JSON validity and privacy checks pass. The local 53,764,960-byte,
240-entry raw bundle has verified CRC/SHA and every entry matches retained local
bytes; it is not uploaded.
Final independent publication review also verified all 240 entries, four exact
public copies, source/binary/resource bindings and report/next-plan wording;
no remaining publication or methodological blocker was found.

See the [report](../scientific/RANDOM_SLOPE_STOPPING_DIAGNOSTIC.md) and its exact
source/resource/raw-bundle receipts for evidence and limitations. Reviewed for
the user's authorized commit/push: execution sources, raw receipts and numerical
results are unchanged since passing validation and final independent review.
Deliver with hooks, verify local/tracking/live remote identity, then follow the
separately reviewed
[fresh-control pilot plan](LMM_RANDOM_SLOPE_STOPPING_CONTROL_NEXT_STEP.md), with a
new freeze/namespace and every numerical/inference/reference gate unchanged.
Production preparation remains conditional on that new pilot's actual results.
