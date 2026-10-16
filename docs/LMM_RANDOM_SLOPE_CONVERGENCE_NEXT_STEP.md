# Next milestone: diagnose the strict-control iteration limit

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
