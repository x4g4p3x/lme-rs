# Living work records

These repository documents preserve the scope, reasoning, evidence and next action
for substantial tasks. Follow the [agent workflow](../../AGENTS.md#durable-work-records).
Read the relevant record before continuing work and verify historical facts against
the current checkout. A record is context, not proof that a claim remains current.

Create one record per meaningful task or milestone using the [template](TEMPLATE.md).
Create it before implementation, maintain it as decisions change, and review/update
it before each commit covering the task. Reuse it across commits. Small edits with
no lasting decisions do not need a new file. This is a required agent workflow;
the current commit hooks do not check the substance of a record.

Keep existing plans at their established paths and register them here. This index
begins with the current milestone; it is not a complete inventory of older plans.
Add older records when they become relevant, and search existing documentation
before creating a duplicate. Use repository-relative links and exclude private
context, credentials and sensitive local paths.

| Record | Status at last update | Next action |
|:---|:---|:---|
| [Near-zero covariance recovery](2026-10-16-near-zero-column-recovery.md) | Delivered in c87b8e0; fresh pilot retains one iteration limit, 16 October 2026 | Follow the strict-control convergence diagnostic plan before numerical readiness |
| [Strict-control convergence diagnostic](../LMM_RANDOM_SLOPE_CONVERGENCE_NEXT_STEP.md) | Delivered in b4cbf07; rounding-scale stagnation established, 16 October 2026 | Fresh full-method pilot with prospectively frozen control; original failed gate retained |
| [Stopping-control pilot execution](2026-10-16-stopping-control-pilot.md) | Completed locally; original analysis abort and three numerical failures retained, 16 October 2026 | Follow boundary/covariance repair plan; preserve failed readiness and separate diagnostic results |
| [Fresh stopping-control pilot](../LMM_RANDOM_SLOPE_STOPPING_CONTROL_NEXT_STEP.md) | Executed under separate freeze; readiness blocked, 16 October 2026 | Follow retained counterexamples and precision/recovery milestone |
| [Boundary recovery and covariance precision](../LMM_RANDOM_SLOPE_PRECISION_RECOVERY_NEXT_STEP.md) | Planned from three fresh counterexamples, 16 October 2026 | Reproduce failures, repair causes with negative controls/budget/backend checks, then freeze separate validation |
| [Production campaign preparation](../LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md) | Conditional future plan, 16 October 2026 | Follow only after a reviewed fresh pilot meets its prospective numerical-readiness gate |
| [Random-slope scaling and caps](2026-10-16-random-slope-scaling-caps.md) | Delivered in 10f7b78; two numerical blockers retained, 16 October 2026 | Follow the near-zero covariance recovery milestone before confirmation |
| [Random-slope streaming harness](2026-10-16-random-slope-streaming-harness.md) | Delivered in bda59f0, 16 October 2026 | Follow separate scaling/cap and native interruption rehearsal before confirmation freeze |
| [Random-slope confirmation planning](2026-10-16-random-slope-confirmation-planning.md) | Delivered in 6e138f0, 16 October 2026 | Follow the streaming-harness milestone before campaign authorization |
| [Random-slope numerical controls](2026-10-16-random-slope-numerical-controls.md) | Delivered in 2d85393, 16 October 2026 | Follow confirmation planning and its streaming-harness milestone |
| [Random-slope optimizer boundary recovery](2026-10-16-random-slope-optimizer-fix.md) | Delivered in 580b270, 16 October 2026 | Follow the numerical-controls record and its fresh frozen pilot |
| [Random-slope pilot and evidence publication](2026-10-16-random-slope-pilot.md) | Pilot completed; numerical blockers documented, 16 October 2026 | Follow the optimizer recovery record and remaining diagnostics before a fresh frozen pilot |
| [Durable work record workflow](2026-10-16-durable-work-records.md) | Implemented and validated for authorized delivery, 16 October 2026 | Apply the rules on subsequent substantive tasks |
| [Scientific evidence audit and boundary milestone](../SCIENTIFIC_VALIDATION_NEXT_STEP_PLAN.md) | Delivered in 1c5057e; raw archive subsequently published, 16 October 2026 | Follow the completed random-slope pilot record and its numerical findings |

Statuses describe recorded evidence, not authorization to commit, publish or start
another phase. Update the row and record when their status materially changes.
