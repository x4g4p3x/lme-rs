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
| [Random-slope confirmation planning](2026-10-16-random-slope-confirmation-planning.md) | Planning implemented and validated locally, 16 October 2026 | Implement/test the streaming harness, fresh resource rehearsal and final freeze before campaign authorization |
| [Random-slope numerical controls](2026-10-16-random-slope-numerical-controls.md) | Delivered in 2d85393, 16 October 2026 | Follow confirmation planning and its streaming-harness milestone |
| [Random-slope optimizer boundary recovery](2026-10-16-random-slope-optimizer-fix.md) | Delivered in 580b270, 16 October 2026 | Follow the numerical-controls record and its fresh frozen pilot |
| [Random-slope pilot and evidence publication](2026-10-16-random-slope-pilot.md) | Pilot completed; numerical blockers documented, 16 October 2026 | Follow the optimizer recovery record and remaining diagnostics before a fresh frozen pilot |
| [Durable work record workflow](2026-10-16-durable-work-records.md) | Implemented and validated for authorized delivery, 16 October 2026 | Apply the rules on subsequent substantive tasks |
| [Scientific evidence audit and boundary milestone](../SCIENTIFIC_VALIDATION_NEXT_STEP_PLAN.md) | Delivered in 1c5057e; raw archive subsequently published, 16 October 2026 | Follow the completed random-slope pilot record and its numerical findings |

Statuses describe recorded evidence, not authorization to commit, publish or start
another phase. Update the row and record when their status materially changes.
