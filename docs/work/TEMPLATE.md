# Task or milestone title

Copy this template to `YYYY-MM-DD-short-topic.md`, replace its prompts, remove
inapplicable sections, and register the record in the index. Prefer an existing
record if it already owns this work. Keep the record concise and link canonical
reports and evidence rather than duplicating them.

## Status and context

- Created: date and whether prospective or retrospective.
- Last updated: date and timezone.
- Status: planned / in progress / blocked / implemented locally / delivered.
- Starting revision and relevant working-tree changes: record the actual baseline.
- Delivery: pending, or a verified commit/PR reference. Do not invent a future SHA.

## Scope and acceptance criteria

State the requested outcome, boundaries and observable completion criteria.
Separate the current task from future proposals. Do not infer new authorization
from a next-action entry.

## Intended implementation and verification

Before implementation, summarize the approach and choose required checks from
`AGENTS.md`. Distinguish planned checks from observed passing results. Record
material changes to the plan and their reasons as they occur.

## Material findings and decisions

Capture discoveries, decisions, constraints and rationale needed by later work,
with links to code, tests, reports or review findings. Preserve superseded decisions
through a dated correction; do not rewrite observed failures into successes.

## Implementation and observed validation

Summarize what actually changed and why. List actual commands/checks, outcomes,
and evidence references; identify failures, skipped/unavailable checks and their
reasons. Identify the revision or working-tree state tested. Keep bulky output
outside this record and distinguish local artifacts from public evidence.

## Limitations and next action

State unresolved questions, accepted limits and the exact next step. Before each
commit, review this record against the diff and checks, update its status and index
row if needed, and include those updates with the work. Never include secrets,
personal information, sensitive paths or raw conversation transcripts.
