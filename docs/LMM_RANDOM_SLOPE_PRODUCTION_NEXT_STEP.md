# Next engineering milestone: production campaign preparation

Created 16 October 2026. Start this milestone only after the repaired engine's
fresh numerical pilot and independent review have no candidate/selected-reference
blocker. Follow the [repair record](work/2026-10-16-near-zero-column-recovery.md).
This plan authorizes no confirmation inputs, production campaign or release.

## Fixed scientific target

Preserve the [confirmation proposal](LMM_RANDOM_SLOPE_CONFIRMATION_PROPOSAL.md):
36 primary cases at 30,000 candidates, the other 58 at 2,000, and a fixed 1,000
reference datasets per case, each with all four R variants. That is 1,196,000
candidate fits plus 376,000 reference fits, with 94,000 paired datasets.
Keep the 72 primary claims, all-attempt failure envelopes, simultaneous exact
Clopper-Pearson bounds, eligibility, reference selection and numerical gates.
No outcome-based extension, replacement or changed calibration band.

## Implement and test before enabling the campaign

1. Create a prospective living work record and a separate production protocol/
   runner. Keep every prior scientific source, input and outcome immutable. Do
   not remove the small-pilot guards from existing runners or silently reuse
   diagnostic/pilot namespaces for confirmation.
2. Design bounded deterministic job scheduling for the fixed case/replicate bank.
   A 900-second whole-phase cap from the small rehearsal cannot simply be applied
   to a million-fit phase. Declare job sizes, worker/thread limits, aggregate
   memory, whole-study/scratch/archive disk budgets, free-volume reserve, per-job
   time caps and whole-campaign stop policy before observations. Preserve started
   attempts on abort and resume only never-started keys; retain failures rather
   than resetting denominators. Treat resource acceptance separately from
   statistical failure handling and final calibration decisions.
3. Exercise scheduling, crash/restart, source/input changes, duplicate starts,
   failed workers, limit events and incomplete final handshakes with distinctly
   labeled engineering fixtures. Verify worker/chunk/job-order independence,
   exact candidate/reference inventories and original outcome/start preservation.
   Reuse the reviewed Windows owned-job and durable-store helpers where their
   contracts apply; unsupported hard-limit backends must fail closed.
4. Provision bounded, resumable analysis and independent verification. The
   rehearsal analysis clock extrapolates beyond a short per-job cap; choose
   analysis checkpoints/segments and immutable aggregate/output receipts. Verify
   exact counts, failure envelopes, p/CI, reference selection, numerical controls
   and independently computed secondary moments. Do not pretend the R binary-
   count checker validates bias/RMSE/SE/width moments. Keep descriptive bootstrap
   intervals deferred or prospectively specify their estimator, seeds, pairing,
   resample count, memory bound and independent checks before including them.
5. Run separately capped storage/analysis load fixtures with explicitly synthetic
   fit records, not confirmatory statistical data. Choose sizes and stopping rules
   in advance. Measure native process/job memory, actual sampling gaps, disk
   payload/pages/indexes/unused space, serialization/checksum/commit cost, audit
   growth, scratch and archive duplication. Preserve failures. The two uniform-
   budget fully paired scaling samples cannot directly predict the differently
   weighted full campaign and sparse reference subset; retain uncertainty and
   negative/noisy intercepts. Do not call a two-point line a production bound.
6. Review actual source/binary/environment identities and all protocol gates with
   an independent AI reviewer, clearly labeled as internal scrutiny. Freeze the
   final engine/wheel/native, generator, scheduler, analysis, independent checker,
   tests, controls, seed namespaces and resource plan. Run required code/docs/
   script/feature/performance checks selected from AGENTS.md; record skips and
   supported-platform limits. Prepare the concrete immutable campaign manifest
   and reviewable cost/storage/stop summary before requesting campaign approval.

## Completion and handoff

Deliver the tested production implementation, engineering load evidence,
resource/storage provision with explicit uncertainty, final protocol/source
review and campaign manifest. Keep the campaign disabled until the user explicitly
authorizes that concrete run. Human statistical peer review remains distinct from
AI scrutiny. A numerical pilot or successful resource rehearsal does not certify
nominal calibration or general library correctness.
