# Capped random-slope scaling and interruption rehearsal v1

Declared before observations, 16 October 2026. Engineering only: no confirmation
or release. Keep earlier frozen sources unchanged; this runner/analysis is separate.

Run all 94 v2 cases at **10** and **20** replicates in two separately frozen,
sequential runs: 940/1,880 candidate fits and 3,760/7,520 R attempts. Use four
workers, one library thread each, unchanged candidate/native/source identity and
the fixed four-variant R bank. Reuse the pinned v1 deterministic response helper
with distinct `scaling-v1-...` namespaces; both scaling banks and fault inputs are
disjoint from one another and every prior bank. These samples cannot establish
calibration. No outcome-based extension, selection or hidden replacement.

Freeze and hash a three-manifest bank index before any of the three runs generates
inputs. Enforce distinct budget/scenario suffixes in the namespaces, audit all
three roles and their shared source identity before every launch.
Source/native freeze precedes generation. Inputs/outcomes use the unchanged
WAL/FULL append-only journals. Commit starts before computation. Interrupted or
aborted attempts remain failures; resume only never-started keys. Preserve all
variants, warnings, numerical/method failures and false software decisions.
Record input-generation/persistence, start/outcome commit, fit-callback and journal
validation/recovery wall costs in workers, keeping their timer scopes explicit.

Each candidate/reference phase has **900 seconds**, a **4 GiB aggregate hard
Windows job committed-memory limit**, **512 MiB sampled study-folder disk cap**
and **1 GiB free-volume reserve**, with **0.1 seconds** between polls plus sampling
and handling time. Record the largest actual sampling gap, including startup.
This job includes
native Python descendants and nested R jobs. The parent is outside the job.
Analysis and independent R verification run in separate single-worker capped
jobs with the same limits, recorded separately. Unsupported hard-memory backends
fail closed; this milestone validates the Windows host, not other OS backends.

The job memory ceiling denies allocations beyond its committed-byte limit; it
does not cap resident RAM or whole-machine memory. Job completion messages are
supplementary evidence, not proof that no allocation was denied when absent.
Any detected limit event, failed worker, unavailable measurement or exceeded
time/disk/free-space guard rejects resource acceptance. Keep failed outputs;
resource failure is not an optimizer calibration result. Sampled time/disk may
overshoot between polls and are not kernel filesystem quotas. Do not delete
evidence to get under a cap. Record final bytes, sampled maxima, job commit peaks,
native resident high-water counters, exit status and every abort reason. The
time cap covers worker launch through the final active-job sample; the separate
end-to-end receipt timer includes cleanup. Require zero active job processes,
successful roots, no observed memory-limit events, and a final free-space/disk
sample within limits. A successful root with a live descendant is rejected after
a 0.5-second accounting grace, then its owned job is terminated.
Bind capped analysis/R commands to exact argument hashes, executable/script
hashes, enforced thread limits, immutable input files and completed output hashes.
A separate final acceptance record checks these after both jobs finish.
On this host the OS peak job-memory counter also records a denied large allocation;
retain it as a raw charge diagnostic. Sample current committed memory separately.
An above-limit raw charge conservatively rejects the execution even if no limit
message arrived; it does not mean a successful allocation or resident-RAM breach.

Before either scaling run, validate owned-process time/disk/memory abort controls.
Use synthetic sleep/write/allocation processes, and verify nested child allocation
denial under a low hard limit. Also freeze a separate **three-input, one-case
native interruption control**: the first v2 case, dedicated namespace and one
candidate worker. Load and verify the exact native engine; complete replicate 0,
durably start replicate 1 and atomically publish a fully written, fsynced
announcement of the loaded native PID/input hash before
pausing. The parent intentionally terminates the owned process tree. Resume with
the same source/input identity: preserve replicate 0 bytes and start timestamp,
record replicate 1 as interrupted without rerunning it, and fit only replicate 2.
There are three unique starts/outcomes, one controlled interruption and no R fits.
Fault records are excluded from scaling/calibration data. Preserve the failed
first execution and its successful resume; acceptance is for this fault control,
not an ordinary successful execution.
Revision before a replacement freeze: the first native control demonstrated that
Windows virtual-environment launchers have a different PID from the loaded
interpreter. Require the marker PID in the queried whole-job process inventory,
rather than equality with the root launcher. Preserve that failed first control,
its three source freezes and generated fault inputs; no scaling observations
were generated in that bank. Replacement banks use fresh namespaces.

Measure the two scaling runs without concurrent tests/builds/other rehearsals.
Run independently checked streamed analysis and unchanged R raw-count verifier,
including all 72 endpoint rules, exact inventories, numerical controls and
reference selection. Sources remain frozen through reporting; retain unsuccessful
runs and use a new freeze/namespace after source changes. AI source/data review
does not substitute for human statistical peer review.

Report both measurements and their descriptive secant growth, separating fixed
database overhead from compressed payload/index costs and timer scopes. Two
points with different random inputs do not identify a precise production cost
model or prediction interval. Preserve negative/noisy extrapolations honestly.
No speed comparison, generalized optimizer reliability or nominal-rate claim.
Secondary bias/RMSE/SE/width remain descriptive; bootstrap intervals are deferred
from this resource milestone. Before a full campaign, finish production-scale
analysis/storage provisioning and final protocol/source freeze with explicit
campaign authorization. Full confirmation stays disabled.

Windows semantics follow Microsoft's [job-memory structure documentation](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information),
[nested-job rules](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs)
and [completion-port notification limitations](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_associate_completion_port).
The [whole-job PID inventory](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_process_id_list)
includes processes in nested child jobs, unlike a root launcher PID alone.
