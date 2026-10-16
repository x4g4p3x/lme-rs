# Random-slope streaming execution and rehearsal

Created before implementation, 16 October 2026 (Europe/Berlin). Status: implemented
and small rehearsal validated, reviewed for authorized delivery.
Baseline `6e138f0289e06b34a5075fa3edfec13b6af331d2` is pushed with enabled
hooks/preflight; HEAD, tracking and live remote agree, checkout initially clean.
The user authorized this next step after delivery of the
[planning milestone](2026-10-16-random-slope-confirmation-planning.md).

## Scope and acceptance criteria

Implement the [proposal](../LMM_RANDOM_SLOPE_CONFIRMATION_PROPOSAL.md)'s bounded
execution/analysis contract and a separately frozen small resource rehearsal.
Preserve all v1/v2/0.2.7 results and frozen sources. No engine, production
dependency, package release or full confirmatory campaign is authorized here.
Keep confirmation-sized execution disabled in this rehearsal implementation.

Use deterministic, fresh, case/replicate keyed inputs in compressed bounded
chunks. Use one SQLite WAL/FULL append-only attempt/outcome journal per case,
OS-released exclusive case locks and commit each start before running a fit.
Persist input chunks first; hash/validate exact inventories and JSON payloads.
Resume only never-attempted IDs; interrupted starts become retained failures,
not retried fits. Preserve orphan/partial chunks and reject corruption instead
of silently repairing or overwriting. Keep each reference variant separately
journaled; generic reference selection uses convergence/finite/covariance
eligibility and objective, without an inference or balanced-oracle filter.

Stream per-case metrics and all-attempt primary failure envelopes; retain raw
method payloads and independent balanced/global and known-variance controls.
No bootstrap tensor or per-campaign list of fits. Measure actual loaded native
workers' OS peak working set, R's own PID, phase wall time and on-disk bytes.
State that a short rehearsal does not establish a long-run resource bound.

## Intended validation

Before scientific observations, exercise forced interruption, no-retry resume,
duplicate/invalid keys, altered/truncated input chunks and output hashes,
exclusive locking, deterministic input independence of chunk/worker order,
bounded iteration, generic reference eligibility and exact endpoint decisions.
Cross-check small streamed summaries against independent raw counts and R CP
bounds. Freeze sources, protocol and native identity before a new rehearsal;
retain any failed execution and use a new namespace/freeze after changes.
Run the full script suite, script lint/format, required R/Julia formatting and
documentation checks. Reuse unchanged engine checks; no Rust/bindings edits.
Arrange a separate AI implementation/method review of concrete code/results.

## Findings, validation and next action

Implemented the durable store, four-worker candidate and persistent R execution,
complete four-variant accounting, streamed analysis, independent R checker and
actual native-process high-water measurement. The
[rehearsal protocol](../LMM_RANDOM_SLOPE_STREAM_REHEARSAL_PROTOCOL.md) and exact
source/native freeze preceded inputs. All previous study sources remain unchanged.

The [report](../../scientific/RANDOM_SLOPE_STREAM_REHEARSAL_V1.md),
[machine ledger](../../scientific/RANDOM_SLOPE_STREAM_REHEARSAL_V1.json),
[resource receipt](../../scientific/RANDOM_SLOPE_STREAM_REHEARSAL_V1_RESOURCES.json)
and [AI review](../LMM_RANDOM_SLOPE_STREAM_REVIEW.md) are canonical evidence.

- All 188 candidate and 752 reference durable start/outcome keys are present
  exactly once across 94 journals; independently regenerated responses agree.
  All eight workers exited zero; environment and receipt acceptance passed.
  Candidate balanced numerical misses and selected-reference numerical/binary
  disagreements are zero. All three methods are evaluable for the 188 candidate
  and 188 selected-reference outcomes. Four imbalanced inputs have no global oracle.
- One unselected tight-default R bobyqa code-3 nonconvergence remains retained,
  including fit, warning and all method-failure flags. Do not confuse 188 selected
  comparator outcomes with all 752 raw attempts or call all R attempts successful.
- R independently verifies 564 count rows, exact 72-claim inventory, reference-bank
  selection/variant failures, numerical tolerance arithmetic, GLS and CP endpoints.
  Maximum CP difference 1.11e-16 and GLS difference 1.33e-15. The analytic global
  oracle itself remains the separately reviewed spectral implementation.
- Candidate phase 8.265s, reference phase 29.333s, streamed analysis 1.879s.
  Native Python candidate peaks 115.95–117.35MiB; native R 406.37–413.59MiB.
  Reference Python/R peak sum 1,781.67MiB is not a simultaneous/whole-machine
  peak. Journals occupy 4,136,960 bytes, analysis artifacts 838,846 bytes; sources,
  installed environments and archive copies are additional. Short-run fixed costs
  prevent turning these observations into a production bound or speed claim.
  Reference timing overlapped documentation validation; future scaling must use
  isolated measurement. Analysis-loop time excludes imports/source/receipt setup.
- Pre-freeze AI review prompted stronger execution acceptance, exact independent
  inventories/hashes/bounds, orphan cleanup, complete variant failure states,
  numeric-bool rejection and source/runtime provenance. All have regressions.
  A bounded-memory test exposed an unnecessary 8MiB decompression allocation;
  fixed chunked reads before observations rather than loosening the test.
- Raw source/journal/log/audit evidence is preserved in a local 2,355,602-byte,
  315-entry SHA/CRC-verified archive; it has not been uploaded. Public summary and
  R-check bytes match raw outputs. Exact archive identity is in its descriptor.

Validation: **118 script tests passed** with R available and no skips, including
20 new stream/store tests and positive/negative independent R controls;
`task lint:scripts`, `task lint:comparisons:required` (R and Julia available) and
`task docs:check` passed. Documentation checks include portable example checks,
six doctests and rustdoc. Read-only source/native/journal/response/count/resource
review is independent. No full CI, production-load campaign, new Rust/binding
suite, release or public raw upload ran; engine sources/dependencies are unchanged.
Planning delivery `6e138f0` is pushed. The user authorized commit/push of this
validated milestone and continuation into scaling/resource caps on 16 October
2026. Reviewed scope, frozen-source integrity and evidence before delivery;
keep hooks enabled.

## Concrete next milestone

Keep confirmation disabled. Prospectively specify a separate larger scaling
rehearsal (for example all 94 cases at 10 and 20 replicates, each with a fresh
namespace and new protocol/freeze) to separate startup/database fixed costs from
per-input fit, durable I/O and analysis costs. Retain every attempt and the complete
four-variant R bank. Do not silently override this two-replicate frozen protocol.

Add explicit phase RAM/disk/time caps and controlled native-worker interruption/
resume beyond the Python-store fixtures. Verify no attempted dataset is rerun,
resource failures are visible, and hashes/decisions remain independent of chunk
and worker order. Decide whether optional bootstrap intervals are needed; if
implemented, test one-resample-at-a-time or bounded memory-mapped paired metrics.

Update the resource provision, complete final execution/analysis review, and
prepare the actual confirmatory source/protocol freeze for explicit campaign
authorization. The 188-dataset rehearsal does not establish calibration, general
optimizer reliability or a production resource guarantee.
