# Random-slope streaming rehearsal v1

16 October 2026. **Completed small engineering/resource rehearsal, not scientific
confirmation.** The user authorized implementation after planning delivery
`6e138f0289e06b34a5075fa3edfec13b6af331d2`. No library engine, production
dependency, release or previous study was changed.

## Implemented behavior and prospective design

The [protocol](../docs/LMM_RANDOM_SLOPE_STREAM_REHEARSAL_PROTOCOL.md) and
[source/native freeze descriptor](RANDOM_SLOPE_STREAM_REHEARSAL_V1_FREEZE.json)
preceded observations. All 94 v2 designs have two fresh datasets: **188 candidate
fits** and **752 R attempts**, all four fixed variants on every dataset.
The new protocol/namespace is disjoint from all earlier inputs. It uses the same
reviewed development binary and fitting controls as v2, whose engine source is
`580b270edfcfd8f544818f3ff8171aacbd30d099`.

The new runner stores compressed bounded input chunks and fit outcomes inside
one SQLite WAL/FULL append-only journal per case. Input chunks and their indexes
commit atomically; starts commit before fits, outcomes afterward. Exclusive OS
locks release on process death. Interrupted starts become retained failures;
resuming never retries attempted IDs. The public entry points reject confirmation
mode and budgets above 20 per case. This is a rehearsal implementation.

The fixed R bank uses generic convergence/finite/covariance eligibility and the
smallest criterion, with fixed-order exact ties. Balanced-oracle checks and method
inference occur after selection. No inference-friendly fallback is allowed.
All variants retain separate failure classifications and flags. Candidate and
selected-reference counts are separate from the full raw R attempt inventory.

Analysis streams inputs/outcomes, maintains bounded counts and secondary moments,
and exports compressed raw rows for independent checking. It does not allocate a
campaign-sized fit list or bootstrap tensor. Secondary means/RMSE/SE/width are
descriptive; no bootstrap intervals were generated.

## Results and retained failure

The [result ledger](RANDOM_SLOPE_STREAM_REHEARSAL_V1.json) records:

- All **188 candidate** and **4 x 188 reference** durable start/outcome keys are
  present exactly once across 94 journals. Independent regeneration matches all
  188 retained response vectors, and all source/input/output identities agree.
- All eight workers exited zero; execution/end/environment/memory receipts pass
  the acceptance audit. Actual native process IDs supplied the memory counters.
- Candidate fits converged and all three methods are evaluable on all 188 inputs.
  There are **zero balanced candidate numerical misses**; the global oracle
  applies to 184 balanced inputs and is unavailable for four imbalanced inputs.
- Selected references are evaluable for all three methods on all 188 inputs.
  There are **zero numerical or rejection/coverage disagreements** for the
  candidate/selected-reference comparisons under unchanged v2 gates.
- **One of 752 R attempts did not converge**: `tight_default`,
  `zero_slope-G8-m5-v0-rundefined-beta0`, replicate 0. Bobyqa returned code 3:
  a trust-region step failed to reduce its criterion. Its criterion equals the
  analytic optimum, but optimizer convergence was not established, so it remains
  ineligible. Its raw fit, warning, all three method failures and flagged row are
  preserved. The other 751 R attempts converged. No retry or replacement occurred.

This is engineering acceptance with a retained reference failure, not evidence of
a zero optimizer failure probability. Two replicates per case cannot support the
proposed rejection band. None of the 72 software-evaluated primary rules certifies
calibration, and `confirmation_claim` is explicitly false.

## Independent checks and implementation review

The [R check](RANDOM_SLOPE_STREAM_REHEARSAL_V1_R_CHECK.json) independently reads
all 188 raw rows and verifies **564 count rows**, exact unique primary/count
inventories, reference-bank selection and variant failure counts, distributions,
numerical tolerance arithmetic, known-variance GLS, and all **72 CP endpoint
calculations**. Maximum absolute CP difference is **1.11e-16**; independent GLS
coefficient/SE difference is **1.33e-15**. It verifies manifest/audit hashes and
rejects missing bounds, duplicated claims/count rows, missing balanced oracle
records and changed input identity. The analytic global oracle itself remains
the separately reviewed spectral implementation; this R check is not a second
derivation of that oracle.

A separate [internal AI review](../docs/LMM_RANDOM_SLOPE_STREAM_REVIEW.md)
identified pre-freeze issues that were fixed and regression-tested: worker receipt
acceptance, exact claim inventories, child-process cleanup, complete R failure
accounting, numeric booleans, source inventory and analysis runtime provenance.
Final data/source/journal/resource checks were independent of the implementation
work. AI review does not substitute for a human statistician's endorsement.

## Observed resource use

Four workers, one library thread each; Windows native high-water working sets.
The [resource receipt](RANDOM_SLOPE_STREAM_REHEARSAL_V1_RESOURCES.json) contains
individual measurements and exact phase times.

| Measurement | Observed |
|:---|---:|
| Candidate phase wall time | 8.265 s |
| Four-variant R phase wall time | 29.333 s |
| Python streamed analysis loop | 1.879 s |
| Candidate Python native peak, per worker | 115.95–117.35 MiB |
| Reference Python native peak, per worker | 34.17–35.06 MiB |
| R native peak, per worker | 406.37–413.59 MiB |
| Sum of reference Python/R process-lifetime peaks | 1,781.67 MiB |
| Analysis Python native peak | 97.34 MiB |
| Physical case journals/locks after fits | 4,136,960 bytes |
| Analysis files, including audit and R check | 838,846 bytes |

Per-process peak sums are **not a measured simultaneous or whole-machine peak**;
they exclude parent/background processes and system file cache. Source snapshots,
environment and archive storage are additional. R workers retain a native
high-water mark over their process lifetime, not an incremental per-fit cost.
The reference phase shared the host with documentation validation. Its wall time
includes that background load; future scaling measurements should run in
isolation. The analysis timer excludes Python/SciPy import and source/receipt
verification setup; independent R verification is additional.

Startup and 94 small databases are material fixed costs in a two-replicate run.
Do not scale these wall/disk totals linearly into a production promise or speed
comparison. The previous 23.46-hour/8.42-GB planning provision remains a proxy,
not a newly measured full-campaign bound. A larger fresh scaling rehearsal must
separate fixed costs from per-attempt durable I/O/fit/audit costs and set explicit
time/RAM/disk caps before enabling a confirmation-sized run.

## Reproducibility and remaining work

All **118 script tests passed**, including both existing R-dependent cases and
20 new streaming/store tests. Fault controls exercise real process exit after a
durable start, root-crash orphan cleanup, failed R startup, duplicate writes,
exclusive access, changed/truncated payloads, bounded 10,000-input iteration,
worker/chunk-independent responses, retained failures, bad worker receipts and R
report-corruption rejection. Script lint/format, required R/Julia formatting and
documentation checks passed; six doctests and portable examples passed. No full
CI, production-load campaign or new Rust/binding test suite was run: those sources
are unchanged and prior engine validation is retained.

Raw directories:
`benchmark-results/scientific/random-slope-pilot/20261016-stream-rehearsal-a`
and `20261016-stream-rehearsal-a-analysis`. The
[local bundle descriptor](RANDOM_SLOPE_STREAM_REHEARSAL_V1_BUNDLE.json) records a
**2,355,602-byte, 315-entry** archive with SHA/CRC verification. It contains the
source freeze, compressed journals, logs, raw audit, analysis and independent R
check. Installed environments are excluded; the candidate wheel is the separately
hashed v2 binary. The archive is **local, not uploaded**. Checked-in summary/R-check
bytes match their original raw files; freeze/resources are descriptive receipts.

Next: add a prospectively specified larger scaling rehearsal and explicit resource
caps, validate end-to-end native interruption/resume beyond the controlled store
fixtures, decide bounded optional bootstrap reporting, and prepare the final
confirmatory implementation/protocol freeze. Keep confirmation disabled until
that review and explicit campaign authorization. This milestone does not change
earlier scientific decisions or establish general library validity.
