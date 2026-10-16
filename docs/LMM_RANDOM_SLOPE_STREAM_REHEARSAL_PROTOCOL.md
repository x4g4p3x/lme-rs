# Random-slope streaming resource rehearsal protocol v1

Prepared before observations, 16 October 2026. This is an engineering rehearsal,
not confirmatory calibration. The actual manifest freezes sources and exact
budgets before input generation. Confirmation-sized execution remains disabled.

Use all 94 v2 designs, the unchanged development native binary from engine commit
`580b270edfcfd8f544818f3ff8171aacbd30d099`, REML and candidate tolerance 1e-12,
max_iterations=5000, require_convergence=false with explicit convergence checks.
Generate **two fresh datasets per case** (188 total) in a new
`lmm-random-slope-stream-rehearsal-v1` protocol namespace, distinct from all
earlier inputs. Use the case ID, replicate and named rehearsal namespace as the
deterministic seed; worker/chunk order cannot affect responses. Retain all inputs.
Use compressed input chunks of at most eight datasets, one SQLite WAL/FULL
append-only journal per case, at most four native workers and one library thread
per worker. The SQLite transaction commits each input chunk with its index.
Commit every attempt start before fitting and each outcome after completion.
Ordinary exceptions are failures; interrupted starts become durable failures on
resume. Never restart an attempted fit. Corrupt records fail integrity checks.

Run all four predeclared R variants on both replicates in every case (752
reference attempts), retaining every baseline and alternate start. Use the
locked R/packages, matched formula/rows/REML and controls in the
[confirmation proposal](LMM_RANDOM_SLOPE_CONFIRMATION_PROPOSAL.md).
Select by smallest criterion among converged finite valid-covariance fits;
exact ties use bank order. Eligibility is generic, including imbalanced cases,
without a p-value or balanced-oracle selection filter. Apply numerical/method
checks after selection and retain failures without switching variants.

Stream all-attempt counts, bounded secondary moment estimates, independent
balanced global REML checks and known-variance GLS. Require coherent p/interval,
finite positive df/SE and unchanged numerical tolerances. Compute the proposed
72-claim failure envelopes for software verification, explicitly labelled
rehearsal; this sample size cannot establish the intended rejection band.
Do not bootstrap: secondary estimates have no bootstrap interval claim here.
Export bounded raw rows for independent R checking of counts, distributions,
reference selection, numerical tolerance arithmetic, GLS and CP bounds.

Acceptance for this engineering rehearsal: all 188 expected input/candidate keys
and 752 reference keys retained exactly once; source/native/environment identity
checks pass; all workers exit successfully; no candidate balanced numerical miss;
all 72 endpoint calculations independently agree with R; complete failure/flag
inventories are preserved. Numerical/inference disagreements or failures are
reported, not silently excluded or used to repair completed results. A failed
rehearsal is retained; changes require a new freeze/namespace. No scientific
power/calibration or general optimizer-reliability conclusion is permitted.

Measure actual native Python workers' own process high-water working set and
R's self-reported native PID, phase wall times, analysis time and physical journal
storage. Per-process high-water sums are conservative sums, not a measured
simultaneous peak. Missing OS counters are explicit unavailable measurements and
prevent resource acceptance. Worker exit/environment receipts are audited before
calling execution accepted. Report every R variant's exception, interrupted,
optimizer-nonconvergence, invalid-fit and method/numerical failure counts, as well
as selected-comparator outcomes. Owned process groups/jobs clean up native
children even if their Python root exits abruptly. Record analysis runtime
versions separately from the candidate environment.
Startup and 94 small databases can dominate a two-replicate rehearsal; do not
linearly scale fixed overhead into a supposed production prediction interval.
Compare counts and input hashes under changed chunk/worker order using fixtures;
exercise real process crashes, immutable writes and corruption rejection before
this run. No full campaign or package release follows automatically.

Before freezing final confirmation, review this execution/analysis implementation,
measure scaling on a larger separate rehearsal if needed, set time/RAM/disk caps,
resolve retained numerical/reference limitations, and obtain campaign authorization.
