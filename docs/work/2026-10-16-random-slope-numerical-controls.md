# Random-slope numerical controls and remaining discrepancies

## Status and context

Created prospectively 16 October 2026 (Europe/Berlin). Status: **selected diagnostics
and fresh pilot validated for authorized delivery; confirmation planning remains next**.
Baseline `580b270edfcfd8f544818f3ff8171aacbd30d099`, committed and pushed with hooks
and preflight; HEAD, tracking and live remote match and the checkout was clean.
The user authorized moving to this step after that delivery. The preceding
[optimizer record](2026-10-16-random-slope-optimizer-fix.md) and
[diagnostic evidence](../../scientific/RANDOM_SLOPE_BOUNDARY_FIX_DIAGNOSTIC.md)
remain the historical record. No release or full confirmatory campaign is implied.

## Scope and acceptance criteria

Investigate boundary classification for
`zero_slope-G8-m5-v0-rundefined-beta0.5`, replicate 17, and Satterthwaite inference
for `imbalanced-G8-mcycle3-5-10-v0.25-r0-beta0`, replicate 18. Separate fitting
precision from derivative/inference behavior; use the balanced analytic oracle
only in its supported design. Keep all original inputs, outcomes, gates and
classification thresholds unchanged. New refits are post-study diagnostics.

Choose and justify prospective numerical controls for a distinct fresh-seed pilot
only after resolving or explicitly bounding the observed issues. If an actual
library defect is established, reproduce it with independent regression evidence
before a minimal production fix. An R result alone is not a correctness oracle.

## Intended implementation and verification

Record isolated candidate/native identity, exact selected inputs, controls and
script/source hashes before refits. Compare the old and tighter tolerances and
matched R results; inspect fitted parameters, objective, covariance rank and
Satterthwaite variance gradients/Hessian. Use independent dense linear algebra or
high-precision differentiation to resolve the imbalanced case if needed. Preserve
failed fits and do not aggregate selected reruns into pilot estimates.

Scientific Python tooling uses focused integrity/analytic tests, script lint and
documentation checks; new R diagnostics use required comparison formatting. Reuse
passing unchanged engine checks. If Rust inference changes, add focused regression
coverage, run the broad Rust slice plus Python lint and affected inference tests;
optimizer/math changes additionally require release parity and fair workloads per
OPTIMIZATION.md. Register new integration targets if any. Keep Cargo serialized.

Before any new pilot observations, independently review and freeze the exact new
engine/binary, source/protocol/environment, seeds and analysis rules. Do not reuse
the v1 seed namespace or edit its frozen runner to launch a different study.

## Material findings and observed validation

The [selected-controls report](../../scientific/RANDOM_SLOPE_NUMERICAL_CONTROLS.md)
records all six newly receipted outcomes. At 1e-12 both prior discrepancies pass
the existing gates; at 1e-14 the boundary fit exhausts its 5,000-iteration budget.
Independent dense analytic REML derivatives, including nonstationary parameter
transformation curvature, support fit-point sensitivity as the main earlier
imbalanced discrepancy. No additional production defect was established.

Prepared the separate v2 harness/protocol/R adapter with unchanged 94 designs and
gates, 1e-12/5,000 controls, disjoint v2 seeds and prospective balanced optimum,
precision and rank checks for both engines. The reporting audit independently
checks raw paired decisions and complete balanced diagnostic inventories. V1
sources remain unchanged. Disabled the misleading Windows launcher memory sampler;
v2 records unavailable memory explicitly. Independent review resolved two reporter
integrity gaps before observations and found no remaining pre-freeze blocker.

Six v2 tests, six independent dense controls and the full 89-test script suite
passed, alongside script lint, required R/Julia formatting and documentation
checks (fresh links after final Markdown additions). The exact engine commit's
isolated wheel has the same native identity as the validated fix. No Rust or
binding code changed; reuse its passing engine checks. V2 sources/protocol/review
were frozen before generating any v2 input. All 12 smoke fits/inference outcomes
and numerical/decision/analytic gates passed before the full pilot launched.

## Completed pilot and retained limitations

The [fresh v2 report](../../scientific/RANDOM_SLOPE_PILOT_V2.md),
[ledger](../../scientific/RANDOM_SLOPE_PILOT_V2.json) and
[freeze](../../scientific/RANDOM_SLOPE_PILOT_V2_FREEZE.json) record the evidence.
All 1,880 candidate and R fits converged with all three methods evaluable.
Candidate checks against all 1,840 balanced global optima show zero objective,
covariance-precision or rank misses (maximum criterion error 4.832e-11).
Exactly nine R parity flags remain, coinciding with nine retained R fits above
the analytic optimum; R also has two common-rank mismatches. No fits were replaced.
All three methods agree on every paired rejection/coverage decision. Forty
imbalanced fits meet R parity gates but have no balanced-oracle global guarantee.

The independent post-run AI audit verified all 87 sources, 94 input and 376 outcome
hashes, wheel/native identity and every R environment record. It regenerated all
1,880 responses and independently enumerated balanced active sets, confirming the
candidate gates, the exact nine-flag/R-nonoptimal overlap and all paired decisions.
No evidence-integrity blocker was identified; the pilot's statistical scope and
small replication budget remain explicit limits. This is not human peer review.

All 1,880 same-key response vectors differ from v1, as expected from the disjoint
protocol seed prefix. Original v1 and fixed-candidate diagnostic artifacts remain
unchanged and verifiable. The raw v2 archive includes sources, matching wheel,
locks, smoke/pilot records and selected-control evidence: 30,322,167 bytes,
524 entries, CRC verified. It is local only; the
[descriptor](../../scientific/RANDOM_SLOPE_PILOT_V2_BUNDLE.json) holds its hash.
No new package/release asset was published. After reviewing this completed work,
the user authorized its commit/push, then confirmation planning. Reuse unchanged
passing checks, retain hooks/preflight and verify local/tracking/live remote
revision equality after delivery. The earlier numerical fix was delivered in 580b270.

Four-worker candidate/reference phase wall times were 41.6/62.6 seconds, with one
computational thread per worker and no peak-memory claim. They are descriptive
cost inputs, not fair speed measurements. Twenty replicates per case cannot
confirm a calibration band; retain pointwise intervals/failure envelopes and do
not average across cases to manufacture a global claim.

## Next action

Prepare a separate confirmation-design/planning milestone before any new
confirmatory observations:

1. Declare the primary family: proposed 36 balanced interior null cases times
   Satterthwaite/KR = 72 rejection claims. Null coverage is the paired complement.
   Keep other strata/normal-Wald exploratory unless prospectively changed.
2. Specify practical bands, all-attempt failure envelopes, familywise error and
   a global decision-power target. Compute exact Bonferroni Clopper–Pearson
   acceptance counts and planning power; do not reuse the old 48-claim correction
   or assume 20,000 replicates is adequate. Include near-band-edge and failure
   sensitivity. Validate the planner with independent exact/binomial controls.
3. Address the nine retained R optimization failures prospectively. Preserve v2;
   any reference robustness experiments need separately named diagnostic receipts.
   Do not choose a comparator by p-value or silently substitute an oracle fit.
4. Budget compute, disk, analysis and numerical audits from this run, including
   uncertainty and the planned reference subset. As an illustration only, 20,000
   candidate replicates across all 94 cases scale to about 11.6 hours at this
   pilot's four-worker wall rate; all-paired R adds about 17.4 hours. This is not
   an approved or reliable production budget. Stream inputs/analysis and bound
   bootstrap memory instead of scaling the small-pilot arrays blindly.
5. Freeze and independently review an exact candidate, protocol, held-out seed
   namespace, environment, implementation, failure rules and resource budget.
   Obtain authorization for the actual full campaign after concrete planning.

Recheck existing records and hashes before continuation. No source or outcome
inside the completed v1/v2 freezes should be edited to accommodate future work.
