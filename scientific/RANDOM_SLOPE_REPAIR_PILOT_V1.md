# Near-zero covariance repair and fresh numerical pilot v1

16 October 2026. **The two false-convergence regressions are repaired; pilot
readiness remains false because one honest iteration-limit failure is retained.**
This exploratory numerical study does not establish nominal calibration, general
optimizer reliability or release readiness. Full confirmation remains disabled.

## Repair and independent regression evidence

Both immutable [scaling counterexamples](RANDOM_SLOPE_SCALING_V1_COUNTEREXAMPLES.json)
failed new public Rust API regressions before production edits, reproducing
objective gaps `0.06063436058` and `0.00534295505`. A separate synthetic quadratic
also reproduced the skipped joint escape. Tiny residual off-diagonals exceeded
the old whole-column coordinate threshold despite negligible objective effect.

The repair in `src/optimizer.rs` compares the cost of zeroing the column with
the incumbent. It retains a genuinely improving zero point or probes joint
curvature at an objective-equivalent, exactly zero origin. Curvature uses that
origin's cost, while acceptance requires improvement over the original best by
the existing objective gate. No coordinate threshold or numerical oracle gate
was relaxed. Feasibility, true-boundary behavior and shared caller budgets remain
checked; this local recovery is not a global-optimality certificate.

Both regressions pass under default and archived starts across five response/
predictor unit transformations: 20 fits. Default and Basin recovery/boundary/
iteration-budget tests pass. Original historical fits and their failed counts
remain unchanged; the regression inputs are diagnostic, not fresh study data.

## Frozen candidate and observations

The [protocol](../docs/LMM_RANDOM_SLOPE_REPAIR_PILOT_PROTOCOL.md) preceded this
pilot's inputs. All **114 sources**, the **64 engine source hashes** and exact
pre-build receipt were frozen before generating 1,880 fresh datasets, 20 per
unchanged v2 case, under namespace `repair-pilot-v1-20261016-a`. Use the unchanged
REML/tolerance `1e-12`/5,000-iteration controls and fixed four-variant R bank.
The engine is explicitly identified as working-tree sources based on
`10f7b7817acd2ec5f2f3fc535587603cc6d5e106`; it is not claimed to be that unchanged
commit. Python 3.11.15, Polars 2.0.0 and platform match the previous candidate.

- Installed native SHA: `34edf7fa6bdf11a5fc0581c14d4ca11a18c017781a1e97eb1e3b0469f44f59cb`.
- Wheel SHA: `b8e3b970c6b9a15d644abb50b87bf2d5d6f6a5714abe803c2faffb1a4d827fa5`.
- Exact [build receipt](RANDOM_SLOPE_REPAIR_PILOT_V1_BUILD_INPUTS.json) and
  [source/native freeze](RANDOM_SLOPE_REPAIR_PILOT_V1_FREEZE.json) are retained.

| Inventory or diagnostic | Observed |
|:---|---:|
| Candidate starts/outcomes | 1,880 |
| R starts/outcomes | 7,520 |
| Converged candidate balanced numerical misses | 0 |
| Candidate genuine iteration-limit failures | 1 |
| Selected-reference numerical/method failures | 0 |
| Eligible paired fits per method | 1,879 |
| Paired numerical/rejection/coverage discrepancies | 0 |
| Unselected R balanced numerical misses | 20 |
| Unselected R nonconvergence | 1 |

All 1,840 balanced inputs have analytic controls; 1,839 candidates are evaluable
and agree, while one fails explicit convergence. The other 40 imbalanced inputs
have no balanced oracle and pass the declared reference/inference checks. The
four R variants and all failures remain retained. Objective-only selection finds
an eligible reference for every dataset. Unselected R failures do not erase
candidate failures or invalidate a correct selected reference.

The [sole candidate diagnostic](RANDOM_SLOPE_REPAIR_PILOT_V1_BUDGET_DIAGNOSTIC.json)
is `interior-G50-m15-v0.25-r0-beta0.5`, replicate 15. Its criterion
`2361.163568871662` matches the analytic optimum to displayed double precision;
residual-variance difference is about `-5.45e-9`, maximum covariance difference
`4.90e-8`, with rank two. Nevertheless, it reports `IterationLimit`, 5,000
iterations, `converged=false`. All three method outcomes stay ineligible. No
replacement, changed tolerance or retroactive convergence label was used. This
is honest nonconvergence under the frozen rule, not a wrong converged optimum.

The case is outside the primary family; all 72 primary failure counts are zero.
All 72 narrow calibration rules fail at this small n, as expected. Conditional
agreement of the other 1,879 pairs cannot remove the observed failure or make the
predeclared readiness gate pass. The
[acceptance record](RANDOM_SLOPE_REPAIR_PILOT_V1_ACCEPTANCE.json) therefore records
engineering acceptance **true**, numerical readiness **false**, confirmation
claim **false**.

The unchanged [independent R check](RANDOM_SLOPE_REPAIR_PILOT_V1_R_CHECK.json)
verified raw eligibility/selection/p/CI, binary counts in 564 rows and all 72
primary bounds; maximum CP difference was `1.12e-16`, known-GLS difference
`1.34e-15`. Separate AI recomputation regenerated all 1,880 inputs, matched all
9,400 starts and 9,400 outcomes, independently enumerated 1,840 balanced spectral
optima and reproduced all secondary moments/counts/bounds. Maximum converged
candidate objective gap was `1.87e-11`; independently recomputed versus stored
oracle criteria differed by at most `9.10e-13`. The R checker itself does not
validate secondary bias/RMSE/SE/width moments. See the
[original summary](RANDOM_SLOPE_REPAIR_PILOT_V1.json) and
[AI review](../docs/LMM_RANDOM_SLOPE_REPAIR_REVIEW.md).

## Resources, performance and validation

Candidate, reference, Python analysis and independent R verification ran
sequentially without concurrent tests/builds/rehearsals. All obeyed the frozen
900 s / 4 GiB job committed-memory / 512 MiB sampled study disk / 1 GiB free-volume
limits, with clean owned-job completion. The parent, environment and later
archive preparation are outside those limits. Sampled guards can overshoot;
committed bytes, raw OS peak charge and resident high-water remain distinct.

| Phase | Cleanup-inclusive wall time | Maximum sampled job commit |
|:---|---:|---:|
| Candidate | 55.068 s | 423.71 MiB |
| Four-variant R | 282.450 s | 1,841.99 MiB |
| Python analysis | 10.157 s | 73.57 MiB |
| Independent R verification | 5.086 s | 152.88 MiB |

Maximum sampled study disk was 23.524 MiB, largest actual sampling gap 0.1672 s.
Physical journals occupy 14,753,792 bytes: input payload 4,199,666, outcome payload
4,658,535, other pages/indexes/metadata/space 5,895,591. The
[resource receipt](RANDOM_SLOPE_REPAIR_PILOT_V1_RESOURCES.json) retains complete
worker/end metrics. Separate capped analyses bind command/executable/script and
input/output/thread identities; engineering acceptance does not assert numerical
readiness.

The [performance receipt](RANDOM_SLOPE_REPAIR_PILOT_V1_PERFORMANCE.json) covers five
Rust-only fair cases with three warmups, 11 measurements and one library thread.
Default before/after objective/coefficient/convergence/iteration checks are
identical in cold and prepared fits. Cold medians changed by roughly +0.64%,
+1.22%, +1.57%, +6.25% and +0.76% for sleepstudy, weighted sleepstudy, large random
slopes, random-intercept control and crossed control respectively. The scalar
control's prepared median fell slightly while its cold median increased. Retain
these observations; one ordered before/after session cannot establish a speedup,
general non-regression, Julia parity or backend dominance. The five Basin fair
cases also completed with converged measured fits, without a Basin-before sample.

Validation passed: `task rust` (four documented heavy production cases ignored),
Python Ruff, two release golden tests, five Basin slope contract tests, three
Basin boundary tests and nine Basin recovery unit tests. All 138 script tests
passed with R available; four affected build-receipt tests passed again after
the final archive guard. A supplementary documentation benchmark-test invocation
skipped two optional R tests because its PATH omitted R; those tests already
passed in the full run. Required documentation components and script lint passed.
No bindings source changed; the isolated wheel and native pilot were exercised,
but a new complete bindings suite, full CI/hosted matrix and production-load
campaign were not run.

## Preservation and next action

The [local raw bundle descriptor](RANDOM_SLOPE_REPAIR_PILOT_V1_BUNDLE.json) records
a **39,888,649-byte, 543-entry** SHA/CRC-verified archive. It includes the exact
source/build freeze, new wheel/environment metadata, journals/logs, raw analysis,
independent checks, failed/fixed regression logs and raw performance reports.
Installed environments are excluded; the archive is local and **not uploaded**.
Public original-file copies match retained bytes. Earlier sources/outcomes are
preserved and historical engine freezes are checked against archived sources.

Next follow the [strict-control convergence diagnostic plan](../docs/LMM_RANDOM_SLOPE_CONVERGENCE_NEXT_STEP.md).
Determine whether the iteration limit reflects search behavior, arithmetic
resolution or a contract defect before changing convergence logic or prospective
controls/readiness policy. Preserve this failed gate. Only a newly reviewed
protocol and fresh pilot can support later numerical eligibility, followed by
the conditional [production preparation plan](../docs/LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md).
AI review is internal scrutiny, not human statistical peer review or certification.
