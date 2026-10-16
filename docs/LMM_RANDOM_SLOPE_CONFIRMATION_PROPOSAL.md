# Gaussian random-slope confirmation proposal

Prepared 16 October 2026 after the completed v2 pilot. **Planning proposal, not a
frozen protocol, an executed confirmation or authorization to launch one.**
The implementation milestone below must precede a final freeze and review.
The [planning report](../scientific/RANDOM_SLOPE_CONFIRMATION_PLANNING.md) provides
the exact calculations, resource estimates and selected reference diagnostics.

## Aims, scope and data-generating process

Assess null slope rejection calibration of the development engine with the v2
controls, separately from numerical correctness and cross-engine agreement.
Neither matching R nor a small pilot establishes calibration. Results apply only
to the declared design, candidate binary, controls and methods.

Retain the 94-case v2 registry and Gaussian generating model from the
[v2 protocol](LMM_RANDOM_SLOPE_PILOT_V2_PROTOCOL.md). Primary cases are the 36
balanced interior cases with true fixed slope zero: G=8,20,50; m=5,15; slope
variance .05,.25; correlation -.5,0,.5. Intercept variance .5, residual variance 1,
fixed intercept 2, centered predictors of within-group mean square one.
Fit `y ~ x + (1 + x | group)` with REML. Preserve exact rows and responses.

Candidate-only primary family: **36 cases times two methods = 72 claims**,
Satterthwaite and Kenward–Roger, two-sided slope test at .05. The same dataset
serves both methods; 72 claims do not require 72 separate dataset banks.
Normal-Wald results, positive-slope power, boundary/imbalanced cases, reference
calibration, failure rates and other performance summaries remain exploratory.
At the null, coherent 95% interval coverage is the complement of rejection; do
not count it as another independent primary claim. Any expanded primary family
requires a revised calculation and freeze before generating confirmation data.

Propose **30,000 datasets per primary case** (1,080,000 datasets; 2,160,000 primary
method evaluations), plus **2,000 per remaining case** (116,000 datasets).
Propose R comparison on the first **1,000 predesignated replicates in every case**
(94,000 datasets). Subset selection is by replicate ID, before seeing outcomes.
Numerical candidate checks apply to every balanced dataset, not only this subset.

Candidate: engine source commit `580b270edfcfd8f544818f3ff8171aacbd30d099`,
v2 isolated wheel/native identity, tolerance 1e-12, maximum 5,000 iterations,
default start and `require_convergence=false`; explicit convergence is still
required for evaluability. No per-dataset candidate retry or oracle substitution.
If engine sources, dependencies or controls change, stop and justify a new
candidate, smoke and freeze instead of treating it as this candidate.

## Estimands, failures and primary decision

For each case/method define evaluability E prospectively: converged fit; finite,
valid covariance and residual variance; valid finite method SE, statistic,
denominator df, p and interval; and independent numerical/self-consistency checks.
Require positive SE/df/critical value, nonnegative F, p in [0,1], matched method
coefficient and coherent test/interval decisions. Reuse v2 numerical tolerances
and common rank definition without loosening them to obtain a pass. The future
runner must encode and test each rule before the final freeze.

For n prespecified attempts, record K evaluable rejections (`p < .05`) and m
unevaluable outcomes. Never delete, replace or silently retry failures. Define
observable iid-replicate endpoints:

- `p_lower = P(E and reject)`;
- `p_upper = P(E and reject) + P(not E)`.

Every hypothetical assignment of failed outcomes to reject/nonreject has rate
within this partial-identification envelope. No missing-at-random assumption is
needed. These endpoints do not define an unobservable p-value for a failed fit.
K and K+m are ordinary Bernoulli-sum counts under the iid replicate-outcome
model. Infrastructure-wide aborts and correlated resource failures require an
execution-integrity investigation; a partial run is not a completed iid study.

With M=72, familywise alpha .05 and `t=.05/(2*M)`, compute:

- lower endpoint bound `L = BetaQuantile(t; K, n-K+1)`, or 0 if K=0;
- upper endpoint bound `U = BetaUpperQuantile(t; K+m+1, n-K-m)`, or 1 if K+m=n.

Certify a claim only if **L >= .04 and U <= .06**. All 72 must certify for the
global statistical decision. The union bound over the 144 one-sided endpoint
bounds gives at least 95% simultaneous envelope coverage, without independence
between methods/cases. These are conservative exact Clopper–Pearson bounds;
[R's binomial-test documentation](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/binom.test.html)
describes their coverage property. Familywise uncertainty is distinct from the
probability that a study will meet its decision rule.

For n=30,000, complete-data acceptance is K in **[1,318,1,661]**. With m failures,
observed K must be in `[1318, min(n-m,1661-m)]`; an empty range cannot certify.
Report the raw attempted/evaluable/failure counts, endpoint bounds and per-claim
decisions, including non-supported claims. No outcome-based stopping, added
replicates, dropped cases or post-hoc band changes. A new design after results
requires fresh data and a separately labelled study.

Balanced global-REML objective, covariance precision and rank are also numerical
guards. Mark failed numerical checks unevaluable for the envelope and retain all
raw values; any observed candidate numerical miss blocks an overall numerical
validation statement even if statistical envelopes happen to certify. Observed
zero misses does not prove a zero population failure probability. The balanced
oracle does not apply to the imbalanced cases. Publish numerical and statistical
decisions separately; exploratory findings cannot repair a failed primary gate.

## Decision-power assumptions and sensitivity

The proposed >=90% global decision-power target is a design choice declared
before the planning calculation, not a universal statistical standard.
For planning only, condition on a fixed m and model evaluable rejections as
`Binomial(n-m,p_valid)`. Sum the two tails outside the admissible observed-count
range to get single-claim failure probability q, then report
`max(0,1-72*q)` as a global power lower bound. This assumes the stated common
conditional rate/failure scenario for every claim, without between-claim
independence. It is not an empirical estimate from 20 pilot observations.

22,000 is the smallest passing **tested** ideal budget, with no failures and
`p_valid=.05`; it is not a proven global minimum over every integer n. The 30,000
proposal adds margin in the .1% fixed-failure scenario: joint bound .982000016.
Sensitivity fractions use decimal ceiling rounding of n times the fraction.
At 30,000 and no failures, p_valid=.045 or .055 gives single-claim powers only
.8171 and .6159; the global union-bound lower bound is uninformative zero.
At either band boundary, containment has very low power. Do not promise 90%
power uniformly throughout [.04,.06] or interpret a zero lower bound as proof
that the actual joint power is zero. See the full stored sensitivity grid.

## Reference optimizer policy

R is a secondary comparator, never automatic truth. Retain the exact R/package
lock, formula, data rows, REML and one-thread settings. For each predetermined
reference dataset, run all four fixed bobyqa variants (maxfun=100000):

| Order | Start theta | Other optimizer controls |
|:---|:---|:---|
| baseline | default | default |
| tight_default | default | rhobeg=.1, rhoend=1e-10 |
| tight_positive | [.5,.25,.5] | rhobeg=.1, rhoend=1e-10 |
| tight_negative | [.5,-.25,.5] | rhobeg=.1, rhoend=1e-10 |

Among converged finite fits with valid covariance, select the smallest REML
criterion; exact ties use fixed bank order. Select without examining candidate
agreement, p-values, df or interval decisions. Retain all attempts, warnings,
errors, timings and the original baseline. If the chosen fit fails numerical
checks or method inference, retain the failure; do not switch to another variant
based on inference. Report selected and baseline comparisons separately.

The selected nine-case diagnostic recovered 9/9 numerical optima with this bank;
tighter default tolerance alone recovered 3/9. These were selected problem cases,
not a random validation sample or a bound on future recovery. Imbalanced fits
have no balanced global oracle. Pre-freeze rehearsal must exercise the bank on
fresh representative inputs, including already well-behaved and imbalanced cases.

## Next implementation milestone and release gates

Implement a **new streaming confirmation harness**, keeping v1/v2 immutable.
The current v2 runner is not ready to scale unchanged to this campaign.

1. Create a new protocol/seed namespace, disjoint from all earlier pilots,
   diagnostics and the 0.2.7 study. Fix case IDs, replicate IDs, integer budgets,
   method/failure rules and reference subset before generation. Record exact
   source/wheel/native/lock hashes and verify installed bytes. Reuse generating
   mathematics only after its generator/control tests pass.
2. Stream deterministic responses per case/replicate with at most four workers
   and one BLAS/OpenMP/Polars thread each. Store compressed bounded-size chunks
   with manifests, ordered keys, checksums and a durable write-ahead attempt
   ledger. Inputs are committed before fits; start/outcome records are flushed.
   Crash/exception/timeout counts remain failures. Resume only never-attempted
   IDs; retain interrupted attempts, prohibit duplicate completed IDs and never
   overwrite outcomes. Test crash/restart, truncated chunks and worker-count
   independence with controlled fixtures before any scientific run.
3. Compute all-attempt counts and analytic checks incrementally. Retain method
   metrics and all flagged pairs without loading millions of fit dictionaries.
   Reproduce independent known-variance GLS controls and balanced spectral
   identities. Verify consistency of p/CI, rank and covariance handling with
   focused invalid/boundary fixtures.
4. Implement exact primary endpoint bounds and independent count verification
   in Python/R. Secondary bias/RMSE/SE calibration/width and paired differences
   stay descriptive. If bootstrap intervals are reported, fix 1,000 resamples
   and disjoint analysis seeds; process one resample/chunk at a time or use
   memory-mapped sufficient metrics. Preserve paired dataset indices for paired
   summaries. Never allocate a `1000 x n` full-fit resampling tensor.
5. Add the fixed R bank, complete attempt inventory and objective-only selection
   tests. Verify numerical and method failures cannot disappear through reference
   selection or reporting. Keep candidate and reference eligibility separate.
6. Run deterministic execution/analysis fixtures, then a small **fresh rehearsal**
   in its own namespace. Measure actual native workers' peak memory, phase wall
   time, write/analysis throughput and disk use; test chunk-size-independent
   decisions and comparison of streamed vs small dense results. Rehearsal inputs
   are excluded from confirmation. Any correction gets a new rehearsal receipt.
7. Update the resource provision from rehearsal, finish a separate internal AI
   methodological/implementation review, freeze the actual sources/protocol and
   obtain campaign authorization. AI review is useful independent scrutiny, not
   a human statistician's endorsement. Do not launch on this proposal alone.

Provisional four-variant, case-weighted budget for the proposed primary plus
secondary/reference subset: **11.73 phase-wall hours**, or **23.46 hours** using
an explicit twofold provision; **8.42 GB decimal** at twice the estimated raw
input/outcome volume. These are pilot extrapolations, not measured production
bounds. They exclude analytic audit, summary/bootstrap analysis, environment
copies and archives. Establish scratch/archive allowance and a measured RAM cap
in the rehearsal; the v2 pilot did not measure native peak memory. A resource
abort is retained and does not permit an outcome-based replacement campaign.

The next deliverable is the tested streaming harness, rehearsal resource report
and final freeze package. No new calibration, general library validation,
release-readiness or performance claim follows from this planning milestone.
