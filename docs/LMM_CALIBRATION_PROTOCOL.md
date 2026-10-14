# LMM calibration protocol v3

Status: Frozen internal confirmatory protocol, 13 October 2026 (superseding exploratory
pilot protocol v2 of 11 October 2026). The user chose internal AI methodological and
implementation review in place of recruiting an external human statistician. See the
[pilot methodological critique](LMM_CALIBRATION_AI_REVIEW.md),
[pilot implementation review](LMM_CALIBRATION_IMPLEMENTATION_REVIEW.md),
[pilot results](../scientific/CALIBRATION_PILOT_0.2.7.md), and
[confirmatory review](LMM_CALIBRATION_CONFIRMATORY_REVIEW.md). Freeze this document,
the generator, case registry, R environment lock, candidate binaries and analysis
script together before confirmatory execution. The
[scientific validation plan](SCIENTIFIC_VALIDATION_PLAN.md) defines the larger
programme. This initial protocol applies only to Gaussian random-intercept models
with an intercept and one continuous fixed predictor.

## Aim and estimands

Evaluate finite-sample inference for the slope beta1 in
`y_ij = beta0 + beta1*x_ij + b_i + e_ij`, with independent
`b_i ~ N(0, tau2)` and `e_ij ~ N(0, sigma2)`. Fix beta0 = 2 and total
variance tau2 + sigma2 = 1. The target is the fixed population slope; it is not
the predictive interval for a future observation or group.

The primary estimands are null rejection probability at two-sided alpha = 0.05
and coverage of a two-sided 95% interval. Secondary measures are slope bias,
RMSE, empirical standard deviation, mean reported standard error, power,
variance-estimate bias, boundary rate, runtime and fitting/inference failure rate.

## Data-generating cases

Use the complete factorial below, yielding 72 correct-model cases. Case IDs
must encode every factor and remain stable when the execution order changes.

| Factor | Values |
|:---|:---|
| Number of independent groups G | 6, 12, 30 |
| Observations per group m | 3, 10 |
| Intraclass correlation rho | 0, 0.2, 0.7 |
| Predictor location | within group, between groups |
| beta1 | 0, 0.5 |

Set tau2 = rho and sigma2 = 1 - rho. Construct a fixed predictor rather than
redrawing it in every replicate: equally spaced values on [-1, 1], centered
and scaled to population mean square one. For the within-group case, repeat
the same m predictor values in every group; for the between-group case,
repeat each of the G predictor values m times. Save the exact design arrays.
These are conditional-on-design simulations; they do not establish results
for every random-covariate distribution.

Add a separately labelled imbalance stratum at G = 6 and 12, rho = 0.2 and 0.7,
beta1 = 0, both predictor locations, with group sizes cycling through 2, 4,
8. This adds eight cases. Center within-group predictors separately in each
group; retain the between-group values from the balanced design. Do not pool
the imbalance stratum into the primary balanced results.

rho = 0 explicitly probes a variance-component boundary. Report that stratum
separately; regular interior asymptotics need not apply there. Non-Gaussian
and heteroscedastic stress tests require a later protocol revision.

## Methods and matching

Fit the exact same rows, formula `y ~ x + (1 | group)`, REML, no weights and no
offset in all implementations. Compare lme-rs Satterthwaite and Kenward-Roger
inference with independent R lme4/lmerTest/pbkrtest equivalents. Include an
unadjusted normal-Wald baseline using the returned slope and standard error,
with the standard normal critical value and tail probability in the harness.
This is the library's default LMM confidence-interval convention. Its
residual-df test API is restricted to OLS and is not an LMM study method.
All methods receive identical response arrays. Save options and explicit
optimizer/convergence controls; archive package versions, R dependency lock
and binary hashes. Verify the R 4.6.1 environment against
[scientific/renv.lock](../scientific/renv.lock) before reference execution.
Do not infer independence from agreement between Rust, Python and MCP/SCP.

For scalar tests, save beta, the method's covariance/SE, denominator df, t or F,
and p-value. Use the same method's adjusted variance and df for its interval.
For KR and Satterthwaite, check that the scalar F test and its interval use the
same adjusted covariance, rank one and scalar scaling one. The Python API does
not expose adjusted covariance: obtain its scalar variance through a
unit-displacement contrast against beta_hat minus one (variance = 1/F), avoiding
beta_hat squared divided by a near-zero observed F. Verify that both the
observed and unit-displacement contrasts have numerator df one, positive finite
unit F, positive finite denominator df, matching denominator df
(relative/absolute tolerance 1e-12), and observed F matching beta_hat
squared/variance (relative tolerance 1e-8, absolute tolerance 1e-10). Record the
isolated candidate Python interpreter version and implementation in
`candidate-environment.json`. In the R reference adapter, verify `KRmodcomp`
against `pbkrtest`'s exposed `vcovAdj` with scaling one.
Distribution probabilities and t quantiles come independently from R for every
attempted candidate and reference fit. LRT/ML and parametric bootstrap require
distinct protocols and budgets; they are not included in the primary REML comparison.

Include a known-true-variance Gaussian GLS z control, with variance blocks
constructed from generating rho. For balanced designs its slope variance is
sigma2/(G*m) within groups and (tau2+sigma2/m)/G between groups. Diagnostic
interior REML df are G*(m-1)-1 and G-2 respectively; these are not imposed on
boundary estimates. Neither control is a replacement for candidate results.

## Randomness, sample size and uncertainty

Use SHA-256 counter uniforms with 52-bit open-interval values and Box-Muller
normal pairs. The exploratory pilot (`lmm-calibration-v2`, `namespace = "pilot"`)
is keyed by `lmm-calibration-v2|pilot|case ID|replicate ID|counter` with
1 to 100 replicates per case. The confirmatory study (`lmm-calibration-v3`,
`namespace = "confirmatory"`) is keyed by
`lmm-calibration-v3|confirmatory|case ID|replicate ID|counter`. Pilot and
confirmatory seed namespaces are strictly disjoint; pilot results are never
pooled into the confirmatory sample. Archive exact generated response arrays in
deterministic per-case compressed shards (`cases/<case_id>/inputs.jsonl.gz`,
zero gzip timestamp and empty header filename) with per-case and aggregate
SHA-256 digests: cross-platform math-library rounding is not claimed bitwise
identical. Worker count (`1 <= workers <= 16`) and case execution order must not
affect the generated inputs, fits or summaries. Each worker process sets
`MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`POLARS_MAX_THREADS=1`.

The confirmatory repetition budget is **20,000 replicates per case** across all
80 designs (**1,600,000 candidate and true-variance GLS datasets**; 480,000
primary null datasets across the 24 balanced interior null cases). At a true
rejection probability of 0.05 with no failed fits, the [exact binomial planning
calculation](../comparisons/scientific/lmm_confirmation_planning.R) gives a
single-claim support probability of 99.794% (eligible rejection counts 905..1,094)
and a union-bound joint probability of at least **90.12%** across all 48 primary
claims under the Bonferroni Clopper-Pearson rule (`alpha = 0.05 / 48`), compared
with 79.84% single-claim and 0% union-bound lower bound at n = 10,000.

In the confirmatory campaign:

- All **20,000 replicates per case** (1,600,000 datasets) are evaluated by the
  isolated published 0.2.7 candidate engine (`normal`, `satterthwaite`,
  `kenward_roger`), the known-true-variance Gaussian GLS control, and R's
  independent reference distribution evaluator (`qnorm`, `pnorm`, `qt`, `pf`,
  plus exact `qbeta` Clopper-Pearson bounds for n = 20,000).
- A fixed prespecified paired reference audit subsample of the first **1,000
  confirmatory replicates per case** (`0 <= replicate < 1000`, **80,000 paired
  datasets** across the 80 designs; 10x the pilot sample size) is additionally
  fitted in R (`lmerTest::lmer`, `contest1D`, `pbkrtest::vcovAdj`,
  `pbkrtest::KRmodcomp`), with exact `qbeta` Clopper-Pearson bounds at n = 1,000
  and paired candidate-versus-reference differences on the common-evaluable
  datasets.

Report Wilson 95% Monte Carlo intervals for rejection/coverage rates and MCSE
for every performance measure. For bias, use sd(beta_hat - beta1)/sqrt(R).
Use sqrt(mean(SE squared))/empirical SD as the SE calibration ratio; mean SE
is descriptive. Report mean interval width and both noncoverage tails.
Estimate uncertainty of RMSE, empirical SD, SE calibration and means using
1,000 paired dataset bootstrap resamples per case. The separate seed is SHA-256 of
`lmm-calibration-v2|analysis|pilot|case ID` for pilot runs and
`lmm-calibration-v3|analysis|confirmatory|case ID` for confirmatory candidate/control
methods (plus `lmm-calibration-v3|analysis|confirmatory|reference|case ID` for the
paired reference subsample when `reference_replicates < replicates`),
interpreted as a big-endian integer for Python's `Random` (archive Python version
and analysis source). Stream analysis case-by-case so memory remains bounded at
20,000 replicates per case. Compare candidate and reference methods with paired
differences on the prespecified paired reference replicates. Account for Monte
Carlo uncertainty rather than demanding identical percentages between runs.

## Decision rules fixed before confirmation

For each balanced interior null case, the target practical rejection band is
[0.04, 0.06] and coverage band [0.94, 0.96]. Classify an estimate with its
conservative 95% simultaneous Clopper-Pearson failure envelope: wholly inside
the band (`lower >= 0.04` and `upper <= 0.06` for rejection; `lower >= 0.94`
and `upper <= 0.96` for coverage) supports the scoped claim (`supported`);
wholly outside (`upper < 0.04` or `lower > 0.06`) contradicts it
(`contradicted`); overlap with a band edge is `inconclusive`.
Boundary and imbalance cases are reported individually as exploratory
calibration evidence. Power is descriptive, not required to match a fixed
target. These bands are engineering acceptance criteria, not a universal
scientific standard or a guarantee of exactness for approximate methods.

There are 24 balanced interior null cases and two candidate adjusted methods:
the joint family contains 48 distinct rejection claims. Null coverage is its
complement when test and interval agree; verify that identity rather than
counting it twice. R and normal-Wald methods are descriptive comparators.
Use Bonferroni Clopper-Pearson intervals at per-claim alpha 0.05/48 for the
global decision; show pointwise Wilson intervals for description. Simultaneous
bounds on comparators are descriptive, outside this declared candidate family.
Require the conservative failure envelope described below, not only the
conditional evaluable interval, to fall inside the acceptance band. Do not
replace casewise failures with an average. Require all 24 primary cases for a
candidate adjusted method to be `supported` before calling that method validated
for this declared primary scope. Record any `inconclusive` or `contradicted`
cases explicitly as gaps. The confirmatory CLI refuses any replicate count other
than 20,000 in the `confirmatory` namespace.

## Failures and exclusions

Every attempted response stays in the denominator for fitting/inference
failure rates. Record nonconvergence, exceptions, nonfinite outputs, singular
fits and invalid df separately. Nonconvergence is unevaluable even when outputs
are finite. Singularity uses theta < 1e-4 in both engines; a finite singular fit is not automatically a
failure; preserve it and assess whether the requested inference is evaluable.
No automatic retries or deletions in the primary run. A separate diagnostic
retry can be reported with its own counts and settings.

Coverage and rejection rates among evaluable fits must state that denominator
and quantify selection: report the worst-case bounds obtained by treating
each failed replicate as either a success or a failure of the binary outcome.
Also compare observable design/response summaries between evaluable and
failed replicates (retaining aggregate evaluable/failed response-mean and
response-SD summaries plus individual records for failed replicates in
confirmatory mode so report files stay bounded). Do not count an unevaluable
fit as a non-rejection or a covered interval. No per-dataset result is replaced
with the reference result. For r observed binary successes, f failures and n
attempts, identification bounds are [r/n, (r+f)/n]. The conservative uncertainty
envelope uses the lower Clopper-Pearson bound for r and upper bound for r+f,
with the declared pointwise or simultaneous alpha.

## Review and archival gate

The AI critiques were performed by separately tasked agents which did not edit
the candidate or harness. Their objections and the implementing agent's responses
are recorded. These agents belong to the same AI system: this is internal
methodological and implementation review, not independent human peer review or
endorsement. Archive the protocol and registry, source and environment locks,
binary hashes, generated inputs/seeds, every attempted outcome, analysis code,
raw results and report. A clean-environment rerun should reproduce deterministic
inputs and agree within the declared numerical and Monte Carlo uncertainty.

This follows the ADEMP structure described by
[Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086): aims,
data-generating mechanisms, estimands, methods and performance measures.
