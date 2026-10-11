# LMM calibration protocol v2

Status: AI-reviewed internal pilot protocol, 11 October 2026. No confirmatory
study or independent human peer review has occurred. The user chose an AI
methodological critique in place of recruiting a statistician for this milestone.
See the [review and responses](LMM_CALIBRATION_AI_REVIEW.md). Freeze this document, the generator,
case registry, candidate binaries and analysis script together before execution.
The [scientific validation plan](SCIENTIFIC_VALIDATION_PLAN.md) defines the
larger programme. This initial protocol applies only to Gaussian random-intercept
models with an intercept and one continuous fixed predictor.

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
optimizer/convergence controls; archive package versions and binary hashes.
Do not infer independence from agreement between Rust, Python and MCP/SCP.

For scalar tests, save beta, the method's covariance/SE, denominator df, t or F,
and p-value. Use the same method's adjusted variance and df for its interval.
For KR, check that the scalar F test and its interval use the same adjusted
covariance, rank one and scalar scaling one. The Python API does not expose
adjusted covariance: obtain its scalar variance through a unit-displacement
contrast against beta_hat minus one (variance = 1/F), avoiding beta_hat squared
divided by a near-zero observed F. Verify the observed F against beta_hat
squared/variance and the reference against pbkrtest's exposed vcovAdj.
Distribution probabilities and t quantiles come independently from R.
LRT/ML and parametric bootstrap require distinct protocols and
budgets; they are not included in the primary REML comparison.

Include a known-true-variance Gaussian GLS z control, with variance blocks
constructed from generating rho. For balanced designs its slope variance is
sigma2/(G*m) within groups and (tau2+sigma2/m)/G between groups. Diagnostic
interior REML df are G*(m-1)-1 and G-2 respectively; these are not imposed on
boundary estimates. Neither control is a replacement for candidate results.

## Randomness, sample size and uncertainty

Use SHA-256 counter uniforms with 52-bit open-interval values and Box-Muller
normal pairs, keyed by `lmm-calibration-v2|pilot|case ID|replicate ID|counter`.
Archive exact generated response arrays: cross-platform math-library rounding
is not claimed bitwise identical. Use a documented derivation keyed by protocol ID, case ID and
replicate ID. Save the generated response or a deterministic generator version
and its content hash. Worker count and case execution order must not affect
the data. Pilot and confirmatory seed namespaces must be disjoint.

Run 100 pilot replicates per case for harness and cost assessment only. Use
10,000 confirmatory replicates per case: at a probability of 0.05, the MCSE
is approximately 0.00218. A null rejection estimate then has an approximate
95% Monte Carlo interval half-width of 0.00427. Pilot results cannot be
silently added to the confirmatory sample.

Report Wilson 95% Monte Carlo intervals for rejection/coverage rates and MCSE
for every performance measure. For bias, use sd(beta_hat - beta1)/sqrt(R).
Use sqrt(mean(SE squared))/empirical SD as the SE calibration ratio; mean SE
is descriptive. Report mean interval width and both noncoverage tails.
Estimate uncertainty of RMSE, empirical SD, SE calibration and means using
1,000 paired dataset bootstrap resamples. The separate seed is SHA-256 of
`lmm-calibration-v2|analysis|pilot|case ID`, interpreted as a big-endian integer
for Python's Random (archive Python version and analysis source). Compare methods with
paired differences from the common simulated datasets. Account for Monte
Carlo uncertainty rather than demanding identical percentages between runs.

## Decision rules fixed before confirmation

For each balanced interior null case, the target practical rejection band is
[0.04, 0.06] and coverage band [0.94, 0.96]. Classify an estimate with its
95% Monte Carlo interval: wholly inside the band supports the scoped claim;
wholly outside contradicts it; overlap with a band edge is inconclusive.
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
replace casewise failures with an average. Require all primary cases to support a claim before calling that method
validated for this declared primary scope. Record inconclusive cases as gaps.
The retained 10,000/case confirmatory budget can readily yield an inconclusive
global result even for well-calibrated methods after multiplicity correction.
No confirmation is enabled by the pilot runner. Final sample-size planning and
a separately frozen confirmatory implementation remain required; bands must
not be relaxed in response to pilot results.

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
failed replicates. Do not count an unevaluable fit as a non-rejection or a
covered interval. No per-dataset result is replaced with the reference result.
For r observed binary successes, f failures and n attempts, identification
bounds are [r/n, (r+f)/n]. The conservative uncertainty envelope uses the lower
Clopper-Pearson bound for r and upper bound for r+f, with the declared pointwise
or simultaneous alpha. Retain response mean and spread for selection diagnostics.

## Review and archival gate

The AI critique was performed by a separately tasked agent which did not edit
the candidate or harness. Its objections and the implementing agent's responses
are recorded. Both agents belong to the same AI system: this is internal
methodological review, not independent human peer review or endorsement.
Human review remains a useful additional source of scrutiny, but is not a
prerequisite for the user-authorized pilot. Before confirmation, resolve the
remaining sample-size and environment-restoration questions and freeze a
reviewed confirmatory protocol, generator, analysis and hashes. A pilot may identify
harness bugs; changes to scientific choices require a new version and a
visible explanation before a new held-out confirmatory seed set is used.

Archive the protocol and registry, source and environment locks, binary hashes,
generated inputs/seeds, every attempted outcome, analysis code, raw results
and report. A clean-environment rerun should reproduce deterministic inputs
and agree within the declared numerical and Monte Carlo uncertainty.

This follows the ADEMP structure described by
[Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086): aims,
data-generating mechanisms, estimands, methods and performance measures.
It does not assert that the library has already passed this study.
