# LMM calibration protocol v1

Status: review-ready draft. No confirmatory study has run, and independent
scientific review has not yet occurred. Freeze this document, the generator,
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
covariance. LRT/ML and parametric bootstrap require distinct protocols and
budgets; they are not included in the primary REML comparison.

## Randomness, sample size and uncertainty

Use a documented counter/seed derivation keyed by protocol ID, case ID and
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
Estimate uncertainty of RMSE and SE calibration using a declared paired
replicate bootstrap with a separate analysis seed. Compare methods with
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

Report all casewise estimates and simultaneous uncertainty for the family
of primary null-case claims (a prespecified Bonferroni interval alongside
pointwise intervals). Do not replace casewise failures with an average.
Require all primary cases to support a claim before calling that method
validated for this declared primary scope. Record inconclusive cases as gaps.

## Failures and exclusions

Every attempted response stays in the denominator for fitting/inference
failure rates. Record nonconvergence, exceptions, nonfinite outputs, singular
fits and invalid df separately. A finite singular fit is not automatically a
failure; preserve it and assess whether the requested inference is evaluable.
No automatic retries or deletions in the primary run. A separate diagnostic
retry can be reported with its own counts and settings.

Coverage and rejection rates among evaluable fits must state that denominator
and quantify selection: report the worst-case bounds obtained by treating
each failed replicate as either a success or a failure of the binary outcome.
Also compare observable design/response summaries between evaluable and
failed replicates. Do not count an unevaluable fit as a non-rejection or a
covered interval. No per-dataset result is replaced with the reference result.

## Review and archival gate

An independent reviewer still needs to be recruited. Seek a statistician with
experience in mixed models, small-sample inference and simulation-study design,
who did not implement the candidate or select its tolerances. Provide this
protocol, the numerical baseline's scoped results, model/API contracts and a
short list of open questions: scalar KR test/interval consistency, the primary
case family, simultaneous decision rules, boundary treatment and the proposed
failure bounds. Ask for written objections and resolutions, rather than a
general endorsement of the library. Agree scope, compensation if applicable,
and consent for any attribution before commissioning the review.

Before confirmation, obtain an independent statistician's review of the
estimand, design, interval/test pairing, practical bands, simultaneous
uncertainty, failure policy, reference implementation and seed separation.
Record the reviewer, date, objections and resolutions with their consent.
Then freeze a reviewed protocol revision and its hash. A pilot may identify
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
