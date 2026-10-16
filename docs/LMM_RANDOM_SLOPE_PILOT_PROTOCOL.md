# Gaussian random-slope pilot protocol: prospective draft

Prepared 16 October 2026. Status: **design only, not frozen or executed**.
This follows the [random-intercept audit](../scientific/BOUNDARY_AUDIT_0.2.7.md)
and [evidence guide](../scientific/INFERENCE_EVIDENCE_GUIDE.md). It neither changes
the completed v3 study nor grants its claims to a different engine binary.

## Aim and estimand

Assess harness correctness, identifiability, numerical/reference agreement,
fitting/inference failures and cost before a new confirmatory study of the
population fixed slope beta1 in:

```text
y_ij = 2 + beta1*x_ij + b0_i + b1_i*x_ij + e_ij
(b0_i,b1_i) independent across groups, jointly Gaussian
Var(b0)=0.5, Var(b1)=v1, Corr(b0,b1)=r
e_ij independent N(0,1), independent of random effects
```

Fit `y ~ x + (1 + x | group)` by REML, matching rows, predictor values, random
effects correlation parameterization and explicit controls. No weights, offsets,
missingness or model misspecification are included. Predictor values vary within
every group; this protocol does not attempt to identify a random slope from a
predictor constant within each group.

Primary future estimands are two-sided alpha=0.05 slope rejection and 95%
interval coverage. Bias, RMSE, empirical SD, root-mean-square SE/empirical SD,
interval width, power, component/correlation estimates, singularity and failure
rates are secondary. Avoid relative variance/correlation error when its true
denominator is zero or the correlation is undefined.

## Fixed design registry

In each group, take m equally spaced values on [-1,1], center them and scale
their population mean square to one. Save the exact arrays. The covariance
generating parameters are not a single ICC: marginal variance changes with x.

| Stratum | Factors | Cases |
|:---|:---|---:|
| Balanced interior | G={8,20,50}; m={5,15}; v1={0.05,0.25}; r={-0.5,0,0.5}; beta1={0,0.5} | 72 |
| Zero slope variance | Same G,m,beta1; v1=0; r=null (undefined) | 12 |
| Near-perfect correlation | G={8,20}; m=5; v1=0.25; r={-0.95,0.95}; beta1=0 | 4 |
| Exact correlation boundary | G={8,20}; m=5; v1=0.25; r={-1,1}; beta1=0 | 4 |
| Imbalanced | G={8,20}; sizes cycle [3,5,10], truncated after G groups; v1=0.25; r=0; beta1=0 | 2 |

Total: **94 cases**, each with a stable ID encoding every factor. Do not create
three copies of a v1=0 case using arbitrary correlation labels. Center/scale
predictors separately within each imbalanced group.

Generate random effects with independent standard Gaussian z0,z1:
`b0=sqrt(0.5)*z0`, `b1=sqrt(v1)*(r*z0+sqrt(1-r*r)*z1)` for v1>0.
At v1=0 set b1=0 and treat r as absent. Exact r=+/-1 uses a rank-one generating
covariance, rather than adding numerical jitter to force positive definiteness.

## Methods and independent controls

Compare candidate Satterthwaite and KR with matched R lme4/lmerTest/pbkrtest
methods. Normal-Wald is descriptive. Save scalar estimates, method covariance,
df, F/p and interval limits; verify rank-one test/interval consistency. An
exposed adjusted covariance is preferable; a unit-displacement public contrast
is an auditable alternative if its rank/df and observed F are checked.

Include known-variance Gaussian GLS using each true block
`V_i = I + Z_i*Sigma*Z_i'`, `Z_i=[1,x_i]`. Validate that oracle first on small
rational dense-matrix identities and an independently derived block calculation.
At v1=0 check the generating model against a random-intercept model with
tau2=0.5, sigma2=1. The fitted larger model may estimate a nonzero slope variance:
do not demand identical inference from differently parameterized fitted models.

Record nonconvergence, exceptions, invalid/nonfinite estimates, invalid df,
singularity and correlation-boundary behavior separately. Derive a common
singularity diagnostic for covariance eigenvalues before execution; the scalar
random-intercept theta threshold cannot simply be reused for a 2x2 covariance.
If using package-specific singularity flags, preserve each definition and do not
pretend they are identical. Finite evaluable singular fits remain included.

Compare reconstructed covariance matrices in the declared intercept/slope order,
using the same predictor centering and scaling; raw Cholesky parameters can have
different conventions. Freeze absolute and relative comparison tolerances before
examining pilot outputs, including explicit near-zero rules.

For implementation review, propose a common diagnostic with
`scale=max(trace(Sigma_hat),sigma2_hat)>0`: count eigenvalues greater than
`1e-8*scale` to determine rank. Both zero random variances give rank zero.
Eigenvalues below `-1e-12*scale` flag invalid covariance; retain tiny negative
roundoff values in records and count them as zero only for this diagnostic.
Never alter the fit or add jitter. Report estimated correlation as undefined
when either diagonal variance is at most `1e-8*scale`. Preserve the raw matrix
and each package's flags; unknown diagnostics are not nonsingular outcomes.
These are proposed engineering thresholds, to be checked with deterministic
negative controls before freezing, rather than statistical standards.

## Pilot size, seeds and budget

The proposed harness/cost pilot is **20 replicates per case: 1,880 datasets**,
all paired with R. Its planning MCSE near probability 0.05 is about 0.0487, so
it cannot demonstrate a 4%–6% calibration band. Start with tiny diagnostic smoke
fixtures before the full pilot; retain failed attempts and name all diagnostic
variants. No automatic per-dataset retry or result substitution is allowed.

Pilot tables must include pointwise exact binomial intervals and worst-case
failure envelopes for rejection/coverage, as well as failure-rate intervals.
Zero failures in 20 attempts still has a two-sided 95% Clopper-Pearson upper
bound of about 16.8%; a zero plug-in MCSE does not demonstrate reliability.

Use a new registry/protocol ID, for example `lmm-random-slope-pilot-v1`, and
separate smoke, pilot, confirmation and analysis namespaces. Derive each input
from protocol/case/replicate/counter as in the existing documented hash generator.
Save exact responses, source/environment hashes and compressed per-case records.
Worker count/order must not change inputs or scientific outcomes.

Begin with at most four worker processes, one BLAS/OpenMP/Polars thread each,
streaming analysis per case. Measure memory/runtime before increasing resources.
Do not reuse the old scalar study's performance estimate as a random-slope cost
measurement. Budget diagnostic fits, distribution evaluation, R reference fits,
analysis and archive space separately.

## Prospective confirmatory decision design

The 36 balanced interior null cases would yield **72 distinct primary rejection
claims** across the two adjusted candidate methods. Null coverage is the exact
complement when intervals and tests are paired, not an additional independent
claim. Other strata remain exploratory unless a new protocol explicitly changes
that designation before execution.

Proposed practical bands remain rejection [0.04,0.06] and coverage [0.94,0.96].
Use simultaneous exact Bonferroni Clopper-Pearson failure envelopes for a global
claim, alongside descriptive pointwise intervals. Every attempted fit enters
failure denominators; conditional binary estimates also need worst-case bounds
for unevaluable outcomes. Preserve each method/case failure; do not substitute
an average over cases or silently restrict the scope after results are known.

Recalculate repetition requirements for 72 claims using the actual exact
interval rule and a declared global decision-power target. Do not automatically
copy the old 20,000 budget or 48-claim correction. Ideal-p=0.05 planning does not
guarantee support when true rates approach a band edge or failures occur. Fix
bootstrap repetitions, paired resampling, independent analysis seeds and MCSE
reporting before confirmation. Pilot observations never enter that sample.

## Freeze checklist before this pilot can run

- [ ] Select an exact candidate release/commit and binary; assess the custom-control fix for that version. Do not mistake the development editable extension for published 0.2.7.
- [ ] Implement and test all 94 IDs, arrays, generating covariances, zero/rank-one controls and seed separation.
- [ ] Choose matching explicit fit controls and numerical comparison tolerances from precision/identities, before examining pilot outputs.
- [ ] Restore/verify and archive the R/Python environments; check reference versions and source licenses.
- [ ] Resolve covariance singularity diagnostics, inference API support, failure policy and resource limits.
- [ ] Test record/hash/order integrity, distribution/interval pairing and bounded analysis; freeze sources and manifest.
- [ ] Record a separate AI methodological/implementation critique and resolutions, accurately labelled internal review.

An internal AI design review on 16 October 2026 checked the 94-case count,
36 interior null cases, rank-one generation and undefined-correlation cases.
Its requests for explicit pilot uncertainty, covariance-order comparisons and
zero/roundoff diagnostic rules are incorporated above. Implementation review
and threshold checks remain prerequisites to freezing this draft.

This draft is ready for implementation review. It does not authorize a full
confirmatory campaign or imply that random-slope inference is already validated.
