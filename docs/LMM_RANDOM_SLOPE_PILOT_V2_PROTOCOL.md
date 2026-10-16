# Gaussian random-slope pilot protocol v2

Prepared 16 October 2026. This document is fixed before v2 observations; the
run's `freeze.json` determines whether it was frozen successfully. **Exploratory
pilot, not confirmation.** V1 and all earlier studies remain unchanged.

## Prospective design and candidate

Use engine commit `580b270edfcfd8f544818f3ff8171aacbd30d099`, version
`0.2.8.dev0`, built as a separate isolated wheel. Hash native/wheel bytes, full
engine source, harness, this protocol, review, tests and dependency locks before
generation. Verify the installed native is the wheel's native member. Candidate
controls: tolerance **1e-12**, maximum **5,000 iterations**, default start and
`require_convergence=false`. No per-dataset retry, tighter-tolerance fallback,
optimizer substitution or oracle replacement is permitted.

The controls follow separately labelled selected-input diagnostics: 42 covariance
precision misses and two further boundary/inference cases. The two-case diagnostic
also retained nonconvergence at 1e-14. Those inputs/results are excluded from v2.
This is a justified prospective change to controls, not a change to acceptance gates
or evidence of calibration. Default library controls remain unchanged.

Use the same 94-case design and generating model as
[v1](LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md): independent Gaussian random effects with
intercept variance 0.5, residual variance 1, fixed intercept 2 and within-group
centered predictors scaled to mean square one. Fit `y ~ x + (1 + x | group)` by REML.

| Stratum | Factors | Cases |
|:---|:---|---:|
| Balanced interior | G=8,20,50; m=5,15; slope variance=.05,.25; correlation=-.5,0,.5; fixed slope=0,.5 | 72 |
| Zero slope variance | G=8,20,50; m=5,15; fixed slope=0,.5; correlation undefined | 12 |
| Near correlation boundary | G=8,20; m=5; slope variance=.25; correlation=-.95,.95; fixed slope=0 | 4 |
| Exact correlation boundary | G=8,20; m=5; slope variance=.25; correlation=-1,1; fixed slope=0 | 4 |
| Imbalanced | G=8,20; group sizes cycle 3,5,10; slope variance=.25; correlation=0; fixed slope=0 | 2 |

Protocol ID **lmm-random-slope-pilot-v2** prefixes all deterministic generator and
analysis seeds. Smoke and pilot namespaces are distinct from one another and from
every v1 input. Case ID, replicate and protocol determine inputs independently of
worker count/order. Persist exact designs/responses. The smoke has two replicates
of the first case in each stratum plus the largest balanced positive-correlation
case (12 datasets). The pilot has **20 replicates per case, 1,880 datasets**, all
paired with R. This budget cannot establish a 4%–6% rejection band.

## Methods, comparators and independent controls

Report normal-Wald, Satterthwaite and Kenward–Roger slope rejection at alpha=.05
and 95% coverage. Retain all failures, nonconvergence, invalid df/covariance,
singularity and numeric/decision disagreements. Use the existing matched R
lme4/lmerTest/pbkrtest engine, bobyqa maxfun=100000, and verify all versions in
`scientific/renv.lock`. Retain R's results even if its fit is nonoptimal.

Known-variance Gaussian GLS remains a generating-model control, independently
checked with rational dense matrices. V2 also prospectively evaluates each balanced
dataset's constrained global REML optimum with the independently reviewed spectral
active-set formula in `lmm_random_slope_optimum.py`, for **both candidate and R**.
Record objective, residual/covariance precision and common rank agreement. The
oracle is unavailable for the 40 imbalanced attempts; do not extrapolate it there.
Do not replace either fitted engine, select a better fit, or remove disagreements.

Keep v1 engineering gates `atol + rtol*abs(reference)`:

| Quantity | atol | rtol |
|:---|---:|---:|
| Slope coefficient | 1e-5 | 1e-4 |
| Residual variance/covariance entries | 1e-6*S | 1e-3 |
| REML criterion | 1e-6 | 1e-8 |
| Method SE | 1e-6 | 1e-3 |
| Denominator df | .05 | .005 |
| Scalar F | 1e-6 | .002 |
| p | 1e-4 | 0 |

`S=max(1,sigma2_a,sigma2_b,trace(Sigma_a),trace(Sigma_b))`, using the analytic value
as reference for analytic comparisons. Classify covariance with the unchanged
thresholds: rank counts eigenvalues greater than `1e-8*max(trace(Sigma),sigma2)`;
eigenvalues below `-1e-12` times that scale flag invalidity. Correlation is undefined
if either variance is below the rank threshold. These are numerical diagnostics
in the declared predictor coordinates, not universal scientific standards.

## Analysis, resources and decision limits

Preserve original candidate payloads when R adds distribution quantiles/checks.
Use exact pointwise 95% Clopper–Pearson intervals, all-attempt failure envelopes,
failure-rate intervals and binomial MCSE. Report bias/RMSE/SE calibration and
interval width with the v1 paired dataset-bootstrap scheme (1,000 resamples,
disjoint analysis seeds); zero estimated MCSE does not establish reliability.

Use at most four worker processes and one BLAS/OpenMP/Polars thread per worker.
Retain all worker logs and phase wall times as run-specific resource information.
**No worker peak-memory estimate is claimed**: v1's Windows launcher counters
sampled shims. V2 records unavailable counters explicitly rather than reusing them
as native-worker measurements. Historical native-process resource diagnostics
support this bounded pilot; they are not fresh v2 memory observations.

Freeze and review the implementation before either smoke or pilot generation.
The smoke is a harness gate; inspect any failure before launching the pilot. Keep
failed smoke attempts, correct sources only with a separate freeze and fresh
namespace, and never overwrite completed records. V2 does not make a global
confirmation decision. Numerical flags, candidate optimum/precision failures,
boundary classification, reference nonoptimality and imbalanced inference must
be assessed before proposing a separately frozen confirmatory family/budget.
Preserve all gates and failures even if they make that proposal premature.
