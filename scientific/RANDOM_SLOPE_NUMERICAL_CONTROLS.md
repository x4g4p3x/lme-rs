# Selected numerical-controls and derivative investigation

16 October 2026. **Post-study diagnostic evidence, not pilot replacement or
calibration validation.** Engine commit `580b270edfcfd8f544818f3ff8171aacbd30d099`
has the same native/source content as the separately archived boundary-fix build.
The [machine-readable record](RANDOM_SLOPE_NUMERICAL_CONTROLS.json) identifies the
isolated binary, original inputs, controls, all six outcomes and dense derivatives.
Every v1 input, fit, gate and decision remains unchanged.

## Prespecified selected-case experiment

Before refitting, a new receipt selected the previously identified boundary case
`zero_slope-G8-m5-v0-rundefined-beta0.5`, replicate 17, and inference case
`imbalanced-G8-mcycle3-5-10-v0.25-r0-beta0`, replicate 18. Both were run at
tolerances 1e-8, 1e-12 and 1e-14, with 5,000 iterations, default starts and no
automatic retry. This selected experiment follows the earlier 42-case precision
diagnostic; none of its results is included in pilot estimates.

| Control | Boundary case | Imbalanced Satterthwaite df | Imbalanced p |
|:---|:---|---:|---:|
| 1e-8 | Reported full rank; oracle rank one | 7.520740 | .444839885 |
| 1e-12 | Correctly singular; converged | 7.576340 | .444664445 |
| 1e-14 | Budget exhausted; adjusted inference unavailable | 7.576413 | .444664002 |
| Retained R | Singular | 7.576573 | .444663571 |

Both cases pass the unchanged relevant numerical gates at **1e-12**. The 1e-14
failure remains an unsuccessful fit even though its objective is near optimal.
These observations support prospective 1e-12/5,000 controls; they do not change
the public default, justify a hidden fallback or establish universal convergence.

## Independent analytic derivative check

The [dense checker](../scripts/check_lmm_satterthwaite_dense.py) computes direct GLS
at saved parameter values, without calling a fitting routine. For physical
covariance parameters, it analytically differentiates the REML criterion and the
fixed-slope variance using the residual projection matrix. Transforming into the
library's relative Cholesky coordinates includes both `J' H J` and the score times
the transformation's second derivatives. Omitting the latter is incorrect away
from stationarity. It does not add jitter or silently substitute a pseudoinverse.

At the imbalanced saved 1e-8 point, analytic Satterthwaite df is **7.520671** versus
library **7.520740**. At the 1e-12 point it is **7.576266** versus library
**7.576340**. Thus the earlier material discrepancy mainly reflects fit-point and
curvature sensitivity; finite-difference remainder is much smaller and meets the
existing engineering gates. This does not prove exact derivative agreement or
inference validity at covariance boundaries. Nonconverged saved points remain
explicitly labelled unevaluable for inferential claims.

Six independent controls check physical score/Hessian/variance derivatives against
central differences, nonstationary transformation curvature, exact rational OLS
limits, invalid-design/variance rejection and evidence integrity. They passed with
CPython 3.11.15 and NumPy 2.3.5. The internal reviewer also independently checked
the selected input identities and the two-case interpretation. This is AI review,
not human statistical peer review.

## Decision and next evidence

Use **1e-12**, maximum **5,000** iterations in the separately frozen
[v2 pilot](../docs/LMM_RANDOM_SLOPE_PILOT_V2_PROTOCOL.md), with unchanged comparison
and rank thresholds, prospective balanced optimum/precision/rank checks for both
engines, and a disjoint protocol seed namespace. Preserve every failure and R
nonoptimality flag. A fresh pilot is needed to assess these controls beyond
selected historical counterexamples; it remains insufficient for confirmation.

Raw receipts/results and source snapshots are local under
`benchmark-results/scientific/random-slope-pilot/20261016-controls-selected/`.
The original selected runner snapshot precedes a later formatting-only line wrap;
its receipt hash resolves to that archived snapshot. The dense check records its
own exact runtime and input/result/source hashes. The JSON's artifact paths are
repository relative; no raw environment paths are published here.
