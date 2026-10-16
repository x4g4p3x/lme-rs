# Random-slope pilot v1: completed, with numerical blockers

16 October 2026. All planned steps were executed under the
[frozen pilot protocol](../docs/LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md) after the
[internal AI implementation review](../docs/LMM_RANDOM_SLOPE_IMPLEMENTATION_REVIEW.md).
The harness controls passed, but the study exposed nonoptimal fitted solutions.
**Do not proceed to confirmatory calibration on the strength of convergence flags.**

## Scope and identity

The candidate was an isolated CPython 3.11.15 Windows wheel of **0.2.8.dev0**,
built from engine commit `1c5057e3270b4010cac2958e5bf25c8826335561`, with
Polars 2.0.0. Its native SHA-256 was
`15daa3fc55c9b2db075a620daa736262867526a5729889622d2f6db956f2cdb0`.
This is a development study build, not the published 0.2.7 package or a new release.
The [freeze descriptor](RANDOM_SLOPE_PILOT_V1_FREEZE.json) identifies engine,
harness, protocol, review, dependency, wheel and native bytes before observations.

Models were Gaussian `y ~ x + (1 + x | group)`, REML, under the 94 fixed designs
and explicit algorithm-specific controls. Candidate tolerance was 1e-8 with
5,000 outer iterations; R used bobyqa with maxfun=100000. All 61 R packages and
51 locked namespaces were verified. Inputs were synthetic; no private user data
was used. The 12-dataset diagnostic smoke had separate seeds and is excluded
from every pilot estimate.

## Observed results

The [case/method ledger](RANDOM_SLOPE_PILOT_V1.json) preserves all 658 rows,
uncertainty, failures, fit diagnostics, per-case discrepancy counts and source
anchors. A separate post-study count audit agreed with candidate/reference raw
rejection, coverage and paired-disagreement counts. No outcomes were substituted.

| Check | Observed outcome |
|:---|:---|
| Diagnostic smoke | 12/12 candidate and R fits converged; all methods evaluable; every predefined numeric gate passed |
| Pilot | 94 designs x 20 = 1,880 paired datasets |
| Reported fit convergence | 1,880/1,880 in each engine |
| Recorded inference failures | Zero in all three methods, in each engine |
| Common eigen-diagnostic singular fits | Candidate 658; R 552 |
| Any predefined numerical discrepancy | 166/1,880 pairs across 76 designs |
| Rejection disagreements | Normal 2; Satterthwaite 2; KR 1 |
| Coverage disagreements | Normal 2; Satterthwaite 1; KR 0 |

Some flags concern small covariance entries; others reflect substantial criterion
gaps. Neither implementation agreement nor a reported convergence flag alone
establishes that optimization reached the constrained REML optimum.

The 20-replicate casewise estimates are exploratory. In the 36 future primary
null cases, candidate rejection rates ranged from 0% to 25% for normal inference
and 0% to 20% for each adjusted method; even the known-variance control ranged
from 0% to 15%. These noisy ranges are not calibration decisions. A case with
zero failures still has a two-sided 95% exact upper failure bound of about 16.8%.
Read the casewise intervals, all-attempt failure envelopes and MCSE in the ledger.
No global scope gate or confirmatory claim was evaluated.

## Post-study independent optimum diagnostic

To investigate the unexpected discrepancies, a separate AI derivation established
the global constrained optimum for balanced designs. This analysis was developed
after the pilot and is explicitly diagnostic. It reads retained responses and
does not refit or change either engine's stored outcomes.

For common centered `Z=[1,x]`, let `D=diag(m,Sxx)`, group OLS coefficients be
`c_i`, `u_i=sqrt(D)*c_i`, and `B=sum((u_i-mean(u))*(u_i-mean(u))')/(G-1)`.
With `q=G-1`, `w=G(m-2)`, within-group OLS residual sum SSW and eigenvalues
`lambda1<=lambda2` of B, write `s=sigma2` and `v_j>=s` for the two transformed
covariance eigenvalues. The criterion is:

```text
C + w*log(s) + SSW/s + q*sum_j(log(v_j) + lambda_j/v_j)
C = (N-2)*log(2*pi) + log|X'X|; X'X = G*D
```

Start at `s=SSW/w`. If this exceeds lambda1, pool
`s=(SSW+q*lambda1)/(w+q)`; if that exceeds lambda2, pool both directions:
`s=(SSW+q*(lambda1+lambda2))/(w+2*q)`. Set `v_j=max(s,lambda_j)` and reconstruct
`Sigma_hat=invsqrt(D)*Q*diag(v_j-s)*Q'*invsqrt(D)`.
This requires G>1,m>2,positive SSW; m=2 cannot identify both unrestricted random
covariance and residual variance this way. It does not apply to the imbalanced cases.

[lmm_random_slope_optimum.py](../scripts/lmm_random_slope_optimum.py) implements
this independent spectral calculation. Rotated rational fixtures verify all
three active sets against exact dense GLS covariance and a 70-digit likelihood;
the exact residual variances are 2, 18/11 and 37/28. This validates off-diagonal
reconstruction and the criterion constant, beyond agreement with fitted packages.

Applying the unchanged prespecified criterion gate to the 1,840 balanced records:

| Engine | Reported-converged fits above the independent optimum | Largest criterion gap |
|:---|---:|---:|
| Candidate | 119/1,840 | 3.9895353286 |
| R | 7/1,840 | 3.5742911866 |

Neither engine produced a criterion below the oracle beyond that gate. The
[selected counterexamples](RANDOM_SLOPE_OPTIMUM_COUNTEREXAMPLES_V1.json) include
complete designs/responses, exact input provenance, original fits and independent
expected solutions. Selection was post-study; these are not extra pilot draws.

The largest candidate example, G50,m5,v1=0.25,r=0.5,beta1=0,replicate19, returned
theta=(0.3903989894,0.2807247723,0) and criterion 800.0530898015, while the global
criterion is 796.0635544729. R agrees with that optimum on this record. Conversely,
the largest R discrepancy has criterion 365.4728646559 versus 361.8985734694;
the candidate agrees with the latter. Thus R agreement cannot serve as the sole
correctness oracle. These are optimization/convergence blockers under the studied
controls; this pilot does not identify the full set of affected APIs or settings.

## Cost and resource observations

Four workers with one MKL/OpenMP/OpenBLAS/Polars thread each took 44.19 seconds
for candidate fitting/inference and 66.97 seconds for R references, including
worker startup. These are study cost observations, not a fair speed benchmark.

The frozen parent-side memory counters observed small Windows launcher programs,
not the interpreter descendants, so their recorded 5-8 MiB values are not usable
worker memory estimates. A separately preserved process-tree observation captured
R interpreter peaks up to about 425 MiB. Candidate workers had already exited
before that observer captured them; a named post-pilot resource-only replay of
one largest-design record measured about 114 MiB in the native Python process.
That replay is not an independent observation and is excluded from all estimates.
These sampled/process-specific values are not aggregate RSS or production limits.

## Integrity, availability and reproduction

| Artifact | SHA-256 |
|:---|:---|
| Original freeze | `5fa9d7c265e47177090f55d0747d2848026ccc10e557682c967787607d94395d` |
| Original pilot manifest | `4611cdfd18967b4f5e7eeb8b43199c2a5520eb71604ad5a1d0704c641ef2f54e` |
| Original pilot summary | `98580c2b5877e16e04c6a337881e20338a1b8fb5f3dac499491d58cbdd946c44` |
| Post-study optimum diagnostics | `ded1b5b443e408283b706a573e574dc105a522004803091c146e9618c7452004` |

Raw smoke/pilot records, the source freeze and study wheel are local under
`benchmark-results/scientific/random-slope-pilot/20261016-v1-*`. A separate portable
bundle is prepared with sources, wheel/dependency pins, raw results, summaries,
post-study diagnostics, resource-observation corrections and license notices.
The [bundle descriptor](RANDOM_SLOPE_PILOT_V1_BUNDLE.json) records its checksum
and local availability: 30,055,290 bytes, 524 entries, ZIP CRC checks passed,
SHA-256 `4db098bc020498c58c03b89157d34f8de5572289b3da29bb011faeddda145916`.
The new pilot archive has not been publicly uploaded. The compact ledger and counterexamples above
are repository artifacts. Do not confuse them with public availability of all
raw pilot records.

Use the archived frozen sources and exact candidate environment; never silently
replace this development wheel with a later build sharing its version string:

```text
python -m unittest discover -s scripts/tests -p "test_lmm_random_slope*.py" -v
python scripts/lmm_random_slope.py generate NEW_RUN --freeze FROZEN_DIRECTORY --namespace smoke
python scripts/lmm_random_slope.py run NEW_RUN --candidate-python ISOLATED_PYTHON --rscript RSCRIPT --workers 4
python scripts/lmm_random_slope.py analyse NEW_RUN
python scripts/lmm_random_slope_optimum.py ORIGINAL_PILOT --output NEW_DIAGNOSTIC_JSON
python scripts/report_lmm_random_slope.py ORIGINAL_PILOT --optimum NEW_DIAGNOSTIC_JSON --output NEW_LEDGER_JSON
```

For the full exploratory pilot, generate a fresh destination using `--namespace
pilot`. Restoring the archived environment means installing the hashed wheel
with `--no-deps` plus its pinned requirements and verifying the R lock; this is
not a new public package release. Frozen source checks reject changed active
files, altered input inventories and misaligned records. Existing outputs cannot
be overwritten by these commands.

The earlier 0.2.7 random-intercept study is unchanged. Its
[full raw archive](https://github.com/x4g4p3x/lme-rs/releases/download/v0.2.7/lmm-calibration-confirmatory-v3-0.2.7.zip)
and [checksum](https://github.com/x4g4p3x/lme-rs/releases/download/v0.2.7/lmm-calibration-confirmatory-v3-0.2.7.sha256)
are now public, with a fresh full download and anonymous HTTP retrieval verified.

## Implementation checks and limits

`task lint:scripts`, required R/Julia comparison formatting, `task preflight`
and `task docs:check` passed. Documentation validation includes the full
74-test script suite, six doctests, example compilation and API documentation.
The independent count audit passed against all paired raw records; final source,
input, freeze, reporter and diagnostic identities were rechecked. These checks
verify the harness/evidence, not the statistical correctness of nonoptimal fits.
No Rust or Python binding implementation was changed by this task. No new full
hosted OS/Python matrix, production-load cases or independent human statistical
review was run. The development study wheel is not a public package release.

## Next scientific gate

First reproduce and fix the candidate's nonoptimal convergence using the retained
counterexamples and independent optimum. Treat the R discrepancy separately.
Preserve v1, then freeze a new candidate and disjoint-seed pilot before selecting
any full-confirmation budget. More repetitions of this unchanged procedure would
not resolve the numerical blocker. New method/calibration claims remain unsupported.
