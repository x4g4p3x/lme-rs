# Confirmatory LMM calibration study: published 0.2.7

Completed 13 October 2026 under [protocol v3](../docs/LMM_CALIBRATION_PROTOCOL.md),
following the [exploratory pilot](CALIBRATION_PILOT_0.2.7.md),
[AI methodological critique](../docs/LMM_CALIBRATION_AI_REVIEW.md),
[pilot implementation review](../docs/LMM_CALIBRATION_IMPLEMENTATION_REVIEW.md),
and [confirmatory harness review](../docs/LMM_CALIBRATION_CONFIRMATORY_REVIEW.md).
This study evaluates the frozen published **0.2.7** engine on a disjoint
held-out seed namespace (`lmm-calibration-v3|confirmatory|...`). Reviews were
performed by separately tasked read-only AI agents; no independent human peer
review has occurred. Scope: Gaussian random-intercept models with one continuous
fixed slope (`y ~ x + (1 | group)`), fixed designs, REML, independent Gaussian
errors, no weights or offsets.

## What ran and what was observed

All 80 prespecified designs received **20,000 simulated responses** per case:
**1,600,000 datasets** in total (480,000 primary null datasets across the 24
balanced interior null cases, 240,000 boundary null datasets across the 12
balanced `rho = 0` cases, 720,000 alternative datasets across the 36 balanced
`beta1 = 0.5` cases, and 160,000 imbalanced null datasets across the eight
`2-4-8` cases). Every candidate fit (`normal`, `satterthwaite`,
`kenward_roger`) and the independent known-true-variance Gaussian GLS control
ran on all 1,600,000 datasets, with exact reference distribution quantiles and
tail probabilities (`qnorm`, `pnorm`, `qt`, `pf`) and exact Clopper-Pearson
binomial bounds (`qbeta`) evaluated independently in R 4.6.1.

In addition, the prespecified paired reference audit stratum—the first **1,000
confirmatory replicates per case** (`0 <= replicate < 1000`, **80,000 paired
datasets**, 10x the pilot sample size)—was fitted in R (`lmerTest::lmer`,
`contest1D`, `pbkrtest::vcovAdj`, `pbkrtest::KRmodcomp`) under the locked
R 4.6.1 dependency environment ([scientific/renv.lock](renv.lock)).

- **Candidate convergence and inference:** The isolated PyPI 0.2.7 engine
  converged on **1,600,000 / 1,600,000** datasets (`0` nonconvergences, `0`
  exceptions, `0` unknown convergence states, `0` invalid df, and `0`
  test/interval or unit-displacement contrast inconsistencies). Exactly
  **327,892 / 1,600,000** fits (`20.49%`) were classified as singular at
  `|theta| < 1e-4` and retained as evaluable per the protocol.
- **R reference convergence and paired agreement:** On the 80,000-dataset paired
  reference audit stratum, R converged on **79,994 / 80,000** datasets (`6`
  `bobyqa` nonconvergences across six distinct cases: replicate 607 of
  `balanced-G6-m10-rho0-between-beta0.5`, replicate 977 of
  `balanced-G6-m10-rho0.2-between-beta0`, replicate 333 of
  `balanced-G12-m3-rho0-within-beta0.5`, replicate 16 of
  `balanced-G12-m10-rho0-between-beta0`, replicate 636 of
  `balanced-G30-m3-rho0-within-beta0.5`, and replicate 173 of
  `imbalanced-G12-m2-4-8-rho0.2-between-beta0`; `0` inference exceptions among
  converged fits; **16,429 / 80,000** fits classified as singular at
  `tol = 1e-4`).
- **Exact decision parity on common-evaluable fits:** Across all **79,994
  common-evaluable paired datasets**, the candidate 0.2.7 engine and R
  (`lme4`/`lmerTest`/`pbkrtest`) made **100% identical binary rejection and
  95% confidence-interval coverage decisions** for normal-Wald, Satterthwaite,
  and Kenward-Roger separately (`0` decision disagreements in all 80 cases).
  The largest absolute slope difference was `1.13e-7`; the largest absolute
  standard-error difference was `1.83e-7`.
- **Balanced interior df identities:** Among nonsingular balanced fits,
  Kenward-Roger denominator degrees of freedom matched the exact analytical
  identities (`G*(m-1)-1` within groups, `G-2` between groups) to within
  `2.68e-11` across all 1,600,000 candidate fits and `6.99e-12` across the
  R reference fits. Candidate finite-difference Satterthwaite df had a maximum
  departure of `0.188` (`0.0432` in R).

## Confirmatory decisions across the 48 primary claims

The joint primary family comprises the 24 balanced interior null cases
(`rho in {0.2, 0.7}`, `beta1 = 0`) times the two candidate adjusted methods
(`candidate:satterthwaite` and `candidate:kenward_roger`), evaluated against the
prespecified `[0.04, 0.06]` rejection band (and complementary `[0.94, 0.96]`
coverage band) using simultaneous 95% Bonferroni Clopper-Pearson failure
envelopes (`per-claim alpha = 0.05 / 48`).

| Candidate method | Primary cases | Supported | Inconclusive | Contradicted | All-24-case primary scope gate |
|:---|---:|---:|---:|---:|:---|
| `candidate:satterthwaite` | 24 | 23 | 1 | 0 | `false` (`balanced-G6-m3-rho0.2-between-beta0` inconclusive) |
| `candidate:kenward_roger` | 24 | 21 | 2 | 1 | `false` (1 contradicted, 2 inconclusive in small-`G`, `rho=0.2`, between-group cases) |

- **Within-group predictors (12 / 12 cases):** Every primary within-group case
  (`G in {6, 12, 30}`, `m in {3, 10}`, `rho in {0.2, 0.7}`) is **supported**
  for both Satterthwaite (rejection rates `4.91%` to `5.17%`) and Kenward-Roger
  (`4.91%` to `5.17%`).
- **Higher-ICC between-group predictors (`rho = 0.7`, 6 / 6 cases):** Every
  `rho = 0.7` between-group case (`G in {6, 12, 30}`, `m in {3, 10}`) is
  **supported** for both Satterthwaite (`4.82%` to `5.24%`) and Kenward-Roger
  (`4.70%` to `5.24%`).
- **Moderate/large-cluster low-ICC between-group predictors (`G = 30` or
  `G = 12, m = 10`, `rho = 0.2`, 3 / 3 cases):** All three cases are
  **supported** for both Satterthwaite (`5.01%` to `5.05%`) and Kenward-Roger
  (`4.92%` to `5.04%`).
- **Small-cluster, low-ICC between-group predictors (`rho = 0.2`, between-group,
  `(G, m) in {(6, 3), (6, 10), (12, 3)}`, 3 cases):**
  - For `balanced-G6-m3-rho0.2-between-beta0` (`31.25%` singular fits),
    Satterthwaite rejection is `4.28%` (simultaneous 95% CP `[3.82%, 4.76%]`,
    **inconclusive** at the 4% lower edge) and Kenward-Roger rejection is
    `2.10%` (`[1.79%, 2.46%]`, **contradicted** as conservative below 4%).
  - For `balanced-G6-m10-rho0.2-between-beta0` (`11.14%` singular fits),
    Satterthwaite is **supported** (`5.24%`, `[4.74%, 5.78%]`), while
    Kenward-Roger is `3.69%` (`[3.27%, 4.15%]`, **inconclusive**).
  - For `balanced-G12-m3-rho0.2-between-beta0` (`18.00%` singular fits),
    Satterthwaite is **supported** (`4.66%`, `[4.19%, 5.17%]`), while
    Kenward-Roger is `4.11%` (`[3.66%, 4.59%]`, **inconclusive**).
- **Normal-Wald comparator:** Unadjusted normal-Wald inference exceeds the 6%
  upper band across small- and medium-cluster between-group designs, reaching
  `12.19%` (`[11.49%, 12.91%]`) at `G = 6, m = 10, rho = 0.7` and `12.17%`
  (`[11.48%, 12.90%]`) at `G = 6, m = 3, rho = 0.7`.

### Why small-cluster, low-ICC between-group designs are conservative under REML

In all three non-supported cases (`G=6, m=3`, `G=6, m=10`, and `G=12, m=3` at
`rho = 0.2`, between-group predictor), `lme-rs` and R (`lme4`/`lmerTest`/`pbkrtest`)
agreed on **100% of all paired rejection and coverage decisions**. The observed
conservatism is therefore an intrinsic finite-sample property of non-negative
REML estimation (`tau2 >= 0`), not an implementation bug in `lme-rs`:

| Case (`rho = 0.2`, between) | Singular rate | Interior (`tau2 > 0`) Satterthwaite & KR rej (mean SE, df) | Singular (`tau2 = 0`) Satterthwaite rej (mean SE, df) | Singular (`tau2 = 0`) KR rej (mean SE, df) | Overall Satterthwaite | Overall KR |
|:---|---:|:---|:---|:---|---:|---:|
| `G=6, m=3` | 31.25% (`6,251 / 20,000`) | 1.17% (`161 / 13,749`, SE `0.3045`, df `4.00`) | 11.10% (`694 / 6,251`, SE `0.2152`, df `16.00`) | 4.16% (`260 / 6,251`, SE `0.2152`, df `4.00`) | 4.28% | 2.10% |
| `G=6, m=10` | 11.14% (`2,228 / 20,000`) | 2.34% (`415 / 17,772`, SE `0.2174`, df `4.00`) | 28.46% (`634 / 2,228`, SE `0.1166`, df `58.00`) | 14.50% (`323 / 2,228`, SE `0.1166`, df `4.00`) | 5.24% | 3.69% |
| `G=12, m=3` | 18.00% (`3,600 / 20,000`) | 3.16% (`518 / 16,400`, SE `0.2044`, df `10.00`) | 11.50% (`414 / 3,600`, SE `0.1560`, df `34.00`) | 8.44% (`304 / 3,600`, SE `0.1560`, df `10.00`) | 4.66% | 4.11% |

For a balanced between-group predictor, an unconstrained ANOVA F-test uses the
sample between-group mean square `MSB` on `G - 2` df and has exact 5% size.
Under REML with `tau2 >= 0`, the effective between-group variance factor is
truncated below at `MSW` (`max(MSB, MSW)`):

1. Conditional on an **interior fit** (`MSB > MSW`, `tau2 > 0`), `MSB` is
   selected to be larger than `MSW`, biasing the slope standard error upward
   and reducing the conditional interior rejection rate (`1.17%` at `G=6, m=3`).
2. Conditional on a **boundary/singular fit** (`MSB <= MSW`, `tau2 = 0`), the
   fitted covariance collapses to pooled OLS (`SE` drops from `0.3045` to
   `0.2152` at `G=6, m=3`), but the two df approximations behave differently in
   both `lme-rs` and R:
   - **Kenward-Roger** evaluates the expected information derivatives with
     respect to the variance component `tau2`, which remain nonzero at
     `tau2 = 0`, and therefore retains `df = G - 2` (`4.00` at `G=6`, `10.00`
     at `G=12`) on singular fits. The wider `t_{0.975, G-2}` critical value
     keeps singular-fit rejection moderate (`4.16%` at `G=6, m=3`), so the
     unconditional average across interior and singular fits is conservative
     (`2.10%` at `G=6, m=3`, `3.69%` at `G=6, m=10`, `4.11%` at `G=12, m=3`).
   - **Satterthwaite** differentiates with respect to the Cholesky factor
     `theta` (where `tau2 = sigma2 * theta^2`), whose gradient vanishes at
     `theta = 0` (and in `lmerTest` falls back to residual df at the boundary),
     yielding `df = G*m - 2` (`16` at `G=6, m=3`, `58` at `G=6, m=10`, `34` at
     `G=12, m=3`) on singular fits. The higher singular-stratum rejection rate
     partially offsets the conservative interior stratum, keeping unconditional
     Satterthwaite rejection inside `[4%, 6%]` for 23 of the 24 primary cases
     and at `4.28%` (`[3.82%, 4.76%]`) for `G=6, m=3`.

## Primary null cases (`n = 20,000` per case)

Counts below are out of 20,000 attempted datasets (`0` failures in all candidate
and GLS control runs). Coverage at the null is the exact complement of
rejection (`20,000 - count`). Simultaneous 95% intervals are exact Bonferroni
Clopper-Pearson bounds (`alpha = 0.05 / 48`), expressed as percentages.

| G | m | rho | Predictor | Normal count (%) | Satterthwaite count (%) | Satterthwaite sim 95%, % (decision) | KR count (%) | KR sim 95%, % (decision) | GLS control count (%) |
|---:|---:|---:|:---|---:|---:|:---|---:|:---|---:|
| 6 | 3 | 0.2 | within | 1528 (7.64%) | 1023 (5.12%) | 4.62–5.65 (supported) | 991 (4.96%) | 4.47–5.48 (supported) | 931 (4.66%) |
| 6 | 3 | 0.2 | between | 1677 (8.38%) | 855 (4.28%) | 3.82–4.76 (inconclusive) | 421 (2.10%) | 1.79–2.46 (contradicted) | 1002 (5.01%) |
| 6 | 3 | 0.7 | within | 1544 (7.72%) | 1025 (5.12%) | 4.63–5.66 (supported) | 1023 (5.12%) | 4.62–5.65 (supported) | 996 (4.98%) |
| 6 | 3 | 0.7 | between | 2434 (12.17%) | 1019 (5.10%) | 4.60–5.62 (supported) | 939 (4.70%) | 4.22–5.21 (supported) | 964 (4.82%) |
| 6 | 10 | 0.2 | within | 1125 (5.62%) | 1034 (5.17%) | 4.67–5.70 (supported) | 1034 (5.17%) | 4.67–5.70 (supported) | 1028 (5.14%) |
| 6 | 10 | 0.2 | between | 2168 (10.84%) | 1049 (5.24%) | 4.74–5.78 (supported) | 738 (3.69%) | 3.27–4.15 (inconclusive) | 1028 (5.14%) |
| 6 | 10 | 0.7 | within | 1086 (5.43%) | 994 (4.97%) | 4.48–5.49 (supported) | 994 (4.97%) | 4.48–5.49 (supported) | 984 (4.92%) |
| 6 | 10 | 0.7 | between | 2437 (12.19%) | 964 (4.82%) | 4.34–5.34 (supported) | 954 (4.77%) | 4.29–5.28 (supported) | 972 (4.86%) |
| 12 | 3 | 0.2 | within | 1278 (6.39%) | 1009 (5.04%) | 4.55–5.57 (supported) | 995 (4.98%) | 4.49–5.50 (supported) | 1010 (5.05%) |
| 12 | 3 | 0.2 | between | 1365 (6.83%) | 932 (4.66%) | 4.19–5.17 (supported) | 822 (4.11%) | 3.66–4.59 (inconclusive) | 976 (4.88%) |
| 12 | 3 | 0.7 | within | 1275 (6.38%) | 1012 (5.06%) | 4.57–5.59 (supported) | 1012 (5.06%) | 4.57–5.59 (supported) | 980 (4.90%) |
| 12 | 3 | 0.7 | between | 1603 (8.02%) | 1049 (5.24%) | 4.74–5.78 (supported) | 1049 (5.24%) | 4.74–5.78 (supported) | 1012 (5.06%) |
| 12 | 10 | 0.2 | within | 1068 (5.34%) | 1024 (5.12%) | 4.62–5.65 (supported) | 1024 (5.12%) | 4.62–5.65 (supported) | 1022 (5.11%) |
| 12 | 10 | 0.2 | between | 1544 (7.72%) | 1002 (5.01%) | 4.52–5.54 (supported) | 983 (4.92%) | 4.43–5.44 (supported) | 978 (4.89%) |
| 12 | 10 | 0.7 | within | 1040 (5.20%) | 981 (4.91%) | 4.42–5.43 (supported) | 981 (4.91%) | 4.42–5.43 (supported) | 1007 (5.04%) |
| 12 | 10 | 0.7 | between | 1560 (7.80%) | 997 (4.98%) | 4.49–5.51 (supported) | 997 (4.98%) | 4.49–5.51 (supported) | 1032 (5.16%) |
| 30 | 3 | 0.2 | within | 1090 (5.45%) | 994 (4.97%) | 4.48–5.49 (supported) | 993 (4.96%) | 4.48–5.49 (supported) | 970 (4.85%) |
| 30 | 3 | 0.2 | between | 1194 (5.97%) | 1011 (5.05%) | 4.56–5.58 (supported) | 1000 (5.00%) | 4.51–5.52 (supported) | 1023 (5.12%) |
| 30 | 3 | 0.7 | within | 1121 (5.61%) | 1025 (5.12%) | 4.63–5.66 (supported) | 1025 (5.12%) | 4.63–5.66 (supported) | 979 (4.90%) |
| 30 | 3 | 0.7 | between | 1137 (5.68%) | 980 (4.90%) | 4.41–5.42 (supported) | 980 (4.90%) | 4.41–5.42 (supported) | 985 (4.92%) |
| 30 | 10 | 0.2 | within | 1041 (5.21%) | 1014 (5.07%) | 4.58–5.60 (supported) | 1014 (5.07%) | 4.58–5.60 (supported) | 997 (4.98%) |
| 30 | 10 | 0.2 | between | 1213 (6.07%) | 1009 (5.04%) | 4.55–5.57 (supported) | 1008 (5.04%) | 4.55–5.57 (supported) | 1041 (5.21%) |
| 30 | 10 | 0.7 | within | 1053 (5.27%) | 1028 (5.14%) | 4.64–5.67 (supported) | 1028 (5.14%) | 4.64–5.67 (supported) | 1044 (5.22%) |
| 30 | 10 | 0.7 | between | 1219 (6.09%) | 1004 (5.02%) | 4.53–5.55 (supported) | 1004 (5.02%) | 4.53–5.55 (supported) | 987 (4.93%) |

## Secondary strata (`n = 20,000` per case)

- **Boundary null stratum (`rho = 0, beta1 = 0`, 12 balanced cases):**
  Singularity rates ranged from `51.24%` to `58.14%` (matching the theoretical
  ~50% point mass at zero for a constrained variance component at the boundary).
  For within-group predictors, rejection rates were `4.86%` to `6.21%` for
  Satterthwaite and `4.86%` to `5.82%` for Kenward-Roger. For between-group
  predictors, positive variance estimates (`tau2 > 0` on ~45% of replicates)
  inflate the estimated between-group SE relative to the true `rho = 0`
  variance, making both Satterthwaite (`3.00%` to `4.00%`) and Kenward-Roger
  (`0.54%` to `3.71%`) conservative.
- **Imbalanced null stratum (`sizes = [2, 4, 8]`, `beta1 = 0`, 8 cases):**
  Satterthwaite rejection rates lay between `4.80%` and `5.50%` across all eight
  imbalanced designs (with simultaneous 95% envelopes inside `[4%, 6%]` for
  seven of eight cases and `4.98%–6.04%` on `imbalanced-G6-m2-4-8-rho0.2-between-beta0`).
  Kenward-Roger matched `[4%, 6%]` on all four within-group imbalanced cases
  (`4.81%` to `5.26%`) and on `imbalanced-G12-m2-4-8-rho0.7-between-beta0`
  (`4.88%`), while exhibiting conservative rejection on small-cluster
  imbalanced between-group designs (`0.53%` at `G=6, rho=0.2`, `3.98%` at
  `G=6, rho=0.7`, and `3.94%` at `G=12, rho=0.2`).
- **Alternative stratum (`beta1 = 0.5`, 36 balanced cases) and SE calibration:**
  Across all 24 primary null cases, maximum absolute slope bias was `0.00291`
  (MCSE `~0.0014` to `0.0029`), and the SE calibration ratio
  `sqrt(mean(SE^2)) / empirical_sd` ranged from `0.9860` to `1.0382` (`0.9919`
  to `1.0086` for the known-variance GLS control). Across the 36 `beta1 = 0.5`
  alternative cases, two-sided 95% interval coverage ranged from `93.98%` to
  `97.03%` for Satterthwaite, `94.38%` to `99.53%` for Kenward-Roger, `87.56%`
  to `95.87%` for normal-Wald, and `94.67%` to `95.25%` for the GLS control;
  rejection power ranged from `18.73%` to `100.00%` (Satterthwaite) and
  `18.38%` to `100.00%` (Kenward-Roger).

## Evidence and reproduction

- Run ID: `20261013T210450Z-2ce9f0b4` (`benchmark-results/scientific/calibration-confirmatory/0.2.7/20261013T210450Z-2ce9f0b4`).
- Candidate release Git SHA: `0b4760fcc5a7558a60ec738401f326a793cb6b1d`; native module SHA-256: `fd754e69381368bf8ccd1105f71aca3fdbca740874ad5bd9c316262699edac9f`.
- Environment: Windows 10 x64 (`10.0.26300`), CPython 3.11.15, polars 2.0.0, R 4.6.1, lme4 2.0.1, lmerTest 3.2.1, pbkrtest 0.5.5, jsonlite 2.0.0, 12 bounded workers with `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `POLARS_MAX_THREADS=1`.
- Wall-clock execution times on 12 workers: candidate fitting and adjusted contrast evaluation took `643.55s` (`7,034.77` total worker CPU-seconds across 1,600,000 fits); R distribution evaluation across 1,600,000 fits plus 80,000 paired reference fits took `690.26s` (`7,226.46` total worker CPU-seconds); total pipeline wall time including input generation, environment creation, and 1,000-resample bootstrap analysis was `1,887.3s` (`31.5` minutes).
- Artifact SHA-256 digests:
  - `summary.json`: `b6f9a8dae787591da4c157ffcc7a9714014e6c8c67f5e3b3edb78bd4c28c399c`
  - `manifest.json`: `24e2d061c781e5d26d375cda58625fe5a117bb38ed798410ac2b1baa18aab3de`
  - `designs.json`: `3a385f18d3419e0a9f3827a4ebf1bdc95b0cd64ccd9f2627b99a7fa58381341b`
  - Aggregate `inputs_sha256` across 80 `inputs.jsonl.gz` shards: `7178b3ea1d3f9a96ada5bbf91e9b103e22d4b234e8982921160551d851f0f346`
  - `binomial-bounds.json`: `03f2679e0c16466d6664c9bd6956c24a8f1b50436f76f0507ca0d34168eb0871`
  - `candidate-environment.json`: `9c66a41467bb4bc986c43e0d94180ec9c67df4b6be04d4e70dc6cc01f07bb631`
  - `reference-environment.json`: `4774c82765c290ff7cf6e4389828b517f01a11c370b65c5cc096518fb7eb1ca7`
  - `R-session.txt`: `99fb91fc07fc76511fb41fa9b84214b2dd3deca12de6d3540a97ba99ff43466f`
  - Aggregate `case_summaries_sha256`: `6e0002d07978f3c667a8a3d5bd59efea41c5d63775be70d9ef955d9bbff0be0a`
- Frozen source SHA-256 digests verified before generation, candidate fitting, reference evaluation, and analysis:
  - `scripts/lmm_calibration.py`: `ee1530d12dcd4f50b29c3e9763bf5b640da7cf0539767b2d5db3aa7216d9de40`
  - `scripts/ci/lme_ci.py`: `f6b45ba2252536926c5e85b9e29b39517f58853edfebb043df8cabc7c3871aa6`
  - `comparisons/scientific/lmm_calibration.R`: `f4122811f3702e8d281ed7e3f9b90867e92b99767ea6937a9303b550c4463dfd`
  - `comparisons/scientific/lmm_confirmation_planning.R`: `cd543812afdcc4aafae24da04f84cecafb5e224a661166ac5d264b17135778c4`
  - `comparisons/scientific/restore_r_environment.R`: `5a8aab12842bee7933f68117eac83ae6d83e58b61ded31a712d816ab6f940512`
  - `scientific/renv.lock`: `b66975674f03d6431e31e830548f1456030d81cec5e931714d6c3c5cce58a0f2`
  - `docs/LMM_CALIBRATION_PROTOCOL.md`: `4f77721d26df02488c27592adc076bda0b4e825971b65a151eeb9e4279b59980`
  - `docs/LMM_CALIBRATION_CONFIRMATORY_REVIEW.md`: `6555311c1b8fefc137ffdc334fe204a599e94431eea1d77c3413913a8f2a34c5`

To verify or restore the locked R 4.6.1 environment and reproduce the confirmatory
study from the repository root:

```bash
Rscript --vanilla comparisons/scientific/restore_r_environment.R --verify
task scientific:calibration:confirmatory
```

## Remaining boundaries

- Random-slope LMMs, crossed/nested multi-factor random structures, non-Gaussian
  and heteroscedastic misspecification stress tests, GLMM/NLMM calibration,
  broader applied-paper reproductions, and independent human statistical peer
  review remain outside this completed Gaussian random-intercept confirmatory
  milestone.
