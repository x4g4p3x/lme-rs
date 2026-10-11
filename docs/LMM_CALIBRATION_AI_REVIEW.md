# AI methodological review and responses

Date: 11 October 2026. Scope: protocol v1 and frozen 0.2.7 numerical results.
A separately tasked Codex agent provided a read-only critique before pilot
execution; the implementing agent recorded the responses below in protocol v2.
This is internal AI review within the same system, not independent human peer
review, certification or a guarantee of correctness.

| Objection | Resolution before pilot |
|:---|:---|
| Null rejection and interval noncoverage duplicate the same claim | Define 24 primary cases times two candidate adjusted methods = 48 claims; check test/interval pairing |
| An all-case decision needs an explicit simultaneous interval | Bonferroni Clopper-Pearson governs a future global decision; pointwise Wilson is descriptive |
| 10,000/case may leave a global conclusion inconclusive after correction | Retain it as an initial budget, disclose that limitation, defer final confirmatory sample-size planning; pilot cannot pass the acceptance gates |
| Evaluable-only success can conceal selective failures | Nonconverged attempts are unevaluable; every attempt remains recorded; use conservative uncertainty envelopes over failure identification bounds |
| Two fitted implementations can agree on a shared mistake | Add true-variance GLS Gaussian controls and balanced information identities; archive exact response arrays |
| KR intervals can accidentally use unadjusted SE | Recover adjusted scalar variance from a unit-displacement public contrast, verify observed F, and independently use R vcovAdj with scaling one |
| Mean SE alone is not variance calibration | Report sqrt(mean(SE squared))/empirical SD, mean width and noncoverage tails; use 1,000 paired bootstrap resamples with separate analysis seed |
| Pilot observations must remain separate from confirmation | Executable runner supports only a bounded pilot namespace; no retries or promotion to confirmatory evidence |

The reviewer additionally proposed balanced interior REML df diagnostics:
G*(m-1)-1 for within predictors and G-2 for between predictors. Fitted boundary
solutions require separate treatment even when generating rho is positive.

The critique follows [Morris, White and Crowther's simulation-study guidance](https://doi.org/10.1002/sim.8086).
The choice to distinguish approximate Wilson intervals from conservative exact
binomial intervals is informed by [Brown, Cai and DasGupta](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.pdf).
The scalar KR test/variance consistency check is supported by the rank-one
derivation in [Morris and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC5901423/).

Remaining work before confirmation: final sample-size/decision-power planning,
a separately frozen confirmatory runner and seed namespace, a restorable R
dependency environment, and assessment of pilot failures. The review did not
inspect all library mathematics or substitute for broader GLMM/NLMM validation.
