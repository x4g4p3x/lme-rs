# Exploratory LMM calibration pilot: published 0.2.7

Completed 11 October 2026 under [protocol v2](../docs/LMM_CALIBRATION_PROTOCOL.md),
with an [AI methodological critique](../docs/LMM_CALIBRATION_AI_REVIEW.md) and
[implementation review](../docs/LMM_CALIBRATION_IMPLEMENTATION_REVIEW.md).
This is an exploratory pilot, not confirmatory calibration or independent human
peer review. Scope: Gaussian random-intercept models with one continuous slope,
fixed designs, REML, independent Gaussian errors, no weights or offsets.

## What ran and what was observed

All 80 prespecified designs received 100 simulated responses: **8,000 datasets**,
with exactly the same rows/responses supplied to each implementation. There are
72 balanced factorial designs and eight imbalance designs; 24 balanced interior
null cases form the future primary scope. Three candidate and three R methods
were evaluated, plus an independent true-variance GLS Gaussian control.

The isolated PyPI 0.2.7 engine converged on **8,000/8,000** fits. R converged on
**7,999/8,000**. Its one failure was replicate 91 of
`balanced-G12-m10-rho0-between-beta0.5`: bobyqa returned convergence code 3
(a trust-region step failed to reduce the objective). It remains recorded and
unevaluable for all R methods, with no retry. There were no recorded inference
exceptions or test/interval inconsistencies among converged fits.

Both engines classified **1,622/8,000** fits as singular at theta < 1e-4. These
finite singular results remained evaluable; singularity alone was not an
exclusion. Every fit had a classifiable singularity flag in this particular run.

On the **7,999 common-evaluable datasets**, candidate and R made identical
rejection and interval-coverage decisions for normal-Wald, Satterthwaite and
Kenward-Roger separately. The largest absolute slope difference was
9.92e-8; the largest absolute SE difference was 1.74e-7. These observations
establish numerical agreement on these sampled cases, not universal correctness.

Balanced fitted-interior df diagnostics also provided independent identities.
Maximum absolute departures were 0.00940 for candidate Satterthwaite and 0.0862
for R Satterthwaite, whose df calculations use numerical derivatives. KR
departures were below 2.22e-11 in both engines. These are descriptive diagnostic
observations; no tolerance was changed to turn them into passes.

Across the 24 primary null cases, Satterthwaite and KR rejection estimates
ranged from **1% to 10%** in both engines; the true-variance control ranged
from **1% to 12%**. With only 100 repetitions, sampling uncertainty is large
even for that correctly specified Gaussian control. At probability 0.05 the
planning MCSE is about 2.18 percentage points. None of these pilot percentages
establishes the planned 4%–6% acceptance band.

The normal-Wald baseline ranged from 2% to 18%. Its largest rate was **18/100**
for G=6, m=3, rho=0.7, between-group predictor, beta=0, in both engines:
pointwise Wilson 95% interval **11.7%–26.7%**. This is an exploratory warning
about that inference convention for small groups, rather than evidence of a
candidate/reference numerical discrepancy. Confirmation must assess each
method and case separately.

## Primary null cases

Counts below are out of 100. All primary-case fits converged in both engines.
R counts match the candidate's counts; coverage at the null is the complement
of rejection because the scalar test and interval are paired. The final column
is the pointwise Wilson interval for candidate KR rejection, expressed as
percentages. Full casewise estimates, failure envelopes, paired differences,
bootstrap MCSE and exploratory strata are in the archived JSON reports.

| G | m | rho | Predictor | Normal | Satterthwaite | KR | KR Wilson 95%, % |
|---:|---:|---:|:---|---:|---:|---:|:---|
| 6 | 3 | 0.2 | within | 6 | 3 | 3 | 1.0–8.5 |
| 6 | 3 | 0.2 | between | 6 | 3 | 2 | 0.6–7.0 |
| 6 | 3 | 0.7 | within | 6 | 3 | 3 | 1.0–8.5 |
| 6 | 3 | 0.7 | between | 18 | 8 | 8 | 4.1–15.0 |
| 6 | 10 | 0.2 | within | 6 | 5 | 5 | 2.2–11.2 |
| 6 | 10 | 0.2 | between | 5 | 2 | 1 | 0.2–5.4 |
| 6 | 10 | 0.7 | within | 2 | 2 | 2 | 0.6–7.0 |
| 6 | 10 | 0.7 | between | 14 | 5 | 5 | 2.2–11.2 |
| 12 | 3 | 0.2 | within | 6 | 5 | 5 | 2.2–11.2 |
| 12 | 3 | 0.2 | between | 4 | 4 | 4 | 1.6–9.8 |
| 12 | 3 | 0.7 | within | 6 | 5 | 5 | 2.2–11.2 |
| 12 | 3 | 0.7 | between | 6 | 5 | 5 | 2.2–11.2 |
| 12 | 10 | 0.2 | within | 8 | 7 | 7 | 3.4–13.7 |
| 12 | 10 | 0.2 | between | 4 | 3 | 3 | 1.0–8.5 |
| 12 | 10 | 0.7 | within | 7 | 6 | 6 | 2.8–12.5 |
| 12 | 10 | 0.7 | between | 7 | 4 | 4 | 1.6–9.8 |
| 30 | 3 | 0.2 | within | 7 | 6 | 6 | 2.8–12.5 |
| 30 | 3 | 0.2 | between | 6 | 6 | 5 | 2.2–11.2 |
| 30 | 3 | 0.7 | within | 2 | 2 | 2 | 0.6–7.0 |
| 30 | 3 | 0.7 | between | 3 | 3 | 3 | 1.0–8.5 |
| 30 | 10 | 0.2 | within | 7 | 7 | 7 | 3.4–13.7 |
| 30 | 10 | 0.2 | between | 6 | 5 | 5 | 2.2–11.2 |
| 30 | 10 | 0.7 | within | 10 | 10 | 10 | 5.5–17.4 |
| 30 | 10 | 0.7 | between | 2 | 1 | 1 | 0.2–5.4 |

## Evidence and reproduction

Run ID: `20261011T133949Z-1b0e5939`. Candidate release Git SHA:
`0b4760fcc5a7558a60ec738401f326a793cb6b1d`; native module SHA-256:
`fd754e69381368bf8ccd1105f71aca3fdbca740874ad5bd9c316262699edac9f`.
Python 3.11.15, polars 2.0.0, R 4.6.1, lme4 2.0.1, lmerTest 3.2.1,
pbkrtest 0.5.5, jsonlite 2.0.0. Complete observed R namespaces are archived.

The [evidence archive](https://github.com/x4g4p3x/lme-rs/releases/download/v0.2.7/lmm-calibration-pilot-v2-0.2.7.zip)
contains exact designs and responses, seeds/controls/source hashes, every
attempted outcome, R session information, frozen sources/locks, distribution
bounds, original analysis, and its explicitly recorded diagnostic revision.
It excludes the virtual environment and third-party package binaries; restore
the exact dependencies to rerun fits. Archive SHA-256:
`d88be7383994c3a304a269948ea7cb4d2ab4d1a50b8acf238b650fa35c7ea95f`.

Original summary SHA-256:
`432f3b52cbb8d21c052cfe70b25a560e53809cf3712cf10cff4ae2cc0a1da637`.
Corrected `summary-reanalysis.json` SHA-256:
`e8142193190f13cb5321e65f0770e308132e4c5ef7721f87a0f82eb7d08b333f`.
Revision `v2-diagnostics-2` separates exceptions/unknown convergence from
nonconvergence and prevents missing singularity flags from becoming known
nonsingular events. Numerical summaries and paired differences are identical
before/after this correction; no responses or fits were repeated or replaced.
The original source snapshot is preserved alongside the corrected analysis.

From a checkout with the harness, `task scientific:calibration` creates a new
isolated pilot. Exact original responses are in the archive; source snapshots
allow inspection of the original run. The diagnostic correction can be rerun
with `python scripts/lmm_calibration.py reanalyse RUN_DIRECTORY` after removing
only an existing corrected report from a separate working copy. The command
checks original artifact/source hashes and refuses to overwrite reports.
Early development smoke runs were separate harness checks and were not pooled
into these 8,000 attempts. Pilot seeds cannot enter a confirmatory campaign.

## What confirmation still requires

Observed fit/inference time was about 19.7 seconds for the candidate and 400.9
seconds for R, including both adjusted inference calls per dataset. These are
budget observations from one Windows environment, not a fair speed comparison.
A linear extrapolation gives roughly 22.3 R hours for 20,000 repetitions across
all 80 cases before parallelism, setup, archival and analysis costs. A full
study needs bounded workers and streaming analysis, rather than unbounded
memory or a long synchronous MCP call.
Runtime uncertainty has not been bootstrap-analyzed; recorded per-fit times
and these totals are descriptive budget evidence.

The reproducible [binomial planning calculation](../comparisons/scientific/lmm_confirmation_planning.R)
uses the exact Bonferroni Clopper-Pearson decision rule for 48 claims. Assuming
every true rejection rate is exactly 0.05 and no fits fail, the probability
that a single claim supports [0.04,0.06] is 79.84% at n=10,000 and 99.794% at
n=20,000. A union bound gives at least **90.12%** probability of all 48 claims
supporting the band at n=20,000, without assuming claim independence. This is
a planning assumption, not an estimate of candidate calibration; rates near
the band edges and failures reduce decision power.

Before confirmation, freeze the final repetition budget and confirmatory
generator/analysis, restore and verify an independent R environment, declare
worker/resource limits, and use fresh held-out seeds. Retain the bands, family
and failure policy. Additional models, distributional stresses, GLMM/NLMM and
independent human review remain outside this completed pilot milestone. See the
[completed confirmatory study](CALIBRATION_CONFIRMATORY_0.2.7.md) for the
subsequent 20,000-replicate campaign under protocol v3.

## Implementation checks

Local preflight passed lint, all-target compilation, Cargo audits, legal and
repository metadata checks. Documentation validation passed examples,
six Rust doctests and API documentation. All 50 script-harness tests passed
when the installed R executable was included on PATH, including ten calibration
controls. Required R/Julia formatting checks passed; the new planning script
also received a focused required R formatting check. A fresh 80-dataset smoke
run exercised the corrected aggregation code successfully.

The uploaded archive was downloaded again and its SHA-256 matched the prepared
file. Full Rust/bindings CI and the hosted OS/Python matrix were not rerun for
this milestone; it changes scientific tooling and documents, not the published
engine's Rust mathematics or Python bindings. This does not replace a future
clean-environment confirmatory rerun.
