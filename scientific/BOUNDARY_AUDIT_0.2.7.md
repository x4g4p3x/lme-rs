# Post-study evidence and boundary audit

Date: 16 October 2026. This addendum corrects the interpretation and checksum
transcription in the [13 October confirmatory report](CALIBRATION_CONFIRMATORY_0.2.7.md).
The original run, protocol, raw records, numerical summary and 48 primary
decisions are preserved. New deterministic fixtures are diagnostic follow-up,
not additional confirmatory observations.

## What the independent audit checked

[audit_lmm_evidence.py](../scripts/audit_lmm_evidence.py) does not import the
campaign runner. It streams all 1,600,000 inputs/candidate records and 80,000
paired reference records, verifies their declared hashes and ordering, checks
per-case and aggregate summaries, recomputes rejection/coverage counts and
failure envelopes, and independently classifies all 48 primary claims. Its
known-variance control uses within-group projection or weighted regression of
group means, rather than the campaign's block-inverse implementation.

The audit passed. Counts, fit/singularity diagnostics, zero paired decision
disagreements and the original global scope gates agree with the stored
summary. Satterthwaite remains 23 supported/1 inconclusive; KR remains
21 supported/2 inconclusive/1 contradicted. Both all-24-case gates remain false.
This audit did not repeat fitting or bootstrap uncertainty calculations. The
archived R distribution tables were checked through their hashes and used for
intervals; this is not a new independent implementation of the t distribution.

The [machine-readable evidence ledger](LMM_EVIDENCE_0.2.7.json) records all 560
case/method rows, verification results, source summary/auditor hashes and the
original method-level gates. Rows outside the original primary family remain
descriptive; supported cases do not redefine the original scope.

## Correction 1: residual variance is re-estimated at the boundary

For a balanced between-group predictor, let G be the group count, m the
observations per group, and x_i the predictor constant within group i. Define:

```text
w = G(m-1), q = G-2, N = Gm
SSW = sum_ij (y_ij - mean(y_i))^2
SSB = m * sum_i (mean(y_i) - a_hat - b_hat*x_i)^2
MSW = SSW/w, MSB = SSB/q
s = sigma^2, v = sigma^2 + m*tau^2, v >= s > 0
```

Here a_hat and b_hat are the OLS coefficients from regression of group means.
Orthogonal within-group contrasts have variance s, while scaled group means
have variance v. Up to a constant, the restricted criterion is
`w*log(s) + SSW/s + q*log(v) + SSB/v`.

Its constrained optimum is:

```text
MSB > MSW:  s_hat = MSW, v_hat = MSB, tau2_hat = (MSB-MSW)/m
MSB <= MSW: s_hat = v_hat = (SSW+SSB)/(w+q), tau2_hat = 0
```

Slope variance is `v_hat / (m * sum_i(x_i-mean(x))^2)`. At a strict boundary
solution, `MSB < v_hat < MSW`. Therefore the original report's `max(MSB,MSW)`
description is incorrect: it holds the within variance at its interior
estimate instead of re-estimating the shared variance at the constrained optimum.

An exact counterexample has G=4, m=2, x=(-3,-1,1,3), group means
(1/10,7/10,23/10,49/10), and observations equal to each mean plus/minus one.
Then b_hat=4/5, MSW=2, MSB=1, and:

| Quantity | Constrained REML | Incorrect maximum formula |
|:---|---:|---:|
| Residual/effective variance | 5/3 | 2 |
| Slope variance | 1/24 | 1/20 |

The incorrect solution raises the restricted criterion by
`6*log(6/5)-1`, approximately 0.09392934. The published engine and R agree
with the correct boundary solution on this fixture. The error is in the
report's explanation, not the boundary solution in those implementations.

[lmm_boundary_oracle.py](../scripts/lmm_boundary_oracle.py) derives the solution
with rational arithmetic and a 70-digit Decimal criterion. Independent dense
rational GLS checks verify the coefficient covariance and criterion for five
fixtures: strict interior, strict boundary, exact boundary and two adjacent
cases. The [fixture evidence](LMM_BOUNDARY_FIXTURES_0.2.7.json) retains inputs,
exact expected values and all observed candidate/reference checks.

The separate group-means t control uses variance `MSB/(m*Sxx)` and q degrees
of freedom. Under the stated balanced Gaussian model this is an exact t
procedure, including when the generating random-intercept variance is zero.
It is not a replacement definition of KR or Satterthwaite. On the counterexample
its variance is 1/40, distinct from the constrained REML variance of 1/24.

These identities help explain why constrained variance estimation and boundary
df treatment matter. Agreement with R alone cannot establish the cause of
every observed calibration departure, nor exclude a shared approximation error.
For this balanced model, v_hat is never below MSB. If a method uses that slope
variance and q df, its rejection region is contained in the exact group-means
t region. The deterministic KR checks support that condition on these fixtures;
they are not a proof for arbitrary random-effects structures or boundary cases.

## A separate implementation defect exposed by the fixtures

With custom controls (`tolerance=1e-10`, `max_iterations=5000`), the published
0.2.7 scalar simplex can converge at tau2=0 when the actual minimum is nearby
and positive. In the near-interior fixture the exact tau2 is 0.002001, whereas
the published adapter returns zero, with a restricted criterion higher by
approximately 2.66e-6. This fails the fixture's prespecified variance/covariance
and criterion checks. It is not hidden by loosening tolerances.

A current-development public-API regression reproduced the same issue. The
development fix uses the existing bracketed scalar profile with caller limits
and tolerance for a single random-intercept variance component. A second
regression ensures a one-iteration search retains its best evaluated center,
while correctly reporting nonconvergence when its budget is exhausted. The original
study used default controls and the existing scalar profile; its stored outputs
are untouched. The fix belongs to the development version and does not change
the published 0.2.7 package or transfer its calibration evidence to a new binary.

The R exact-boundary fixture also returned a tiny positive tau2 (1.24e-7),
missing the declared absolute 1e-7 variance gate while its other numerical
checks passed. This numerical-resolution disagreement is retained. No R
optimizer settings or fixture limits were adjusted to manufacture a pass.

## Correction 2: five report checksums were transcribed incorrectly

The actual files match the checksums stored in the hash-anchored original
summary. Five checksum strings in the report text do not match those files.
The verified values are below; this correction changes no file in the original run.

| Artifact | Verified byte-level SHA-256 |
|:---|:---|
| `binomial-bounds.json` | `1567d05d66df4709334572d5a092d48d155b4c378d58fb056c74f1b8263ae592` |
| `candidate-environment.json` | `cda1dee38f5eccd1ee3415b01e1ccc5a5f368bd99957a35cd3fb5c61155750d6` |
| `reference-environment.json` | `426ee91f16992a6c2f5a4926921649e7092c039a1af18e7c8e26828abb6de1fc` |
| `R-session.txt` | `7f748ce8db3907b81ee15884f7a2ef905c232aa332e0bdc0699862dfe72994b9` |
| Aggregate case-summary digest | `cc100a045e294b19ebaab32a64d9bb3d41bad1d00b5113c8f49462e5643efc79` |

The summary trust anchor remains
`b6f9a8dae787591da4c157ffcc7a9714014e6c8c67f5e3b3edb78bd4c28c399c`.
Manifest, design, candidate binary and frozen-source identities also verified.

## Reproduce without another campaign

```text
task scientific:audit -- --run-dir RUN_DIRECTORY --output NEW_AUDIT_JSON
python -m unittest scripts.tests.test_audit_lmm_evidence scripts.tests.test_lmm_boundary_oracle -v
python scripts/lmm_boundary_oracle.py --output NEW_FIXTURES_JSON
Rscript --vanilla comparisons/scientific/restore_r_environment.R --verify
Rscript --vanilla comparisons/scientific/lmm_boundary_checks.R NEW_FIXTURES_JSON NEW_R_JSON
```

An isolated environment containing the exact published 0.2.7 package can also
run `scripts/lmm_boundary_oracle.py --candidate --output NEW_CANDIDATE_JSON`.
That command intentionally exits unsuccessfully when the observed near-interior
fixture fails, preserving the report. It must not be confused with a failing
repeat of the completed confirmatory campaign.

The full raw confirmatory evidence is present locally. At this audit's release
asset inspection, the public 0.2.7 release contained the pilot archive and
numerical baseline but no full confirmatory archive. A portable bundle is now
prepared locally, excluding the installed virtual environment:

```text
benchmark-results/scientific/calibration-confirmatory/0.2.7/lmm-calibration-confirmatory-v3-0.2.7.zip
Size: 1,815,464,268 bytes; 425 entries; ZIP CRC verification passed
SHA-256: 3cd1ccb4768e487698154da2a62230e94182f5623ee279803bdf733f03527902
```

It includes original inputs, outputs, frozen sources/locks, summary, audit and
post-study fixture sources/results. At preparation time it had not been uploaded.
Raw evidence availability must not be inferred from the readable public report;
future publication should expose and verify this bundle separately. Restore
dependencies/binaries from the archived locks to refit; the evidence audit
itself needs only standard-library Python.

### Public availability addendum, 16 October 2026

The [full confirmatory archive](https://github.com/x4g4p3x/lme-rs/releases/download/v0.2.7/lmm-calibration-confirmatory-v3-0.2.7.zip)
and [checksum file](https://github.com/x4g4p3x/lme-rs/releases/download/v0.2.7/lmm-calibration-confirmatory-v3-0.2.7.sha256)
are now attached to the existing 0.2.7 release. No original asset was replaced.
The GitHub asset digest matches the SHA-256 above. A fresh download from the
release was also hashed locally and matched; the public file has 1,815,464,268
bytes. The archive preserves the original evidence and post-study diagnostics;
publishing it changes no statistical outcome or global scope gate.

## Development validation and remaining limits

After both optimizer regressions were fixed, Rust lint and all-target compilation
passed, as did 139 unit tests, 403 integration tests and six doctests. Four heavy
production tests remained intentionally ignored by the normal consolidated
suite. Both scalar regressions also passed with the optional Basin feature.
Release golden parity passed (two tests); the affected `random_intercept_10k`
Rust/Julia fair-harness case had no failures and passed its fit-agreement gate
(three warmups, eleven repeats, phases recorded). This is an applicable focused
check, not a general speed or before/after performance claim. Its local output is
`benchmark-results/scientific/boundary-fair-20261016.json`.

Documentation checks passed, including all 62 script tests, examples and API
documentation. Script lint, Python lint and required R/Julia comparison formatting
passed. Preflight passed, including Cargo security audits, legal/provenance,
workflow and repository metadata checks. The locked R environment verifier
checked 61 packages and 51 namespaces. Full hosted OS/Python validation, a new
Python wheel and the four heavy production cases were not run for this milestone.

A separate internal AI review checked the constrained derivation, search controls
and prospective random-slope design. Its low-budget search finding was reproduced
and fixed; its pilot reporting requests were incorporated into the draft. This
is internal AI critique, not independent human statistical review. The retained
published-candidate near-interior failure and R exact-boundary precision miss
above remain limitations of those fixture results. During the original audit no
new calibration campaign was run and no original observation was replaced.
The subsequent archive publication is documented in the availability addendum.
