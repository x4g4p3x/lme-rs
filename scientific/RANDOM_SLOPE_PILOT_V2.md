# Fresh Gaussian random-slope pilot v2

16 October 2026. **Completed exploratory pilot; no confirmatory calibration claim.**
The [frozen protocol](../docs/LMM_RANDOM_SLOPE_PILOT_V2_PROTOCOL.md),
[pre-observation review](../docs/LMM_RANDOM_SLOPE_PILOT_V2_REVIEW.md),
[freeze descriptor](RANDOM_SLOPE_PILOT_V2_FREEZE.json) and
[machine-readable ledger](RANDOM_SLOPE_PILOT_V2.json) own the design, source/binary
identity, all case/method estimates, numerical flags and uncertainty intervals.
All v1 and 0.2.7 inputs, outcomes, decisions and sources remain unchanged.

## Candidate, controls and independence

Engine commit `580b270edfcfd8f544818f3ff8171aacbd30d099`, isolated CPython 3.11.15,
Polars 2.0.0, package version `0.2.8.dev0`. Native SHA256:
`2ed6b683889945bf25e772ac6d7feaddf60a5592cad7ec25053328fff4999daa`.
The wheel rebuilt at that exact commit has SHA256
`cdb073b7be355065df44af5c90c633ed36906b628e2c62a1d4e61fbd1fce1469`;
the installed native matches its native member. This is a development study build,
not a newly published release.

Prospectively fixed controls are tolerance **1e-12**, maximum **5,000 iterations**,
default starts and retained nonconvergence. R uses the unchanged locked
lme4/lmerTest/pbkrtest environment, bobyqa maxfun=100000. No per-dataset retries,
substitutions or tolerance fallbacks were allowed. The [selected controls and
dense derivative experiment](RANDOM_SLOPE_NUMERICAL_CONTROLS.md) informed this
choice but contributed no observations to this pilot.

The protocol ID `lmm-random-slope-pilot-v2` supplies disjoint smoke, pilot and
analysis seed namespaces. The 94 designs match v1; each receives 20 fresh paired
datasets. Exact response arrays were saved. No response vector matched v1 at
any of the 1,880 corresponding case/replicate keys. This is a fresh study, so its
rates are not paired before/after comparisons with v1.

## Observed results

The separate 12-dataset smoke passed every method, numeric/decision and applicable
analytic check before the full pilot began. The full run contains 1,880 paired
datasets: 1,840 balanced and 40 imbalanced.

| Measure | Candidate | Retained R reference |
|:---|---:|---:|
| Reported convergence | 1,880/1,880 | 1,880/1,880 |
| Evaluable normal/Satterthwaite/Kenward–Roger | All | All |
| Singular by common numerical criterion | 519 | 521 |
| Balanced objectives above analytic global gate | 0/1,840 | 9/1,840 |
| Balanced objectives below oracle beyond gate | 0/1,840 | 0/1,840 |
| Balanced variance/covariance precision flags | 0/1,840 | 9/1,840 |
| Balanced common-rank disagreements with oracle | 0/1,840 | 2/1,840 |

The candidate's maximum absolute balanced criterion gap is **4.832e-11**. R's
maximum gap is **1.609**. The analytic diagnostic uses the independently reviewed
constrained REML active-set solution, not the generating covariance or another
numerical fit. These checks cover only this balanced centered random-intercept/
slope design; they do not establish global optimality for the 40 imbalanced fits.

There are **nine numerical parity flags** across all 1,880 pairs. They coincide
with the nine retained R fits above the analytic optimum. All other pairs meet
the unchanged numerical gates, including the 40 imbalanced comparisons. The
reference flags remain in the data and ledger; none was replaced with the oracle
or a better R refit. R is not treated as the sole correctness oracle.

Rejection **and** coverage decisions match on all 1,880 pairs for each of normal,
Satterthwaite and Kenward–Roger inference. Independent raw-count auditing checked
per-engine and paired decisions, convergence and balanced diagnostic inventories.
Agreement between methods/packages does not establish calibration against truth.

After completion, an independent AI audit verified all 87 archived/active source
hashes, 94 input hashes, 376 outcome hashes, wheel/native identities and all R
environment records. It regenerated every response vector and independently
enumerated balanced spectral active sets, confirming the table, the nine-flag
overlap and all raw paired decisions. This is internal AI audit, not human peer review.

## Statistical and computational limits

Each of the 36 balanced interior null cases contributes only 20 outcomes per
method. Observed Satterthwaite and KR rejection counts range from zero to three
per case. The ledger retains pointwise exact intervals, binomial MCSE,
all-attempt failure envelopes and descriptive performance estimates. A zero count
in 20 attempts still has an approximately 16.8% two-sided exact upper bound.
These data cannot establish a 4%–6% rejection band, simultaneous coverage or a
general small-sample accuracy claim. Do not pool cases to replace a required
casewise decision rule or infer reliability from zero observed failures.

With four processes and one computational thread each, candidate/reference phase
wall times were **41.6 s / 62.6 s**. These include study and inference work and are
not a fair speed benchmark or a general cost guarantee. Peak worker memory is
explicitly unavailable; the known Windows launcher-shim counters were disabled.
Prior native resource observations guided the bounded pilot but are not new v2
memory measurements.

All 89 script/scientific controls passed before observations, including six v2
integrity/analytic controls and six independent dense-derivative controls. Script
lint, required R/Julia formatting, documentation/examples/doctests and link checks
passed. No Rust or binding code changed in this step; the preceding engine's
validated Rust/Basin/golden/fair checks were reused. This is internal AI review
and local validation, not human statistical peer review or a new hosted platform
matrix. No full confirmatory campaign or package release was run.

## Artifact anchors and next action

- Freeze SHA256: `480f910c278913ac752b0d13cf3b1b1afe7254a414ae5cd5656cda30356ed9a2`.
- Pilot manifest SHA256: `616c858a10d9acce88f57f0d7580dc90936002cd4afef4b36b2bbc9d0de0eda4`.
- Raw summary SHA256: `011e1692e1958c230a7ec1ce551b094a5bce5b36599e3c18298a231c5c3b54f0`.

The local raw evidence lives under
`benchmark-results/scientific/random-slope-pilot/20261016-v2-{freeze,smoke,pilot}/`.
The [archive/checksum descriptor](RANDOM_SLOPE_PILOT_V2_BUNDLE.json) identifies the
30.3 MB raw bundle (524 entries, CRC verified). It is locally available and has
not been uploaded or attached to the old 0.2.7 release.

The next milestone is **confirmation design and exact decision-power/resource
planning**, including how to handle the reference optimizer's retained failures.
The proposed scope has 72 primary rejection claims (36 balanced interior null
cases times two adjusted methods); null interval coverage is its paired complement,
not an additional independent claim. Recalculate repetition requirements for that
family and a declared global decision-power target; do not automatically copy
20,000 repetitions or the old 48-claim correction. Freeze a distinct confirmatory
protocol, seeds, failure rules, analysis implementation and budget before any
confirmatory observations. See the [living work record](../docs/work/2026-10-16-random-slope-numerical-controls.md).
