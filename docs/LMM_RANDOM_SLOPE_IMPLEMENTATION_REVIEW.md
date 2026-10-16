# Internal AI review: random-slope pilot implementation

16 October 2026. This is internal AI critique, not independent human statistical
peer review. No pilot observations were inspected for this review.

The separate reviewer checked the 94-case registry, 36 future interior null cases,
zero/rank-one Gaussian generation, candidate covariance parameterization and
exact-binomial uncertainty/failure envelopes. Independently derived GLS agrees:
for each centered group, `c_i=(mean(y_i),sum(x_i*y_i)/Sxx_i)`,
`C_i=Sigma+diag(1/m_i,1/Sxx_i)`, and
`Cov(beta_hat)=(sum inverse(C_i))^-1`. Positive residual variance makes `C_i`
invertible even at zero/rank-one random covariance; no jitter is required.

## Findings and resolutions before freeze

- A native PyO3 panic derives from `BaseException`, not `Exception`. The adapter
  now retains these failed attempts and propagates only interrupts/system exits;
  a synthetic panic regression checks this distinction.
- Hashing a wheel and installed native separately does not link them. Freeze now
  verifies the wheel's sole native member against the installed native hash.
- A matching original-line hash does not protect the enriched candidate payload.
  Analysis now verifies all original fit/method values, accepting only R-added
  quantiles/distribution checks or a distribution-error wrapper with preserved raw
  values. Original candidate numbers are used for analysis, with a tight tolerance
  solely for verifying the JSON/R round trip. Negative controls alter fit/SE values.
- Direct R package version checks were insufficient. Every R worker now verifies
  all locked packages and namespaces immediately before reference fitting.
- Fit summaries distinguish exceptions, nonconvergence, invalid covariance and
  unavailable diagnostics; method failure reasons and original errors are retained.

The independently proposed numerical comparison gates are explicitly diagnostic,
not statistical standards, optimizer accuracy guarantees or exclusions. Their
absolute floors reflect the standardized predictor/unit-error scale; covariance
and SE relative limits are 0.1%, with a wider df limit for derivative sensitivity.
Every binary decision disagreement is recorded regardless of these gates.

The lead must run all deterministic/integrity controls and inspect diagnostic
execution before launching the pilot. A new full confirmation campaign still
requires its own protocol, decision-power budget and disjoint observations.

The separate reviewer rechecked the resolutions and reported no remaining
substantive pre-freeze blocker. Ten focused controls and the full 72-test script
suite passed before freezing; the installed study wheel also passed the fixed
scalar near-boundary regression. A positive freeze round-trip control caught and
resolved Windows newline dependence by storing a separate canonical JSON digest.
