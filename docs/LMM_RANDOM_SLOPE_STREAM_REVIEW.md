# Internal AI review: streaming rehearsal implementation

16 October 2026. A separate AI reviewer inspected the store, runner, analysis,
R worker/checker, prospective protocol and retained rehearsal data. It did not
run fits or builds or change the implementation. This is internal independent
scrutiny, not human peer review or statistical certification.

Before source freeze, review identified and implementation corrected:

- Reports must audit worker exit/environment/native-memory receipts, even when
  fit rows look complete; failed final handshakes cannot silently pass acceptance.
- Independent R checks require exact unique primary/count keys, raw identity
  checks, finite two-endpoint bounds and balanced-oracle records.
- Owned Windows jobs assign suspended roots before resume and kill descendants
  on close; POSIX sessions cover inherited R children. Failed R construction
  cleans up; actual root-crash and startup-failure regressions pass.
- Every reference variant has attempted, interrupted, exception, genuine optimizer
  nonconvergence, invalid-fit and method/numerical counters. Selected comparisons
  cannot hide failed unselected variants. Python rejects numeric booleans.
- Freeze the complete original engine input inventory and record analysis runtime;
  unknown R namespaces fail environment checks.

The reviewed implementation preserves durable starts, immutable outcomes,
bounded iteration, generic reference eligibility and criterion-only selection
without switching to an inference-friendly reference. After these corrections,
the updated focused tests and independent R fault controls passed before inputs.

Post-run review independently verified source/native/freeze identity, 94 immutable-read
journals, all 188 regenerated inputs, exact 188 candidate plus 752 reference
start/outcome inventories, execution receipts and native high-water provenance.
The sole tight-default bobyqa code-3 failure is retained and correctly ineligible,
even though its objective matches the oracle. Per-native-process peaks/sums are
not a measured simultaneous or whole-machine peak.
Independent raw analysis reproduced all 564 count/moment rows and 72 CP bounds.
Separate NumPy active-set enumeration checked all 184 balanced candidate and 184
selected-reference objective/covariance/rank results; maximum criterion gaps
were 1.27e-11 and 1.82e-12 respectively. All 188 inputs differ from corresponding
v1/v2 inputs. Public summary/R-check bytes, resource/freeze descriptors and every
315-entry raw archive member match the retained evidence and SHA/CRC descriptor.

The [rehearsal report](../scientific/RANDOM_SLOPE_STREAM_REHEARSAL_V1.md) records
actual validation and limitations. A larger scaling/cap rehearsal, full execution
integrity review and final confirmation freeze remain necessary. Two replicates
per case cannot establish calibration or future optimizer reliability.
