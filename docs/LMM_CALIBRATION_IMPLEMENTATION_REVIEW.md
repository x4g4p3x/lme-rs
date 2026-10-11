# AI review of the pilot implementation

Date: 11 October 2026. A separately tasked agent inspected the frozen pilot
generator, candidate adapter, R adapter and analysis without editing them.
This remains internal AI review, with the same limits as the
[protocol critique](LMM_CALIBRATION_AI_REVIEW.md).

The review found no consequential error in candidate/reference row matching,
unit-displacement scalar variance recovery, attempted-dataset denominators,
Clopper-Pearson failure envelopes, paired bootstrap schedules or paired method
comparisons. A separate group-projection calculation checked known-variance
GLS on balanced and imbalanced designs; that identity is now a durable harness
test, rather than relying only on the reviewer's assertion.

Two aggregation defects were identified during execution: a missing singularity
flag was counted as a known nonsingular event, and exceptions/unknown convergence
states were included in a counter labelled nonconvergence. The raw records
preserve the required information. A separately identified analysis revision
corrects the labels and reports classifiable/unknown singularity counts and
failure bounds; no fit or response is replaced or retried.

Further evidence improvements for future frozen runs are an explicit isolated
candidate interpreter version and checking the unit-displacement contrast's
rank and denominator df against the observed contrast. The pilot already
records the generating/analysis Python version, candidate binary hash, and
observed scalar rank/F consistency. These suggestions are not observed defects
in the pilot's fitted numerical results.
