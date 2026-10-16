# Next milestone: boundary recovery and covariance precision

Created after the fresh stopping-control pilot, 16 October 2026. Its 1,880
candidates all reported convergence, but three fail unchanged independent
numerical gates. The original analysis also aborted on a disk-sampling race.
A separate hardened controller completed diagnostic analysis without changing
any fit or granting retrospective acceptance. Preserve all these outcomes.

## Reproduce the three retained failures before production edits

Use the [immutable counterexamples](../scientific/STOPPING_CONTROL_PILOT_V1_COUNTEREXAMPLES.json)
and [report](../scientific/STOPPING_CONTROL_PILOT_V1.md). Create a prospective
living record and failing public API regressions using exact input bits, original
controls/default starts and independent analytic expectations. Do not turn
diagnostic refits into new validation data.

1. `interior-G8-m5-v0.05-r0-beta0`, replicate 3: converged after 53 iterations
   at effectively rank-zero covariance; the independent optimum has rank one
   and criterion lower by `0.00154515777153`. The first Cholesky diagonal is
   about `3.05e-6`, larger than the existing coordinate guard. Inspect whether
   that outer guard skips the objective-equivalent zero-column check before it
   can run. Evaluate the retained/zero-column/nearby feasible objectives to test
   this mechanism; do not simply enlarge a coordinate threshold.
2. `interior-G8-m15-v0.25-r0.5-beta0`, replicate 15: criterion agrees closely,
   but covariance `01` differs by `2.7078722e-6`, beyond `1.7733984e-6`.
3. `interior-G50-m15-v0.25-r0-beta0.5`, replicate 13: covariance `01` differs
   by `-1.5900777e-6`, beyond `1.5339017e-6`, despite a small objective gap.

The last two cases test parameter precision: a small simplex cost spread does
not certify individual covariance entries. Distinguish flat curvature, objective
rounding, parameter scaling and stopping semantics. Compare profile/full likelihood
and independently assembled dense/balanced calculations at fixed points, as well
as genuinely separate diagnostic refits. Keep the original `1e-10` outcomes and
the earlier `1e-12` iteration-limit failure unchanged.

## Repair the cause and preserve the public contract

Use objective-based evidence for boundary classification if confirmed. Include
genuinely nonzero/true-boundary/near-zero negative controls, joint escapes,
response/predictor units and row order. For interior precision, investigate
stable objective evaluation, scaling or principled local refinement. Any added
refinement must respect the existing total iteration budget and report failure
honestly. Exercise default and Basin backends. Do not loosen oracle gates, add
data-specific exceptions or silently reinterpret explicit absolute FitControl
tolerance. A public semantic change requires a documented contract, independent
stationarity/negative controls and a new prospective validation policy.

Add durable regressions that fail before the repair and pass afterward. Run the
required Rust tier, affected feature tests, release golden parity and fair cases
for throughput changes; maintain Unreleased changelog and the living record.
Retain exact before/after engine and binary identities. No general performance
or global-optimization claim follows from local recovery.

## Validate the repaired engine separately

The separate [hardened sampler](../scripts/lmm_resource_caps_stable.py) now handles
vanished SQLite children and rejects denied subtree scans, missing roots and
other measurement errors. Use it explicitly in a new independently reviewed
protocol/runner; never patch old frozen helpers. Exercise the cap/store negative
controls and preserve source/command/resource identities.

After actual repair and source review, build/identify the candidate and freeze a
separate fresh pilot with a disjoint namespace, justified prospective controls,
all numerical/inference/reference gates unchanged and the reviewed resource
policy. Preserve every failed study. No retry or replacement of frozen study
outcomes, calibration claim or production campaign is authorized by this plan.
Only a reviewed fresh
pilot meeting its own gates can reopen
[production preparation](LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md).
