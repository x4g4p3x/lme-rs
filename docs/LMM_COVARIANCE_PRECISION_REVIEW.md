# Covariance precision repair and fresh pilot review

16 October 2026. This document records independent AI scrutiny; it is not human
peer review or a certificate of statistical correctness. The source review
described below occurred before the fresh study freeze and inputs. This written
record was assembled during execution; result review is complete, with final
archive/public-copy review recorded below.

## Source and numerical method

The reviewer examined the optimizer change, exact-bit regressions, prospective
[protocol](LMM_COVARIANCE_PRECISION_PILOT_PROTOCOL.md), successor runner/reporter/
controller, hardened resource sampler and synthetic controls without editing,
building or fitting models. The independent likelihood diagnostic uses existing
balanced projection algebra checked against dense GLS, at 60 and 80 digits.

Required corrections before execution:

- Retain improved sampled endpoints when curvature is flat, negative or invalid;
  also retain improved boundary differentiation samples when later joint probes
  miss a narrow well. Continue the stronger joint probe before choosing the best
  evaluated improvement.
- Use feasible centered stencils and resolve curvature above sampled-cost
  rounding scale. Clip interpolation to the sampled interval and evaluate the
  actual proposal. Skip unusable origins in quadratic estimation.
- Use coordinate-specific verified displacement/representable steps when
  restarting, avoiding an enormous step in a small coordinate. Preserve the
  shared outer-iteration budget and return nonconvergence with the best point
  when recovery exhausts it.
- Distinguish objective-equivalent origin classification from feasible face
  proposals when a centered diagonal stencil would cross zero. Face proposals
  need not be equivalent; acceptance must beat the original incumbent.
- Freeze the cap-control import closure and reconstruct command identity from
  original interpreter, fixed program and disk target. Verify raw plan/start/end/
  log bytes and outcomes; reject successful workers and self-consistent command
  substitutions. Require allocation denial with its expected exit or a native
  limit notification; OS peak charge alone is not enforcement evidence.

All were addressed and re-reviewed before freeze. The final source review found
no remaining blocker. The reviewer emphasized that coordinate curvature and
feasible face proposals are local search heuristics, with no joint stationarity
or global optimality certificate. The `64*eps*cost` rounding guard is a practical
resolution heuristic, not a proved universal factorization-error bound. The
caller's absolute cost-SD criterion and all frozen numerical gates remain intact.

## Result and publication review

After all four measured jobs ended, a separate read-only audit regenerated all
1,880 fresh responses, matched all 9,400 durable starts/outcomes to the immutable
journals/raw audit, and verified 136 current/archived source identities, 64 engine
inputs and pre-build/native/wheel identities. Historical archived sources remain
unchanged; 20 historical manifest identities cover 12 distinct prior namespaces.

Independent active-set calculations reproduce 1,338 interior, 500 one-binding
and two two-binding balanced solutions. All 1,840 candidate and selected-reference
checks/ranks pass. Maximum candidate criterion error is `8.23e-11`; the largest
fraction of a numerical gate is `0.0690524`, for covariance `01` in
`interior-G20-m15-v0.25-r0-beta0`, replicate 13. Maximum candidate iteration count
is 174. All three methods have 1,880 eligible pairs and zero numerical/binary
decision discrepancies. All 564 count/moment rows and 72 exact bounds reproduce.
There are zero eligible-method failures and zero passing calibration rules.
Known-variance GLS gives 843 rejections and 1,782 covered intervals.

All four R variants remain retained: 30 unselected balanced numerical misses,
zero nonconvergence, selection counts 163/890/494/333 in the frozen variant order.
The audit independently checked objective-only selection, not oracle-based
selection. No outcome replacement or tuned controls were found.

Eight Python/four R native PID/environment receipts, both capped analysis
commands and all executable/script/input/output/thread/start/end bindings match.
The eight-source cap-control closure, original/copied plan/result and fixed
invocation/raw/log bindings match. Time/disk controls terminate with expected
nonzero exits; the memory control has denial marker/exit 91 and a native event.
Its 546 MiB OS peak charge remains diagnostic, not evidence of successful
allocation. All four study jobs meet their prospective caps; largest sampling
gap is `0.14167` seconds, maximum sampled disk `25.950` MiB.

No numerical, accounting, provenance or resource blocker was found. This supports
only the frozen exploratory pilot's declared eligibility. Forty imbalanced inputs
lack a balanced oracle; 20 inputs per case cannot confirm calibration; earlier
studies remain failed. Fresh-pilot runtime is not a before/after speed comparison.
Final publication review passes: archive SHA/size/CRC and every one of 637 unique
entries match the local study/evidence/engine/public mapping. All nine public-copy
hashes, raw summary/freeze/acceptance/build/R/diagnostic copies, derived resource
receipts and 43 sanitized/original console-log identities match. The reviewer
independently recomputed the fixed-point likelihood at 110 digits and the final
performance medians from raw samples; the published errors, improvements and
overhead ranges agree. Validation-log totals also match. New prose and archive
content contain no private-context or credential-pattern matches. Availability
is explicitly local/nonuploaded, all claims stay bounded, and confirmation stays
disabled. No publication blocker found.
