# Next milestone: fresh pilot with prospectively chosen stopping control

Proposed 16 October 2026 after the
[stopping diagnostic](../scientific/RANDOM_SLOPE_STOPPING_DIAGNOSTIC.md).
The reused-input `1e-10` sensitivity motivates a control choice; it is not new
validation evidence. The completed repair pilot remains numerically unready.

## Protocol and source identity before any new inputs

1. Create and register a prospective work record. Implement a separate protocol,
   runner, reporter and tests; do not edit completed scientific execution sources
   or reinterpret their outcomes. State this is exploratory numerical validation,
   not confirmation or a release.
2. Keep the identical 94-case registry and 20 replicates per case: 1,880 candidate
   fits, all four fixed R variants for 7,520 reference fits. Prospectively fix
   absolute Nelder-Mead cost-SD tolerance **`1e-10`**, maximum 5,000 iterations,
   default starts, REML and explicit convergence eligibility. No automatic retries,
   start changes, fit replacement, tolerance adaptation or outcome-based extension.
3. Keep the repaired production engine unchanged. Reuse its exact isolated wheel/
   native only after verifying the original pre-build receipt, all 64 engine
   hashes, installed-native/wheel hashes and matching environment. Explicitly test
   intentional same-engine reuse rather than bypassing an older runner's
   new-native safeguard. Reject a mismatch. Capture the actual current Git base
   and execution-source hashes; never invent a new engine revision.
4. Independently review the new protocol, implementation and control rationale
   before freezing them. Freeze all imported helpers, engine/protocol/test/report
   sources, exact binary/environment identities, gates and controls. Use a new
   `stopping-control-pilot-v1-...` namespace, disjoint from every historical run,
   only after freeze. Tests must reject old namespaces, changed controls/sources/
   binaries, input mutation and output overlap.

## Execute and assess without changing acceptance

Use the reviewed append-only journals, start-before-fit discipline, resume-only-
never-started policy, four workers and one numerical thread. Keep existing
900-second/4-GiB aggregate job-commit/512-MiB sampled-study-disk/1-GiB-free caps,
owned nested processes, measured polling gaps and separate capped analysis/R
verification. Run phases sequentially, without overlapping builds/tests.

Keep all original numerical, method, reference-selection and readiness gates.
Check all 1,840 balanced inputs against independent constrained likelihood and
all 1,880 against the existing GLS/inference/p/CI/count controls. Retain the 40
imbalanced-oracle exclusions. Recompute 564 count rows and 72 exact bounds; small
n still cannot confirm nominal calibration. Secondary moments remain descriptive.
Keep unselected R failures visible and prospectively nongating as before.

Any candidate/selected-reference failure or paired numerical/decision discrepancy
blocks this pilot's readiness. The diagnostic's weakest numerical margin was
only about 19%; unseen inputs may expose another problem. Preserve a failed run
and diagnose it before proposing a separate new protocol. Do not loosen gates or
declare success just because the optimizer reports convergence.

Have separate AI scrutiny verify actual inputs, starts/outcomes, selected and raw
R variants, independent optima/moments/counts, source/binary/environment identities,
resource/analysis-command receipts and publication copies. Clearly distinguish
AI review from human peer review. Archive original raw data and exact sources,
state local/public availability, and publish acceptance independently from
calibration claims. Run required script/docs checks and any affected API checks.

Only a reviewed fresh pilot meeting its own gates allows proceeding to the
[production preparation milestone](LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md).
No production campaign or release is authorized by this plan.

## Execution update, 16 October 2026

Executed under a separately reviewed 122-source freeze and fresh namespace;
all 1,880 candidates report convergence, but three fail numerical gates. The
original analysis also aborted on transient-file disk sampling. A separate
127-source diagnostic freeze and hardened sampler completed the identical
retained-fit analysis/R verification without changing inputs or outcomes.
Original engineering acceptance and numerical readiness stay false. See the
[work record](work/2026-10-16-stopping-control-pilot.md),
[report](../scientific/STOPPING_CONTROL_PILOT_V1.md) and
[next repair milestone](LMM_RANDOM_SLOPE_PRECISION_RECOVERY_NEXT_STEP.md).
