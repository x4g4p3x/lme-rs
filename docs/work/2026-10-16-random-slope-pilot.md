# Random-slope pilot and raw-evidence publication

Created prospectively 16 October 2026 (Europe/Berlin), before implementation.
Status: **pilot completed; numerical blockers documented; validated for delivery**.
The user authorized commit/push and continuation with the numerical fix on
16 October 2026. This record was reviewed before delivery; Git history identifies
the containing commit. Original frozen study sources and outcomes are unchanged.
Baseline: `1c5057e3270b4010cac2958e5bf25c8826335561`,
clean working tree. The user authorized all steps described in the prior handoff:
build and check the harness, freeze it, run diagnostic checks and the 1,880-dataset
pilot, and publish the earlier confirmatory raw-evidence archive. A new release
or full confirmatory random-slope campaign is outside this task.

## Scope and acceptance

- Implement the [94-case draft](../LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md), deterministic
  disjoint smoke/pilot seeds, independent known-variance GLS controls, paired R
  execution and explicit uncertainty/failure reporting.
- Validate zero/rank-one covariance, diagnostic thresholds, scalar inference,
  covariance-order comparisons, immutable records/hash/order checks and resources.
- Identify the exact fixed engine source and isolated binary/environment. Obtain
  internal AI critique, resolve findings and freeze sources/settings before pilot
  observations. Preserve every failure and diagnostic variant.
- Run small diagnostic smoke checks, then all 94 cases x 20 paired replicates.
  Report implementation agreement, failures, cost and uncertainty without a
  confirmatory calibration claim. Preserve the earlier study unchanged.
- Verify and publish the existing 0.2.7 confirmatory archive without replacing
  existing release assets; verify its public checksum and retrieval.

## Intended implementation and checks

Use a separate pilot harness and R adapter, reusing only audited utility functions
where appropriate. Freeze the implemented protocol and exact source snapshots;
retain rational dense-matrix controls independent of the fitted engines. Start
with at most four processes and one computational thread per worker.

Run focused scientific script tests, `task lint:scripts`,
`task lint:comparisons:required`, `task docs:check`, and `task preflight` if adding
CI-runner aliases. Build/install and exercise an isolated wheel from the recorded
engine source; this is a study build, not a published version. Apply the binding
or Rust validation tiers only if defects require implementation changes.

## Decisions, observed results and limitations

Implemented the separate harness, R adapter, exact/closed-form controls, frozen
candidate/source identity and paired failure/uncertainty analysis. The 12-dataset
smoke passed all gates. All 1,880 pilot fits and inference results were evaluable
in both engines, but 166 pairs exceeded numerical gates. A post-study independent
global optimum found 119 nonoptimal candidate and seven nonoptimal R fits among
1,840 balanced attempts. The [report](../../scientific/RANDOM_SLOPE_PILOT_V1.md),
[ledger](../../scientific/RANDOM_SLOPE_PILOT_V1.json) and
[counterexamples](../../scientific/RANDOM_SLOPE_OPTIMUM_COUNTEREXAMPLES_V1.json)
own the scientific evidence. No pilot fit was retried or replaced.

The internal AI reviewer independently derived GLS and the post-study spectral
optimum. Its pre-freeze findings produced panic-retention, wheel/native identity,
paired-payload and full R-environment integrity checks, with negative controls.
The original v1 freeze and outputs remain unchanged; post-study oracle/resource
diagnostics are separately identified and excluded from pilot estimates.

The existing 0.2.7 raw archive and checksum were added to its public release,
without replacing old assets. GitHub's digest, a fresh full download and anonymous
HTTP retrieval agreed with the recorded archive identity. The new pilot's raw
bundle is local; it has not been uploaded to the 0.2.7 release or published as a
new engine release.

Before freeze, ten focused controls, the 72-test script suite, docs/examples/
doctests, required R/Julia formatting, script lint and preflight passed. The exact
isolated wheel passed the existing scalar-boundary regression. The later optimum
diagnostic passed three rotated rational active-set fixtures against dense GLS,
plus degeneracy rejection. Final checks after reporting are recorded in the report.

Memory counters in the frozen orchestrator sampled Windows launcher shims. This
measurement limitation was retained, with a separate R process-tree observation,
explicit correction and one native-candidate resource-only replay; no replay was
included in the statistical results. No Rust/binding implementation changed, so
the earlier engine validation was reused; this task does not claim a new full
hosted platform matrix, package release or confirmatory study.

## Next action

Start the next numerical-fix milestone with the candidate's largest retained
counterexample: G50,m5,v1=0.25,r=0.5,beta1=0,replicate19. Add a public-API
regression against the independent optimum/covariance, inspect scalar-coordinate
boundary collapse in the three-parameter optimizer, and fix the cause without
loosening gates. Review default/custom controls and optional Basin separately.
R is not the sole oracle: its largest counterexample is retained too. Follow the
applicable bug-hunt/AGENTS validation, preserving all v1 records. After fixes,
freeze a distinct binary/protocol with disjoint seeds before a new pilot; choose
the full-confirmation family/budget only after numerical blockers are resolved.
