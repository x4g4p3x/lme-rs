# Scientific validation: next milestone and AI handoff

Prepared 16 October 2026 against checkout `615ea2528f90b148229b77491ade6f2c9f668c40`.
Status: implemented locally on 16 October 2026; final validation is recorded in
the [audit addendum](../scientific/BOUNDARY_AUDIT_0.2.7.md). The original plan
below is retained as the scope record. Read the current `AGENTS.md` before further
work and preserve unrelated changes. This supersedes the earlier suggestion to
launch confirmation: that campaign has now completed.

Follow-up, 16 October 2026: the [random-slope milestone record](work/2026-10-16-random-slope-pilot.md)
and [completed pilot report](../scientific/RANDOM_SLOPE_PILOT_V1.md) now supersede
the implementation handoff below. The previous raw archive has been publicly
published and its fresh download verified; the new pilot exposed nonoptimal
reported-converged fits, which must be resolved before further calibration.

Registered in the [living work record index](work/README.md). This existing plan
is the record for this milestone; no duplicate task file is needed. The user
authorized commit/push delivery on 16 October 2026 after validation. This record
was reviewed before that delivery; Git history identifies its containing commit.
The starting revision above identifies the baseline, not a commit containing the fixes.

## Material decisions recorded after implementation

Updated 16 October 2026 (Europe/Berlin). The audit preserved the original study's
records and false global gates. Independent constrained-REML identities exposed a
reporting error and a custom-control optimizer defect; the development fix and
regressions do not change the published 0.2.7 binary. The five report checksum
corrections, retained R fixture precision miss, actual verification commands/results
and unrun checks belong to the [audit addendum](../scientific/BOUNDARY_AUDIT_0.2.7.md).
The random-slope design remains prospective and unfrozen. Raw evidence is bundled
locally but has not been uploaded. These are observed outcomes and follow-up
decisions, not a claim that they were all known before implementation.

## Objective

Turn the completed random-intercept study into an independently checked,
reproducible statement of what the published engine supports and where its
inference is limited. Resolve the explanation of the unexpected cases before
expanding calibration to random-slope models. Do not change mathematics merely
to make the observed rejection percentages pass.

## Verified starting point

The [confirmatory report](../scientific/CALIBRATION_CONFIRMATORY_0.2.7.md) evaluates
the published **0.2.7** engine, not the current `0.2.8-dev.0` checkout. It records
80 fixed Gaussian random-intercept designs, 20,000 datasets per design,
1,600,000 candidate/control datasets and a prespecified 80,000-dataset paired
R audit. The model was `y ~ x + (1 | group)`, REML, with one continuous slope,
fixed designs, Gaussian errors, no weights and no offsets.

| Result | Recorded outcome |
|:---|:---|
| Candidate fits | 1,600,000 converged; no reported inference failures |
| Paired R audit | 79,994 evaluable; six retained nonconvergences |
| Common-evaluable binary decisions | No candidate/R rejection or coverage disagreements for each method separately |
| Satterthwaite primary cases | 23 supported, one inconclusive, none contradicted |
| Kenward-Roger primary cases | 21 supported, two inconclusive, one contradicted |
| Original all-24-case validation gates | Both false |

At `G=6, m=3, rho=0.2`, with a between-group predictor and null slope, KR
rejection was 2.10% (simultaneous interval 1.79%–2.46%), below the prespecified
4%–6% band. Satterthwaite was inconclusive in that case. Normal-Wald rejection
reached approximately 12% in some small-group between-predictor cases. Agreement
with R supports implementation agreement; it does not by itself prove nominal
calibration or the report's proposed causal explanation.

Do not repeat the pilot or full confirmatory campaign as routine setup. Existing
outcomes remain valid historical evidence even if later analysis finds an error.

## Read first

- [Programme](SCIENTIFIC_VALIDATION_PLAN.md) and frozen [protocol v3](LMM_CALIBRATION_PROTOCOL.md).
- [Confirmatory results](../scientific/CALIBRATION_CONFIRMATORY_0.2.7.md) and
  [AI harness review](LMM_CALIBRATION_CONFIRMATORY_REVIEW.md).
- [Runner](../scripts/lmm_calibration.py), [CI command runner](../scripts/ci/lme_ci.py),
  [R adapter](../comparisons/scientific/lmm_calibration.R) and
  [harness tests](../scripts/tests/test_lmm_calibration.py).
- [Scientific environment instructions](../scientific/README.md),
  [Python lock](../scientific/uv.lock), [R lock](../scientific/renv.lock) and
  [R restoration adapter](../comparisons/scientific/restore_r_environment.R).

Only if an implementation discrepancy is reproduced, inspect the relevant
[contrast](../src/contrast.rs), [Satterthwaite](../src/satterthwaite.rs),
[KR](../src/kenward_roger.rs) and [KR comparison](../src/kr_modcomp.rs) code.

## Work in this order

### 1. Verify and preserve the completed evidence

Inspect status/diffs and the latest instructions. Locate run
`20261013T210450Z-2ce9f0b4` under
`benchmark-results/scientific/calibration-confirmatory/0.2.7/`.
The following local digests matched the report when this plan was written:

| Artifact | SHA-256 |
|:---|:---|
| `summary.json` | `b6f9a8dae787591da4c157ffcc7a9714014e6c8c67f5e3b3edb78bd4c28c399c` |
| `manifest.json` | `24e2d061c781e5d26d375cda58625fe5a117bb38ed798410ac2b1baa18aab3de` |
| `designs.json` | `3a385f18d3419e0a9f3827a4ebf1bdc95b0cd64ccd9f2627b99a7fa58381341b` |

Verify the per-case input/output and summary hashes against their manifest
declarations, not just these three files. Check missing/duplicate replicates,
failure denominators, the paired-audit selection and the 48-claim decision
counts. Report unavailable artifacts explicitly. If the local run is missing,
locate a verified archive; do not silently regenerate substitute evidence.

Retain the exact published candidate version and binary hash. Development
changes after 0.2.7 have not inherited its statistical validation. Frozen-source
checks may reject current checkout files: use the matching archived sources in
an isolated location rather than altering old manifests or bypassing hashes.
Verify the locked R environment using the documented restoration adapter.

Deliver a short evidence audit and a reusable verifier with negative controls
for corrupted, missing and duplicate records. Prefer streaming case-by-case;
do not load the full raw campaign into memory unnecessarily.

### 2. Independently check the boundary explanation

The report attributes conservative inference to constrained variance estimation.
Treat this as a hypothesis requiring independent mathematical and numerical
support, rather than concluding it solely from candidate/R agreement.

Derive a small balanced between-group Gaussian reference calculation using
orthogonal within-group residuals and regression of group means. State the
normalization and degrees of freedom of every sum/mean square. Derive the
interior REML solution and the constrained boundary solution, including
re-estimation of residual variance. In particular, audit the report's
`max(MSB, MSW)` description: imposing a zero variance component is not generally
equivalent to holding residual variance at its interior estimate and clipping
one parameter. Verify the actual constrained likelihood optimum.

Implement the resulting small oracle without calling candidate, lme4,
lmerTest or pbkrtest fitting routines. Use exact/rational arithmetic where
possible and separately justified high-precision calculations where necessary.
Include deterministic interior, exact-boundary and near-boundary examples,
along with an exact group-means t/F control for the balanced between-predictor
design. Compare variance estimates, slope covariance, criterion and scalar
test/interval consistency against both implementations. The group-means control
is a distinct inferential procedure; do not relabel it as KR or Satterthwaite.

Use preserved representative replicates for diagnostics and label such selection
as post-study analysis. Distinguish a reporting error, a method limitation, an
oracle error and an implementation defect. Correct a report through a dated
addendum with reproducible evidence; preserve the original study and decisions.
For a confirmed library defect, add a regression first and follow `AGENTS.md`
validation. A changed engine needs new version-specific evidence and fresh
confirmation; never replace its predecessor's outputs.

### 3. Publish a precise evidence scope in project documentation

Produce a machine-readable, versioned case/method evidence table and a concise
reader-facing guide. Separate implementation agreement, numerical identities,
statistical calibration, failure rate and exploratory results. Preserve the
original 48-claim family, simultaneous bounds, acceptance bands and both false
global gates. Report supported cases as the tested finite set; do not turn a
post-hoc selection into a claim that the original primary scope passed.

Explain the supported and non-supported settings, observed boundary frequency,
and normal-Wald limitations. Generating rho is known in a simulation, not in an
ordinary user's dataset: avoid a runtime assurance rule that requires true rho,
or an automatic method switch presented as newly validated. Changing defaults
or implementing a hybrid procedure is separate work requiring its own study.

Audit whether the full confirmatory inputs, outputs, frozen sources, locks and
checksums are durably retrievable outside one workstation. Prepare any missing
portable evidence bundle and reproduction instructions under repository release
rules. Identify raw-data availability separately from a readable summary.

### 4. Design the following extension; do not launch it implicitly

After the evidence audit and boundary explanation are resolved, write a new
protocol for Gaussian random-intercept/random-slope models, beginning with
`y ~ x + (1 + x | group)`. Define the slope estimand, within/between predictor
variation, group counts/sizes, random-slope variances and correlations,
near-singular cases, matched reference methods, failure policy and MCSE budget.
Introduce complexity in stages, starting with a compact diagnostic pilot.
Keep misspecification, crossed/nested structures and GLMM/NLMM campaigns as
explicit later extensions rather than silently adding them to this milestone.

Choose the repetition budget from the new decision family and precision needs;
do not automatically reuse 20,000 or the old family size of 48. Freeze a new
protocol/source/environment revision and disjoint seeds before confirmation.
New scientific choices informed by the old study are prospective hypotheses
for fresh data. Record AI critique accurately as internal review.

## Required verification and boundaries

For implementation, select checks from the current `AGENTS.md` before editing.
At minimum, new harness code needs focused independent controls and
`task lint:scripts`; CI-runner/tooling changes require `task preflight`; R/Julia
changes need the applicable comparison formatting checks. Documentation needs
`task docs:check`. Rust/binding fixes require their full applicable tiers,
regression coverage and changelog entries. Keep Git hooks enabled and report
unrun or skipped checks. This instruction-only plan needs no changelog entry.

Do not modify the frozen v3 protocol, old manifests/results or completion-score
scope to obtain a passing claim. Do not silently retry failures or discard
singular fits. Do not launch the existing full confirmatory command merely to
answer an audit question. Do not release, change defaults or send messages to
third parties as a side effect of this handoff.

## Milestone acceptance checklist

- [x] Evidence integrity, replicate counts and all primary decisions are independently checked.
- [x] Boundary interpretation is supported by an independent constrained-REML derivation and fixtures; the general causal explanation remains explicitly limited.
- [x] Reporting and implementation defects are reproduced and corrected without rewriting history; see the addendum for validation.
- [x] Versioned evidence table and guidance preserve the unsupported/inconclusive results and original scope gates.
- [x] Raw evidence availability and clean-environment reproduction requirements are explicit; a verified local archive is prepared, with public upload still outstanding.
- [x] A separate random-slope pilot protocol is ready for implementation review, with execution and confirmation gated on new freezes.
- [x] The implementation handoff below identifies the exact next action; checks and limitations are recorded in the addendum.

## Implementation handoff after this milestone

The audit, [evidence ledger](../scientific/LMM_EVIDENCE_0.2.7.json),
[inference guide](../scientific/INFERENCE_EVIDENCE_GUIDE.md) and
[boundary fixtures](../scientific/LMM_BOUNDARY_FIXTURES_0.2.7.json) are now present.
The current-development scalar optimizer fix has two public-API regressions in
`tests/test_model_contracts.rs`. The original run and published binary are unchanged.

The next executable action is to implement the 94-case registry and independent
known-variance GLS controls specified in
[LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md](LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md), starting
with deterministic zero/rank-one covariance and seed-separation tests. Then
complete that draft's freeze checklist and run tiny diagnostic smoke fixtures
before its 1,880-dataset pilot. Do not launch a full confirmation study from this
handoff. The prepared confirmatory archive also needs separate durable public
publication and checksum verification under the repository's release rules.

## Methodological sources

Use [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086) for the
ADEMP structure, Monte Carlo precision and transparent reporting.
[White and colleagues, How to check a simulation study](https://doi.org/10.1093/ije/dyad134)
supports investigating unexpected results using known-answer settings and
separate checks of generation, fitting and analysis. These are methodological
guides; the project's 4%–6% band is a declared engineering criterion, not an
external certification standard.
