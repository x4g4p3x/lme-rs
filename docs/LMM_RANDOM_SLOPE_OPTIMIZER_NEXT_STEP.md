# Next milestone: near-zero covariance false convergence

Created 16 October 2026 after the capped scaling rehearsal. Implement this bounded
correctness milestone before any confirmatory campaign. Full confirmation and
release are not authorized by this plan. Read the
[living record](work/2026-10-16-random-slope-scaling-caps.md) and
[results](../scientific/RANDOM_SLOPE_SCALING_V1.md) first. Preserve every historical
input, outcome, freeze and local raw archive.

## Reproduce before editing

Use both entries in
[`RANDOM_SLOPE_SCALING_V1_COUNTEREXAMPLES.json`](../scientific/RANDOM_SLOPE_SCALING_V1_COUNTEREXAMPLES.json).
Each contains its complete design/response, controls via the source freeze,
original native/engine identity, failed candidate, analytic optimum, selected R
fit and immutable raw-audit/input hashes. Both are balanced G8/m5 random-slope
models `y ~ x + (1 + x | group)`, REML, tolerance `1e-12`, maximum 5,000 iterations.
Require convergence for eligibility even though the fitting control permits
returning a nonconverged fit.

Create focused Rust public-API regressions using these exact synthetic inputs
and independently recorded expected optima. Keep the original failure observable
before implementation. The n10 near-correlation/null rep0 gap is
`0.06063436057795`; the n20 zero-slope/null rep5 gap is `0.00534295504636`.
The original candidate converges at numerical covariance rank zero, but the
sample's correct constrained REML solution has rank one. Tests should establish
the criterion gap and covariance/residual-variance accuracy, rather than only a
convergence flag or agreement between two copies of the same optimizer. Do not
use the generating variance as the expected sample estimate.

Start with `src/optimizer.rs::covariance_restart_with_eval`,
`zero_column_restart`, existing covariance-boundary recovery tests and
`tests/test_bug_hunt.rs`. Register a new integration target in
`tests/ci_consolidated.rs` if one is added. Relevant lower Cholesky fits have theta
approximately `[0, -2.2765685e-7, 0]` and `[0, -2.4960139e-7, 0]`.
The whole-column `sqrt(epsilon)` guard appears to skip a necessary joint escape
when a practically zero column retains a tiny off-diagonal. This is a hypothesis:
confirm through the actual regression and objective probes before choosing a fix.

## Repair and acceptance

Make a principled recovery decision based on the covariance/objective scale and
verified feasible improvements. Merely enlarging a coordinate threshold or
loosening analytic tolerances is insufficient. Preserve truly optimal boundaries,
equivalent column orientation, finite arithmetic, caller iteration limits and
best-improving-probe retention. Exercise response/predictor rescaling and nearby
nonzero-column cases; ensure the fix does not turn a local escape diagnostic into
a claimed global-optimality certificate.

Read `OPTIMIZATION.md` before editing this throughput path. Run failing focused
regressions first, then their fixed forms and affected recovery tests. Required
validation includes `task lint`, `task test:fast`, affected integrations and the
broader `task rust` slice for this cross-module public fitting behavior; avoid
duplicating components satisfied by the broader slice. Run applicable fair
performance-harness cases with comparable environment/revision and correctness
checks; report regressions and avoid a general speed claim. If bindings change,
also run their required checks. Add an Unreleased changelog entry and create/
maintain the new milestone's living record before implementation and before commit.

Once the repair passes, build a new isolated candidate wheel and record all
source/lock/wheel/native/installed-byte identities. Keep old scientific runners,
protocols, manifests and source snapshots immutable. A historical freeze is
verified against its archived sources, not rewritten to match the new engine.
Use a separate reviewed runner/protocol with disjoint fresh seeds for the new
engine's numerical pilot; do not refit or replace these failed archived outcomes.
Require independent balanced-control/known-GLS and fixed-bank R checks, retained
all-attempt failures and independent AI source/result review before declaring the
new binary eligible for confirmation planning.

## Gates after the repair

The 940/1,880-data engineering rehearsals do not prove calibration or provision
a full campaign. Production work still needs bounded job scheduling and durable
resume for the planned 1,196,000 candidate / 94,000 reference datasets, analysis/
audit/storage/scratch/archive sizing, an explicit decision on descriptive
bootstrap reporting, and actual production caps. The present runner only permits
10 or 20 replicates per case; do not simply remove that guard. Freeze the final
implementation/protocol, review the concrete campaign and obtain explicit campaign
authorization before generating confirmatory inputs. AI review does not constitute
human statistical peer review.

## Dated implementation update, 16 October 2026

The [near-zero column repair milestone](work/2026-10-16-near-zero-column-recovery.md)
reproduced and fixed both cases, with Rust/default/Basin and independent fresh
pilot evidence. Original failures remain unchanged. The fresh pilot has no
converged-candidate balanced numerical miss, but one honest iteration-limit
failure keeps numerical readiness false. Follow the separate
[convergence diagnostic plan](LMM_RANDOM_SLOPE_CONVERGENCE_NEXT_STEP.md) before
the conditional production preparation milestone. This update does not change
the completed pilot's controls, failed outcomes or acceptance gate.
