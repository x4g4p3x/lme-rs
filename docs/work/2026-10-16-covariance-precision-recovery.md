# Boundary recovery and covariance precision

Created prospectively, 16 October 2026 (Europe/Berlin). Status: complete; repair, fresh pilot and
independent result/publication audits pass. Starting revision `93dc06acec14671348432a1155d3efc80b62261c` is
already pushed; local/tracking/live remote agree and checkout was clean. The user
authorized this next milestone. Follow the
[repair plan](../LMM_RANDOM_SLOPE_PRECISION_RECOVERY_NEXT_STEP.md).

## Scope and acceptance

Reproduce all three retained stopping-control failures with exact input bits,
original controls/default starts and independent analytic expectations before
changing the engine. Preserve every historical source, fit and failed gate.
Investigate the small-nonzero-diagonal guard separately from interior covariance
precision. Repair confirmed causes without loosening numerical gates or explicit
absolute FitControl tolerance, and preserve feasibility and shared budgets.

## Intended implementation and checks

Use existing model-contract and optimizer tests, a bit-preserving derivative of
the immutable public fixtures, and focused fail-before/pass-after logs. Compare
fixed-point objective/full evaluation and analytic covariance before choosing a
repair. Consider objective-equivalent column origins independently of a diagonal
coordinate cutoff. For interior precision, investigate a bounded, curvature-
scaled local correction accepted only after verified feasible objective decrease;
avoid data-specific thresholds or a new arbitrary stopping tolerance.

Test genuine interior/boundary optima, nonstationary/flat-curvature controls,
response/predictor units, row order, default/Basin and hard iteration budgets.
Read OPTIMIZATION.md/BENCHMARKS.md; retain a fair-harness baseline before production
edits. Required checks: focused regressions, `task rust` plus Python Ruff, affected
Basin tests, release golden parity, applicable default/Basin fair cases, script/
documentation checks for actual harness changes. Report regressions and skips.

After actual repair and source review, identify a new isolated wheel and separately
freeze fresh numerical validation using the reviewed hardened sampler, justified
prospective controls and unchanged gates. No release/production/confirmation run.
Independent AI scrutiny is separate from human peer review. Maintain this record,
Unreleased changelog and exact source/binary identities.

## Observations and next action

All three exact-bit public regressions failed before production edits with the
archived criterion/covariance differences; two older counterexamples passed.
Two synthetic regressions also failed for the skipped diagonal and weak interior
coordinate before the repair. Retain the local fail-before logs.

The objective-equivalent origin check now runs independently of the tiny-diagonal
cutoff. A diagonal too near its face for a feasible centered precision stencil
also receives feasible face proposals. This is search coverage, not an equivalence
or optimality claim: every accepted proposal must improve the original incumbent.
The original first fixture's independent origin penalty is `1.0219e-10`, slightly
above the requested `1e-10`; origin equivalence alone is insufficient.

Interior recovery uses centered coordinate differences at relative `eps^(1/4)`
scale, a rounding-resolution curvature guard, bounded interpolation and verified
actual descent. Preserve improving sampled endpoints even when curvature is
unresolved or negative. Restart both backends with coordinate-specific verified
displacements and representable own-scale floors, preserving shared budgets and
the caller's absolute cost-SD semantics. A failed/limited search stays ineligible.
Independent review also identified improved single/pair boundary curvature samples
that the older search could discard; retain those samples and test narrow wells.

The 31 public contract tests pass with all three new inputs under seven response/
predictor-unit, reversed-row and alternative-start variants, including residual
variance and the original eigenvalue rank convention. Local optimizer tests cover
true optima, quantized/flat/invalid costs, negative-curvature sampled descent,
anisotropic coordinates and precision recovery at the final budget iteration.

A diagnostic executable and independent existing balanced likelihood compare
fixed points at 60 and 80 decimal digits. All checked cached/full Rust costs agree
within the diagnostic rounding-resolution check; maximum discrepancies for the
three inputs are `5.31e-14`, `1.12e-12`, `3.14e-12`. Observed improvements are
`0.00154515777`, `1.6657e-10`, `6.3725e-10`. This is diagnostic replay, not fresh
validation. Recompute final identities after source formatting/review.

Save all fair measurements, including the first proposal's 1.5--2x slowdown.
Coordinate-specific restarts reduced the observed overhead; small slope fits
still cost roughly 40--50% more in local single-session samples. Do not turn
these correctness/performance tradeoffs into a general speed or parity claim.

Broader validation and independent review are in progress. The separate
[fresh pilot protocol](../LMM_COVARIANCE_PRECISION_PILOT_PROTOCOL.md) is declared
before any inputs. Freeze and execute only after final engine/source checks;
preserve any new failures without tuning that study into success.


## Final repair checks and fresh freeze

`rust-all` passed on the final engine: 152 unit tests, all ordinary integration
coverage and doctests, 575 passing tests in total; four existing heavy cases
remain ignored. Final focused optimizer coverage is 22 default/35 Basin tests;
both backends pass all 31 contract tests and two release golden-parity checks.
Python and script Ruff pass. All 172 script tests pass with R on PATH, including
the optional R controls. `docs-check` and local Markdown links pass.

Independent source review required preservation of every actually improved
boundary/precision sample, coordinate-specific restart steps, source/command/raw
receipt/log bindings for cap controls, and real memory-denial evidence. All were
implemented and reviewed before freeze. The unchanged historical study files and
archived sources remain verified; only current engine and regression-target
changes are explicitly allowed. AI review is not human peer review.

Build receipt pins all 64 engine inputs before the isolated wheel build, with
identical source hashes afterward. Native SHA:
`77211eac27abdd1be2c27243b5a60d295e47a52f839664998a78222fe6ff6cbc`.
Wheel SHA:
`f37776f76d8d7cc3ee83cf68291408f67d422612411987bbfa5f4ed43b72f09c`.
Python 3.11.15, Polars/runtime 2.0.0, package 0.2.8.dev0; installed dependencies
are compatible. Time/disk/memory synthetic controls passed before any pilot
input; their full verified source/command/plan/start/end/log evidence is copied
and hash-bound into the new study.

The new namespace `covariance-precision-v1-20261016-a` freezes 136 sources and
unchanged controls/gates/94-case inventory at 20 replicates. No fitting has begun
at this update. Run the four phases sequentially; publish actual outcomes and
failed gates, independently review results and archive raw evidence. Earlier
`final` fair measurements overlapped an unrelated verification job; retain them
but exclude them from performance conclusions. A separate `serial` pair of fair
runs is in progress without concurrent tests/builds/fits.


## Fresh outcome and independent audit

All four capped jobs completed sequentially. Engineering acceptance and the
prospective numerical-readiness gate pass; confirmation remains false. All
1,880 candidates and 7,520 R variants converge. All 1,840 balanced candidate
checks pass; all three methods have 1,880 eligible pairs with zero numerical,
rejection or coverage discrepancies and zero method failures. Maximum criterion
error is 8.23e-11; maximum 174 outer iterations. The 40 imbalanced inputs lack the
balanced oracle. Thirty unselected R numerical misses remain retained. At n=20,
all 72 calibration rules fail: no nominal-rate claim.

Independent read-only result audit regenerated every input, matched 9,400
starts/outcomes, recomputed active sets (1,338 interior/500 one-binding/two
two-binding), all 564 counts/moments and 72 bounds, source/build/native/wheel,
variant-selection, command/cap/PID and copied/original control receipts. No
blocker found. See the [report](../../scientific/COVARIANCE_PRECISION_PILOT_V1.md)
and [review](../LMM_COVARIANCE_PRECISION_REVIEW.md).

Final serial fair cases retain all fit checks and final binary/source identities.
Default small slope fits cost about 38--45% more; 100k slopes 13--20%; Basin ranges
are 39--61% and 10--26%. Scalar/crossed cases stay near baseline. Baseline-first
local samples are not order-balanced general performance evidence. All earlier
prototype/overlapping reports remain retained and excluded from final comparisons.

Candidate/reference/analysis/R-check jobs take 52.823/242.964/9.976/4.637 seconds.
Peak sampled reference-job commit is 1,853.96 MiB; maximum sampled disk 25.950 MiB;
max sampling gap 0.14167 seconds. Actual scope/caps/charges/events remain in raw
receipts. Pilot timing across different inputs is not a performance comparison.

Complete archive/public-copy verification and final local-link/whitespace checks.
Then the next milestone is the existing
[production preparation plan](../LMM_RANDOM_SLOPE_PRODUCTION_NEXT_STEP.md): implement
and test bounded scheduling, segmented analysis and resource/storage provisions,
prepare a concrete immutable campaign manifest, and keep the full confirmation
campaign disabled until explicitly authorized. No release or full campaign ran.
Current milestone changes remain uncommitted after the already-pushed 93dc06a.


## Evidence package and delivery boundary

The local raw archive contains 637 entries / 42,985,066 bytes; SHA
`ca3a1a675b7bbc9c3520c88378f54b3a0d9e8f2ff051806d0eb0d97e2e4fe69d`.
CRC and every entry/local-byte comparison pass. It includes frozen sources,
inputs/starts/outcomes, wheel/build/package metadata, cap/analysis/R receipts,
diagnostics, all performance attempts and four distinct benchmark CSV fixtures,
baseline optimizer source snapshots and 43 sanitized pre-packaging console logs.
Installed environments are excluded. The initial packaging command stopped on a
list-shaped diagnostic JSON while extracting benchmark fixture metadata; resume
with a type guard preserved all scientific sources/outcomes and prior evidence.
The [descriptor](../../scientific/COVARIANCE_PRECISION_PILOT_V1_BUNDLE.json) states
local-only availability; no upload is implied.

All ten public JSON artifacts are finite/parseable. Final links and whitespace
checks pass, and a final freeze check confirms all 136 source copies/current
files and cap-control bindings still match. Public-context/credential-pattern
checks find no matches. Final independent publication audit passes: all 637
archive entries/local mappings, public-copy hashes, resources, 43 original/
sanitized log identities, fixed-point likelihood and performance figures agree.
No publication blocker found.

The user's requested prior delivery was already present at turn start. A fresh
fetch and live remote query still show local/tracking/remote master equal to
`93dc06acec14671348432a1155d3efc80b62261c`. No empty commit or duplicate push was
needed. The completed next milestone remains in the working tree for its own
commit; it does not authorize a release/full confirmation campaign.


Final record review: implementation and all 136 frozen execution inputs remain
unchanged since successful validation. Final documentation links/whitespace pass.
The next milestone is production campaign preparation; no confirmation input,
full campaign or release was generated. This completed milestone is uncommitted.


## Delivery follow-up: Python summary regression

The existing Python smoke test expected an exact historical intercept variance
string (611.9033), while the precision-recovery fit returned 612.0911. A fresh
independent R 4.6.1/lme4 REML fit on the same sleepstudy rows and formula using
BOBYQA with `rhoend=1e-10` produced 612.089748274669 and REML criterion
1743.62827195849. The test now checks the structured variance against that
independent value within 0.05 and verifies that the formatted summary contains
the same model value; it no longer asserts an obsolete exact optimizer output.
The focused regression passed; the full Python test directory passed with 93
passed and one skipped, and `task lint:python` passed. The prior pilot receipts
remain historical and are not a new validation of the integrated head. Recheck
the combined Rust and dependency state before final delivery.

The public performance receipt was redacted for delivery: two machine-local
fixture-directory paths were replaced with their repository-relative path. The
bundle descriptor records both the unchanged archive original and the new
public-copy hash. Measurements, input identities, and the raw archive remain
unchanged; the earlier byte-for-byte public-copy audit predates this redaction.
All nine current public-copy SHA-256 values were rechecked against the updated
descriptor and match.

On the original working tree, the follow-up `cargo test --locked --test
test_model_contracts` passed all 31 tests before integration with newer master.
This local commit must be integrated with the separately delivered dependency
and CI changes; validate the combined head before remote delivery.

After integrating later dependency, Python-binding, CI and Basin changes, nine
of the 136 historical frozen inputs differ from the combined checkout:
`Cargo.lock`, `Cargo.toml`, `python/Cargo.lock`, `python/Cargo.toml`,
`python/pyproject.toml`, `python/src/input.rs`, `python/src/lib.rs`,
`scripts/ci/lme_ci.py` and `src/optimizer/basin_backend.rs`. The historical
archive and frozen identities remain unchanged; its pilot and timing results
do not validate this new combined source. Re-run applicable software checks
on the combined tree and do not treat historical measurements as current.


## Integration regression investigation

The combined checkout passed `task preflight`, but `task rust` failed three
previously passing precision contract cases. The small-nonzero-diagonal case
reproduced alone with criterion 118.3728812168659 versus independent expected
118.37133605909438. The original committed pre-merge snapshot passed the same
focused test in a separate worktree. The dependency and configuration changes
are being isolated; the combined tree is not ready for delivery. Do not loosen
tolerances or represent historical pilot results as validation of this tree.

The focused case also passed with the newer complete dependency configuration in
an isolated checkout. Both checkouts had shared a Cargo target directory during
this investigation, so a forced rebuild of the integrated checkout was required
to distinguish its artifacts. After rebuilding the integrated library, all 31
model-contract cases passed, including the three earlier failures. This is
consistent with stale cross-worktree artifacts, but does not establish exactly
which previous artifact caused the failure. The integrated checks below passed; the historical pilot remains historical.


## Integrated validation and delivery

The combined tree passed `task preflight`, a fresh `task rust`, `task
basin:check`, `task lint:python`, and `task python` including its isolated-wheel
tests (95 passed, one skipped). `task docs:check` passed; the documentation links
and staged whitespace were checked. Release golden parity passed both default
and Basin configurations (two tests each). The initial failed `task rust` and
the shared-target investigation are retained above rather than omitted.

The Rust-only fair harness ran sleepstudy REML, weighted sleepstudy REML and
the 100,000-row random-slopes case, for both optimizer backends, with three
warmups and eleven recorded fits per case. All recorded fits converged and both
reports had zero failures. The reports are locally available under
`benchmark-results/integrated-followup-default.json` and
`benchmark-results/integrated-followup-basin.json`; they are not published.
They record the dirty integrated source content hash
`d3aedba240943ad57ef61091825d0bef7898a9e358e7a8dea900ce96f8f08770`. These
unpaired Rust-only runs establish no cross-language or speedup claim. The
historical pilot and archive remain scoped to their original frozen inputs.

Status: integrated software checks complete; delivery is pending the normal
merge-commit and push hooks. No release or new scientific validation is claimed.
