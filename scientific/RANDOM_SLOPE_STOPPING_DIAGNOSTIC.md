# Random-slope stopping precision diagnostic

16 October 2026. Post-study investigation, not a fresh pilot or calibration claim.
The repair and original pilot were delivered in `c87b8e0`. Production math,
optimizer behavior and the public absolute `FitControl.tolerance` contract are
unchanged. The original pilot's numerical-readiness gate remains **false**.

## What caused the retained iteration limit

The exact public Rust API replay reproduces the retained case
`interior-G50-m15-v0.25-r0-beta0.5`, replicate 15, including theta, sigma squared,
objective, final criterion, `converged=false` and 5,000 iterations. Input doubles
are passed as IEEE-754 bit patterns to avoid JSON-parser rounding differences.

Argmin 0.11 checks the absolute **sample standard deviation** of simplex costs.
Its unchanged stopping method, queried on solver clones without objective calls,
reports SD about `1.818989403546e-12` from iteration 200 through 5,000. This exceeds
the requested `1e-12`. The best point is already unchanged at checkpoint 100.
The final 200 evaluations cycle among four adjacent representable points, with
coordinate span about `1.39e-17`; all their cached costs are eight ULPs above the
retained best. A one-best/three-trial four-vertex pattern has exactly the observed
cost SD. This is consistent with a stalled simplex; the dependency's private
simplex was not directly dumped.

At the retained theta, all 100 repeated cached evaluations give
`2361.163568871659`, and all full evaluations give `2361.163568871662`. The cached
value is identical after neighboring probes. This is deterministic rounding
sensitivity across parameters and two evaluation paths, not random variability.

An independently assembled balanced likelihood uses group projections, 2x2
covariance algebra and 60/80-digit Decimal arithmetic. It explicitly recenters
the predictor and transforms the covariance: the original doubles have mean
about `-1.40e-17`, rather than being exactly centered. The retained point's true
criterion is `2361.163568871662848120386229...`; the analytic interior optimum is
`2361.163568871662107451802183...`. The true gap is **`7.406685840455e-13`**.

Separate AI scrutiny independently used a 110-digit uncentered Gram formulation
and independently calculated pi. It agreed with the optimum and all neighboring
likelihoods. Their true final-tail variation is about `2.53e-22`, much smaller
than the computed double-cost spread. This supports rounding-scale search
stagnation; no iteration-accounting defect was demonstrated. AI scrutiny remains
internal review, not human statistical peer review.

## Diagnostic refits and control selection

These refits never replace the completed pilot's outcome. From the default start,
Argmin converges at `1e-8` in 68 iterations and `1e-10` in 82, but retains the
iteration limit at `1e-12` and `1e-14`. Starting from the retained theta converges
even at those strict tolerances: representability does not make convergence
impossible for every search path. Basin converges for all eight start/tolerance
combinations; its default-start `1e-12` result takes 95 iterations. This does not
establish general superiority of either backend.

Before the broader diagnostic, declare one fixed sensitivity: all 94 cases and
20 retained inputs each, absolute tolerance `1e-10`, default starts, maximum
5,000 iterations, the identical installed Python native, four workers and one
numerical thread per worker. No inference or reference refits are included.

- All **1,880** durable starts have unique matching outcomes, all converged;
  maximum 295 iterations.
- All **1,840** balanced constrained-likelihood checks pass unchanged gates.
  Maximum absolute objective gap is `1.30e-9`.
- The largest fraction of any numerical gate is **0.8091**: covariance `01` in
  `interior-G20-m15-v0.25-r0-beta0`, replicate 9. There is finite margin; do not
  infer guaranteed correctness for unseen samples.
- The other 40 imbalanced inputs converge but have no balanced numerical oracle.
  Inference, p/CI, R parity and nominal calibration are not tested by this run.

At the observed criterion range 96.16–2,401.08, `1e-10` is roughly 220–7,037 ULPs.
The present strict-control failure occurs at about 2.2 ULPs. Together with the
unchanged-gate sensitivity, this motivates a **prospective** control proposal.
It is not a universal error bound or permission to reinterpret a caller's
explicit absolute tolerance. No production tolerance floor or data-specific
convergence exception is introduced.

## Provenance, preservation and resources

[Results](RANDOM_SLOPE_STOPPING_DIAGNOSTIC.json),
[source/binary identities](RANDOM_SLOPE_STOPPING_DIAGNOSTIC_PROVENANCE.json),
[sensitivity declaration](RANDOM_SLOPE_STOPPING_DIAGNOSTIC_FREEZE.json) and
[resource receipt](RANDOM_SLOPE_STOPPING_DIAGNOSTIC_RESOURCES.json) bind the evidence.
All 64 engine inputs remain unchanged. The Rust default and Basin executables
are distinct from the retained Python native; their hashes are recorded separately.
All 94 original journals match the published pilot inventory, with nonempty WAL
files rejected. All 32 runtime helper sources match the original freeze.

Initial exploratory outputs and a first sensitivity run are retained separately.
The exploratory analysis predates its captured source and is not used for final
claims; the final run has exact source/provenance receipts.
Before final publication, review added helper-source binding and checks against
the published journal inventory. The fixed-control sensitivity was rerun into
a new directory; no old result or source was overwritten. The final independent
analysis matches its exact executed script hash.

The final sensitivity job took 6.205 seconds, sampled peak committed memory about
450.37 MiB, sampled disk 140.82 MiB (including captured Rust executables), and
maximum polling gap 0.1080 seconds. Caps were 900 seconds, 4 GiB aggregate job
commit, 512 MiB sampled study disk and 1 GiB free reserve. There were no observed
limit events or remaining descendants. Build/preparation/archive and small Rust/
Decimal probes are outside that measured job; it is not a whole-milestone cap.
No speed claim is made. Raw evidence is [archived locally](RANDOM_SLOPE_STOPPING_DIAGNOSTIC_BUNDLE.json),
not uploaded.

## Validation and next gate

Six new script tests cover an independent dense GLS calculation with shifted
predictors, high-precision/local-optimum agreement, bit preservation/nonoverwrite,
invalid designs/output overlap, changed helpers, and changed journals/nonempty WAL.
The complete script suite passes 144 tests with R available. Required lint, 145
Rust unit tests, five affected slope contract tests and focused example Clippy
pass; both release diagnostic examples execute successfully. Documentation checks
are recorded in the [execution record](../docs/LMM_RANDOM_SLOPE_CONVERGENCE_NEXT_STEP.md).

An extra `cargo clippy --all-targets -- -D warnings` fails on five unchanged
test-code lints: `items_after_test_module` in `src/mcp.rs` and four
`single_range_in_vec_init` findings in optimizer tests. Ordinary required Clippy
and the new example pass. No broad CI/hosted matrix, complete bindings suite,
new performance campaign or full scientific confirmation was run.

Follow the [fresh-control pilot plan](../docs/LMM_RANDOM_SLOPE_STOPPING_CONTROL_NEXT_STEP.md).
It requires a separate reviewed protocol, frozen sources and fresh namespace,
all numerical/inference/reference gates unchanged, before numerical readiness
can be reconsidered. Preserve this diagnostic and every earlier failure.
