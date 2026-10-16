# Correlated random-slope boundary recovery: diagnostic evidence

16 October 2026. **Post-study diagnostics, not a replacement pilot or confirmation.**
The [v1 pilot](RANDOM_SLOPE_PILOT_V1.md), its binary, inputs, results and decisions
remain unchanged. The [machine-readable record](RANDOM_SLOPE_BOUNDARY_FIX_DIAGNOSTIC.json)
contains counts, all remaining numerical flags, controls and artifact identities.

## Candidate and numerical change

The development candidate uses base commit
`922e4600e8635c3a03cd7f3b74a9db1186b0dbe3` plus the separately archived source
changes. This is an uncommitted-source diagnostic build, not an engine release.
CPython 3.11.15, Polars 2.0.0 and one computational thread were used. Native SHA256:
`2ed6b683889945bf25e772ac6d7feaddf60a5592cad7ec25053328fff4999daa`.
Wheel SHA256:
`8c2b9fe6dc01f76ece8d03e6ff312a0e0c6a3d5c28325f6c282dd817f21b9512`.
Per-file source hashes and the working-tree patch accompany the raw diagnostics.

The default optimizer could declare convergence after its simplex collapsed onto
a covariance boundary. LMM recovery now checks equivalent Cholesky column signs,
and entirely zero columns also receive a local quadratic-curvature check. Both
backends require an actual objective improvement and share the caller's iteration
budget across recovery attempts. These finite local checks do not certify global
optimality. The balanced closed-form oracle is a diagnostic, never production code.

## Retained-input refits

All 1,880 original inputs were refitted into a separate directory with the original
pilot controls: tolerance `1e-8`, maximum 5,000 iterations. Original input and outcome
hashes were checked before and after. No original fit was retried or replaced.

| Diagnostic | Original v1 candidate | Fixed development candidate |
|:---|---:|---:|
| Reported convergence | 1,880/1,880 | 1,880/1,880 |
| Evaluable normal, Satterthwaite, Kenward–Roger results | All | All |
| Singular under common eigenvalue criterion | 658 | 548 |
| Balanced objectives above analytic global optimum gate | 119/1,840 | 0/1,840 |
| Balanced objectives below oracle beyond gate | 0/1,840 | 0/1,840 |
| Pairs flagged against retained R numerical gates | 166/1,880 | 51/1,880 |

The largest positive objective gap is `2.0468587536e-7`. The independent AI reviewer
recomputed balanced optima from raw inputs by enumerating feasible active sets and
confirmed the counts. Seven retained R fits remain above that global optimum;
R is not the sole correctness oracle. Forty imbalanced attempts have no global
optimality claim from this balanced oracle.

Objective agreement does not establish parameter precision. At the original controls,
42 balanced fits miss the covariance-entry gate: 39 off-diagonal entries and three
slope variances. The worst miss is about 7.215 times its gate (absolute error about
`1.06056e-5`). Retained-R rejection disagreements are zero for normal and
Satterthwaite inference and one for Kenward–Roger. Coverage was not recomputed in
these diagnostics; the frozen pilot coverage results remain the original results.

## Separate precision experiment

The 42 flagged inputs were selected **after** observing these misses, then refitted
with the same binary, tolerance `1e-12` and the same 5,000-iteration budget. Controls,
selection identities and script hash were saved before fitting. All 42 converged
and none missed the original covariance-entry gate. These selected results cannot
be pooled into pilot estimates. They support investigating stricter prospective
search controls before a new frozen pilot; they do not establish calibration.

Two further limits remain outside that selected experiment. One balanced fit
(`zero_slope-G8-m5-v0-rundefined-beta0.5`, replicate 17) passes covariance-entry
gates but is classified full rank at the shared threshold, while the analytic
optimum is rank one. One imbalanced fit (`imbalanced-G8-mcycle3-5-10-v0.25-r0-beta0`,
replicate 18) retains a Satterthwaite p discrepancy: about `0.4448399` versus
retained R `0.4446636`. The balanced oracle cannot resolve its cause. The sole
Kenward–Roger rejection disagreement is in a different fit where retained R is
above the analytic optimum. Remaining differences must not all be attributed to
near-zero covariance or to R.

## Validation and limits

The public saved-counterexample regression failed before the fix and passed after.
Independent rational rotated controls cover zero, one and two active covariance
bounds. Separate controls cover coupled missing columns, unusable differentiation
scales and recovery at an exhausted iteration budget. The real coupled-column
public example already passed in the old binary; it is a baseline control, not an
additional reproduced public failure.

Rust lint, all-target checks, 143 unit tests, 406 consolidated integration tests,
six doctests, examples and API docs passed. Four heavy production tests remained
ignored. Feature-enabled Clippy, 27 optimizer tests, 26 model-contract tests and
both release golden-parity tests passed; default release golden parity passed too.
The 77-test script suite initially skipped two R-dependent controls; both passed
separately after the installed R executable was added to the command's PATH. Python
lint, script lint and documentation checks passed. No binding code changed; the
isolated wheel's 1,880 fits exercised the affected Python entry point. A new full
binding/platform matrix was not run.

The fair harness ran sleepstudy REML, weighted sleepstudy REML and the 100,000-row
correlated-slope case with three warmups, 11 repeats and phase reporting. Default
Argmin ran in both execution orders; Basin ran in one order. All stages completed.
The unweighted and large-slope comparisons meet their fit-agreement gates. Weighted
objectives retain the [previously documented mismatch](../benchmarks/fair-evidence-2026-09-11.md)
and remain unverified. Prepared/full-fit ratios are not equivalent timing boundaries.
This task makes no new speed, default-backend, platform-wide or calibration claim.

## Reproduction and next step

The separate runner is [diagnose_lmm_random_slope_fix.py](../scripts/diagnose_lmm_random_slope_fix.py).
Use a distinct installed wheel and a new output directory; `--audit-only` reads the
completed diagnostics and retained R records. Raw evidence is stored under
`benchmark-results/scientific/random-slope-pilot/20261016-boundary-fix-joint/`;
the repository JSON records its hashes. The original refit script and later audit
script are archived separately because the audit was added after the refits.

Next: decide and review the tighter prospective numerical controls, retain the
remaining R discrepancies, examine boundary classification and the imbalanced
inference/derivative case, freeze an exact new engine/protocol identity and use
disjoint seeds for a new pilot. Do not promote this selected-input exercise into
confirmation or overwrite v1. The [living work record](../docs/work/2026-10-16-random-slope-optimizer-fix.md)
tracks the implementation and next action.
