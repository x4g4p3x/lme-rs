# AI review of the confirmatory LMM calibration protocol and harness

Date: 13 October 2026. Scope: [protocol v3](LMM_CALIBRATION_PROTOCOL.md),
[scientific/renv.lock](../scientific/renv.lock),
[restore_r_environment.R](../comparisons/scientific/restore_r_environment.R),
[lmm_calibration.py](../scripts/lmm_calibration.py), and
[lmm_calibration.R](../comparisons/scientific/lmm_calibration.R).
A separately tasked read-only agent audited the confirmatory protocol and
implementation before freezing and launching the 20,000-replicate campaign.
This is internal AI methodological and implementation review within the same
system, not independent human peer review or certification.

## Verified controls and invariants

- **Seed and namespace isolation:** Confirmatory data generation uses
  `lmm-calibration-v3|confirmatory|<case_id>|<replicate>|<counter>`, which is
  disjoint from the pilot namespace `lmm-calibration-v2|pilot|...`. Bootstrap
  resampling seeds use `lmm-calibration-v3|analysis|confirmatory|<case_id>` for
  the 20,000-replicate candidate/control series and
  `lmm-calibration-v3|analysis|confirmatory|reference|<case_id>` for the paired
  1,000-replicate R reference audit stratum. Pilot outputs cannot enter a
  confirmatory directory or summary.
- **Deterministic sharding and worker invariance:** Per-case `.jsonl.gz` shards
  use `filename=""`, `compresslevel=1`, and `mtime=0`. Worker processes operate
  on disjoint case subsets and aggregate hashes/summaries in canonical registry
  order, so worker count (`1 <= workers <= 16`) and completion timing do not
  alter generated inputs, fits, or summaries.
- **Unit-displacement contrast and reference KR checks:** Both Satterthwaite
  and Kenward-Roger candidate evaluations verify `num_df == 1` on the observed
  and unit-displacement contrasts, positive finite `unit.f_value`, positive
  finite `den_df`, matching `den_df` between observed and unit-displacement
  contrasts (`rel_tol=1e-12`, `abs_tol=1e-12`), and observed `f_value` matching
  `beta^2 / variance`. R reference KR checks `pbkrtest::KRmodcomp` against
  `pbkrtest::vcovAdj` with `F.scaling == 1`.
- **Independent R distributions, Clopper-Pearson bounds, and 48-claim rules:**
  R evaluates exact `qnorm`/`pnorm`/`qt`/`pf` critical values and distribution
  probabilities for all 20,000 candidate replicates per case, and computes
  `qbeta` Clopper-Pearson bounds at `n = 20,000` (`pointwise` and Bonferroni
  `simultaneous` at `alpha = 0.05 / 48`) and at `n_ref = 1,000`
  (`reference_pointwise` and `reference_simultaneous`). Null rejection and
  coverage failure-envelope complementarity holds even when failures are
  present (`failed > 0`) by Beta distribution symmetry.

## Findings resolved before confirmatory execution

| Finding | Resolution before freezing the campaign |
|:---|:---|
| `candidate()` bypassed `THREAD_LIMITS` when `active_workers == 1` by running in-process after importing `lme_python` and `polars` | Always dispatch confirmatory candidate workers through `run_workers(..., active_workers)` so `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `POLARS_MAX_THREADS=1` are set before native module import for all worker counts |
| CLI `--workers` defaulted to `1`, shadowing `manifest["workers"]` on standalone `candidate`/`reference`/`analyse` invocations | Default `--workers` to `None` in CLI parsing (`args.workers or 1` for `generate`), falling back to `manifest["workers"]` when omitted |
| `methods["normal"]` was outside the per-method `try...except` block and unexpected post-convergence exceptions could leave `converged: True` alongside `fit_error` | Wrap `normal` evaluation in its own `try...except` block with finite positive SE validation, and force `converged = False` if `fit_candidate_dataset` catches a top-level fit error |
| Document the separate reference subsample bootstrap seed in protocol v3 | Explicitly record `lmm-calibration-v3|analysis|confirmatory|reference|case ID` alongside the candidate analysis seed in `docs/LMM_CALIBRATION_PROTOCOL.md` |
