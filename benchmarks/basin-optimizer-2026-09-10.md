# Basin optimizer comparison (updated September 11, 2026)

Basin 1.10.0 and Argmin produce effectively identical numerical results for all
twelve fair-harness cases, including `nested_10k`. Both builds use upstream
`349e07f` with the optional Basin backend added.

The larger workloads have similar timings in this run. Basin/Argmin cold-fit
ratios are 0.998 for large random slopes, 0.989 for crossed 20k, and 1.018 for
nested 10k; prepared-fit ratios are 1.003, 0.996, and 1.014, respectively.
Smaller cases have mixed results. Variation on unchanged scalar searches limits
interpretation of small differences, so these measurements do not establish a
general speed advantage for either backend.

## Method

- Fresh release builds from the same source, compiler, release settings, and
  deterministic CSV fixtures, differing only in the `basin` Cargo feature.
- Basin 1.10.0 with `ndarray_v0_16` and default features disabled; Argmin
  0.11.0.
- Intel(R) Core(TM) Ultra 7 155U, NixOS, rustc 1.98.1 (48a229cea 2026-09-01).
- Both backends pinned with `taskset --cpu-list 0,2`: one logical CPU from each
  of the two physical performance cores.
- Two A--B--B--A blocks, with three warmups and eleven measured fits per
  process: 44 samples per backend, case, and metric. The table pools all samples
  and reports medians. Per-process medians and paired ratios are in the
  artifact.
- One BLAS thread and two Rayon threads. Timing instrumentation disabled;
  deviance counts collected in separate instrumented prepared fits.
- Measurements ran from 07:06:26 to 07:07:06 UTC on September 11. Total CPU use
  averaged 7.2% before the run and 10.9% during measurement, peaking at 19.5%
  during measurement, including the benchmark itself. Afterward, CPU use
  averaged 23.0% and peaked at 63.7% in the three-second monitoring window.
- [Raw samples, CPU monitoring, numerical results, source and binary hashes, and
  diagnostics](basin-optimizer-2026-09-10.json). [Feature usage and harness
  commands](../docs/BASIN.md).

This report and its raw artifact replace the September 10 Basin 1.9.0 results.
The dependency update and upstream rebase both postdate that run, so timing
changes between reports cannot be attributed solely to Basin 1.10.0. Existing
Rust/Julia reference artifacts and completion scores are unchanged.

## Timings

Ratios are Basin/Argmin; values below one mean a lower measured time.

  | Case                       | Argmin cold (ms) | Basin cold (ms) | Cold ratio | Prepared ratio | Deviance evaluations, Argmin -> Basin |
  | :------------------------- | ---------------: | --------------: | ---------: | -------------: | ------------------------------------: |
  | `sleepstudy_reml`          |            0.532 |           0.536 |      1.008 |          0.971 |                            124 -> 124 |
  | `sleepstudy_weighted_reml` |            0.463 |           0.426 |      0.919 |          0.931 |                                     - |
  | `penicillin_crossed_reml`  |            0.126 |           0.117 |      0.929 |          0.962 |                              87 -> 87 |
  | `pastes_nested_reml`       |            0.163 |           0.159 |      0.972 |          0.946 |                              85 -> 85 |
  | `random_intercept_10k`     |            0.658 |           0.644 |      0.979 |          0.930 |                              21 -> 21 |
  | `random_intercept_50k`     |            2.972 |           2.864 |      0.964 |          0.981 |                              21 -> 21 |
  | `random_intercept_100k`    |            4.488 |           4.464 |      0.995 |          0.991 |                              21 -> 21 |
  | `large_random_slopes_100k` |           58.207 |          58.087 |      0.998 |          1.003 |                            162 -> 162 |
  | `crossed_20k`              |           28.098 |          27.794 |      0.989 |          0.996 |                            103 -> 103 |
  | `nested_10k`               |           10.939 |          11.132 |      1.018 |          1.014 |                            124 -> 124 |
  | `cbpp_binomial_ml`         |           12.013 |          12.336 |      1.027 |              - |                                     - |
  | `grouseticks_poisson_ml`   |            8.126 |           8.275 |      1.018 |              - |                                     - |

Evaluation counts include post-fit work and are distinct from optimizer
iteration counts. Counts were identical across all four diagnostic runs for each
backend and match between backends for all nine unweighted LMM cases.

For crossed 20k, paired process cold-fit ratios ranged from 0.976 to 1.030, and
prepared-fit ratios from 0.974 to 1.034. Both backends made 103 deviance
evaluations. The unchanged scalar controls had pooled cold-fit ratios ranging
from 0.964 to 1.027; this run does not isolate optimizer overhead from other
timing variation.

## Numerical results and validation

All 96 untimed fit reports and 72 diagnostic reports reproduce their backend's
numerical results across processes and report convergence. Across all twelve
cases, the largest theta difference is below `2.4e-14`, and the largest
optimizer-objective difference is below `2.7e-9`. Optimizer objectives agree
with final evaluated objectives within an absolute tolerance of `1e-7` or a
relative tolerance of `1e-10` in every case.

Both backends return theta `[4.987494145328055, 5.059652262073164]` for
`nested_10k`, with optimizer deviance `5944.826928395`, evaluated deviance
`5944.826928378`, and 124 deviance evaluations. The fair GLMM fixtures use
retained scalar searches; vector GLMM and NLMM coverage comes from the
implementation tests.

The refreshed checkout passes 131 default and 140 Basin unit tests, 307
integration tests per build, and six doctests per build. Four pre-existing heavy
production-load cases remain ignored. Release golden parity passes with both
backends (two tests each). Preflight and the Basin feature checks pass,
including Rust/Python lint, all-target compilation, dependency audits, legal
checks, repository metadata dry run, and Basin documentation generation.
Documentation checks also pass, covering local links, dashboard JSON drift, Rust
examples, doctests, and generated documentation. The optional repository-admin
token check was skipped because the token is unset. Python bindings and the
hosted multi-OS matrix were not run.
