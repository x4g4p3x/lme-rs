# Dependency migration evidence — 10 October 2026

The remaining dependency PRs required coordinated API changes. This migration
retains Argmin as the default optimizer and preserves the existing statistical
controls and reference tolerances.

## Compatibility findings

- **#34, grouped maintenance:** Basin 1.15.1 deprecated `run_loop` and
  `TerminationCriterion`. Replace them with `run_loop_with_control` and a
  `RunControl::stop_when` hook. Keep strict sample cost standard deviation,
  projected bounds, invalid-value rejection, error propagation, and one budget
  across recovery restarts. Require Basin 1.15.1 in the manifest so consumers
  cannot resolve an earlier API.
- **#37, random distributions:** rand_distr 0.6 requires rand 0.10. Migrate
  sampling to `RngExt` and OS-seeded construction to `rand::make_rng`, including
  cross-validation, simulation, bootstrap, examples, and both benchmark suites.
  Keep seed derivation per replicate unchanged.
- **#35/#36/#39, array types:** ndarray-linalg 0.18.1 and ndarray 0.17.2 must
  migrate together in the library, Python crate, and all three BLAS target
  entries. Both lockfiles now resolve one ndarray version, 0.17.2. Basin has a
  matching backend. Argmin-math 0.5.1 still targets ndarray 0.16, so select its
  `vec` feature and reuse an ndarray parameter buffer for objective calls.
  A regression with strided starting parameters checks logical coordinate order.
  The coordinated implementation is carried by #35; #36 and #39 are duplicate
  manifest-only updates once that implementation is merged.

Primary references: [Basin run controls](https://docs.rs/basin/1.15.1/basin/fn.run_loop_with_control.html),
[Argmin-math features](https://docs.rs/crate/argmin-math/0.5.1/features), and the
[rand 0.10 migration guide](https://rust-random.github.io/book/update-0.10.html).

## Matched-fit measurements

Baseline: `732ef366dcd0e682808f032f66e63f7587fb2723`. The candidate was measured
as a dirty migration snapshot based on PR #35's original head; the retained
[compressed source patch](dependency-migrations-2026-10-10/candidate-source.patch.gz) reconstructs
the measured Rust source and Cargo changes. Raw reports retain source, lockfile, harness,
input, and binary hashes. Later documentation and merge-parent changes do not
change the measured Rust source. Check those hashes when reusing this evidence.
Windows Git checkouts can convert lockfile line endings. The retained patch
reconstructs the measured Rust source hash; comparing lockfiles after newline
normalization confirms the same dependency graph. Normalize the candidate root lockfile
to LF when checking its recorded measurement hash.

Windows x86_64, Ryzen 5 8600G, Rust 1.99.0, Python 3.11.15, release profile,
Intel MKL; BLAS/Polars/Rayon limits set to one thread before process startup.
Each case runs in a separate process with two warmups and ten measured fits;
prepared-fit phases use the same controls and data. Builds and timings were
serialized. Julia was not included in this bounded dependency comparison.

Both backends retained exactly matching objectives, independently reevaluated
objectives, coefficients, theta, convergence flags, and iteration counts in all
three cases, including each recorded cold/prepared fit check. The intermediate
Basin control migration also matched every recorded fit check exactly.

Timings below are medians in milliseconds with min–max ranges. These are one
session per revision/backend, without balanced repeated sessions; the small
changes do not establish a performance improvement or regression across the
library. No optimizer-default change or general parity claim follows.

| Backend | Case / phase | Baseline ms (range) | Candidate ms (range) |
|:---|:---|---:|---:|
| argmin | sleepstudy_reml / cold_fit | 0.628 (0.619–0.644) | 0.601 (0.592–1.068) |
| argmin | sleepstudy_reml / fit_prepared | 0.601 (0.593–0.634) | 0.572 (0.569–0.619) |
| argmin | sleepstudy_weighted_reml / cold_fit | 0.581 (0.555–0.614) | 0.544 (0.530–0.582) |
| argmin | sleepstudy_weighted_reml / fit_prepared | 0.504 (0.503–0.616) | 0.487 (0.486–0.489) |
| argmin | cbpp_binomial_ml / cold_fit | 14.582 (14.401–15.064) | 12.723 (12.564–13.889) |
| basin | sleepstudy_reml / cold_fit | 0.584 (0.566–0.592) | 0.560 (0.550–0.670) |
| basin | sleepstudy_reml / fit_prepared | 0.543 (0.535–0.786) | 0.526 (0.522–0.531) |
| basin | sleepstudy_weighted_reml / cold_fit | 0.527 (0.509–0.565) | 0.514 (0.502–0.584) |
| basin | sleepstudy_weighted_reml / fit_prepared | 0.476 (0.461–0.668) | 0.453 (0.452–0.455) |
| basin | cbpp_binomial_ml / cold_fit | 13.025 (12.870–14.157) | 12.767 (12.597–13.248) |

Raw evidence: [Argmin baseline](dependency-migrations-2026-10-10/baseline-argmin.json),
[Basin baseline](dependency-migrations-2026-10-10/baseline-basin.json),
[Basin control migration](dependency-migrations-2026-10-10/maintenance-basin.json),
[Argmin candidate](dependency-migrations-2026-10-10/candidate-argmin.json), and
[Basin candidate](dependency-migrations-2026-10-10/candidate-basin.json).

## Validation

- Default build: 133 unit tests and 380 integration tests passed.
- Basin build: 148 unit tests and 383 integration tests passed, including the
  existing independent lme4 covariance-boundary fixtures and shared-budget checks.
- Both builds passed the two release golden-parity tests without fixture or
  tolerance changes. Each test exercises the maintained reference cases.
- Python editable, isolated CPython-wheel, and isolated stable-ABI-wheel suites
  each passed 86 tests, with three optional skips: the workflow-comparator module
  and two artifact-notice checks looking for a persistent `python/dist` wheel.
  Clean wheel consumer examples passed in both wheel modes, and the Rust
  sleepstudy quick-start ran successfully. These reuse the passing components
  of the consumer-smoke flow.
- Default and Basin doctests/docs passed, including warnings-denied Basin docs.
  Documentation checks validated links, examples, dashboard data, and 34 harness
  regression tests. Feature-enabled Clippy and all-target checks also passed.
- The rand migration separately passed the three sequential/parallel simulation,
  bootstrap, and grouped cross-validation regressions.
- Four heavy integration cases remain intentionally ignored in normal PR runs;
  they belong to the release/manual production gate. The local Windows build
  cannot establish macOS Apple Silicon BLAS behavior; hosted validation is required.

Commit and push hooks remain enabled. The exact final PR head must pass hosted
security audits and the supported OS/Python matrix before merge.

## Delivery

All 19 hosted validation checks passed on each reviewed head, including the
supported OS/Python matrix, minimum Rust, and Cargo/Python security audits.
The two publication jobs skipped as expected for PR validation.

| PR | Reviewed head | Merge commit |
|:---|:---|:---|
| [#34](https://github.com/x4g4p3x/lme-rs/pull/34) | `2769a0b3130cc1669233069454e9ac92ffdc21d7` | `f4e0905431e0b04223f8cf5f29eb1e8439525261` |
| [#37](https://github.com/x4g4p3x/lme-rs/pull/37) | `0ccf7517e0c137e398cdf512ba83b134f072aa2a` | `5cc64dd1c2673ec9c9ccc2f1ee23184e6a6f6072` |
| [#35](https://github.com/x4g4p3x/lme-rs/pull/35) | `300f3344b50e75862d516b20767da8a3aa7724a5` | `6ac62da8f4453cb9423cfcc97f72080ad0f8d2b1` |

PRs [#36](https://github.com/x4g4p3x/lme-rs/pull/36) and
[#39](https://github.com/x4g4p3x/lme-rs/pull/39) closed as absorbed duplicates;
their requested array upgrades are included in #35.
