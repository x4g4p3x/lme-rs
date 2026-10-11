# Initial deterministic results: 0.2.7

On 11 October 2026, a fresh Windows x64 CPython 3.11.15 environment installed
the published PyPI 0.2.7 wheel using the frozen uv lock. **All eight cases and
533 scalar comparisons passed.** This is scoped deterministic verification;
the correlated fitted-value checks are not independent experiments.

| Contract | Comparisons | Result | Largest absolute error / declared tolerance |
|:---|---:|:---|---:|
| Rational OLS | 6 | Passed | 0.00000111 |
| NIST Longley | 15 | Passed | 0.001242 |
| NIST Wampler1 | 7 | Passed | 0.05731 |
| Balanced LMM, ML | 8 | Passed | 0.2147 |
| Balanced LMM, REML | 10 | Passed | 0.2093 |
| Bates sleepstudy | 189 | Passed | 0.1237 |
| Kuznetsova TVbo | 245 | Passed | 0.04877 |
| Halekoh beets | 53 | Passed | 0.03658 |

A ratio of one is the contract limit, with tolerance defined as
`atol + rtol * abs(reference)`. Definitions, sources, controls and limits are
in [cases.json](cases.json); commands and provenance policy are in the
[baseline guide](README.md). The Python environment used polars 2.0.0 and
polars-runtime-32 2.0.0. R 4.6.1 reference-package and loaded-namespace versions
matched the registry exactly. All paper models used REML; the author's beets
ML models are internally refitted with REML for KR, and the candidate matches
that inferential fit.

## Provenance

- Release commit: `0b4760fcc5a7558a60ec738401f326a793cb6b1d`, tag `v0.2.7`.
- Report generated: `2026-10-11T12:34:57` UTC.
- Run ID: `20261011T123452Z-c613ae6a`.
- Full report SHA-256: `61a3a061ad52758a822937f8e94483a1a3792f6e82d9dfd4c39ed139dc147024`.
- Native module SHA-256: `fd754e69381368bf8ccd1105f71aca3fdbca740874ad5bd9c316262699edac9f`.
- Harness SHA-256: `fdc973f22486a216e4fac56d803c45382884dceb641878940d9919930fe6805c`.
- Registry SHA-256: `39b3b2c1c2f010c73bfcaee0345c3c1ce74063a8b91987b84dc7e709d24e8b01`.
- Python lock SHA-256: `163c1e5cede8fd8f4e4bb6e1cbd8b9bc8480753b47c741f806e631585a65b79d`.
- R adapter SHA-256: `2573cc062543c9799496f0de27dd2947ce5df3f0c7438ff6786ce0c710d34bab`.

The full report, original downloads, exported inputs and R session are retained
in the ignored run directory documented in the guide. This public summary
does not redistribute the package datasets or downloaded author scripts.
Long-term archival of a shareable evidence bundle and a portable, restorable R
environment remain reproducibility work. The exact numerical study has run
on one OS/interpreter; the separate release CI checks installation and existing
tests across the supported matrix.

## Defects exposed before freezing the engine

The worked examples produced actual failures in the pre-release engine:

1. TVbo's three unequal random-intercept blocks caused an indexing panic.
   A small unequal-block regression independently compared the solver with
   fresh sparse-profile evaluations; rectangular solves and fill handling
   were corrected.
2. TVbo Type III main effects were evaluated at the reference factor level.
   A hand-computed 2-by-2 OLS control demonstrated the wrong hypothesis.
   Model-aware categorical averaging now preserves treatment/sum and level-order
   invariance.
3. The beets harvest KR comparison collapsed near two denominator degrees of
   freedom. An independent paired-block identity and direct moment control
   reproduced the removable singularity. The scalar analytic limit now returns
   finite df and F values.

These fixes are in the release and [changelog](../CHANGELOG.md), with durable
Rust/Python regressions. Agreement with these selected examples does not
validate every method or the original papers' scientific conclusions.

## Remaining scientific gate

The [LMM calibration protocol](../docs/LMM_CALIBRATION_PROTOCOL.md) is ready
for an independent statistician to review. Recruitment is planned; no review,
pilot or confirmatory calibration campaign has occurred. GLMM, NLMM, general
variance-component uncertainty and applied-study reproduction remain outside
this initial completed numerical scope.
