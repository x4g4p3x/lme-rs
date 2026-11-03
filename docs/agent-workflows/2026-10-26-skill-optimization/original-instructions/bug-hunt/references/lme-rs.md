# lme-rs correctness hunts

Use this reference only in `lme-rs`, a Rust statistical library with PyO3/maturin
bindings. Read the checkout's current `AGENTS.md` and `Taskfile.yml`; they govern
validation and may evolve. There is no application server to start.

## Choose a focused path

| Suspected behavior | Implementation and evidence |
|---|---|
| Fit contracts and reusable responses | `src/model.rs`, `src/prepared.rs`, `src/lib.rs` |
| Formula source columns and design construction | `src/formula.rs`, `src/model_matrix.rs`, `src/basis.rs` |
| OLS and sequential hypotheses | `src/ols.rs`, `src/anova_contrasts.rs` |
| Marginal means and reference grids | `src/emmeans.rs`, prediction/design reconstruction |
| Contrasts and degrees of freedom | `src/contrast.rs`, `src/ddf.rs`, Satterthwaite/Kenward-Roger modules |
| LMM/GLMM fitting | `src/math.rs`, `src/optimizer.rs`, `src/glmm_math.rs`, `src/family.rs` |
| Bindings and independent evidence | `python/src/lib.rs`, `python/tests/`, `tests/`, `tests/data/`, `comparisons/` |

Search existing regressions before adding new ones. In particular,
`tests/test_bug_hunt_postfit.rs` illustrates the unit-invariance and reference-grid
tests from which this workflow was developed; its passing cases are examples,
not a reason to stop investigating related contracts.

## Numerical and statistical probes

- Rescale a predictor over representable small and large units. Fitted values and
  residual variance should agree; coefficients and standard errors transform with
  units; equivalent hypothesis F statistics and p-values should agree. Keep a
  genuinely collinear control. Beyond representable covariance, check explicit errors.
- Verify a Type I ANOVA term's sum of squares independently through the reduction
  in residual sum of squares between the corresponding nested OLS models.
- For marginal means, distinguish fixed-effect source covariates from responses,
  unused data columns, and random-only variables. Check irrelevant/all-null columns
  do not alter fixed-effect inference and invalid `at` references are rejected.
- Exercise source tracking through transforms, bases, colon-only interactions,
  star interactions, and dot expansion. Compare a simple grid's marginal means
  against direct predictions at the same covariate values when statistically valid.
- For inference reconstruction, check fitted weights, offsets, contrasts, and basis
  metadata are preserved. Do not compare models with different row selection,
  family/link, weights, offsets, or ML/REML settings as if they were equivalent.

Do not apply these identities outside their conditions: covariance estimates differ
with model assumptions, and marginal means may average nuisance-factor levels.

## Validation and operational details

- Start with `cargo test --locked --test <target> <filter>` or
  `cargo test --locked --lib <filter>`; verify a nonzero test count.
- Register each new integration test file in `tests/ci_consolidated.rs`.
- For Rust changes, the current minimum is `task lint`, `task test:fast`, and
  affected integration tests. Use `task rust` for cross-module/public-API changes.
  It includes Rust lint, compilation, tests, doctests, and documentation; add
  `task lint:python` for the Ruff component of `task lint`.
- When linking is expensive on Windows/macOS, the current equivalent broad slice
  is `task lint:rust`, `task check`, `task test:consolidated`, and
  `python scripts/ci/lme_ci.py doc`. Confirm the current guide before substituting.
- Binding changes require `task lint:python` and `task python`. Documentation or
  portable-example changes require `task docs:check`; install/example behavior
  changes also require `task consumer:smoke`.
- Throughput-path changes require `OPTIMIZATION.md` and applicable fair-harness
  cases alongside Rust checks. Do not claim performance from correctness tests.
- Add user-visible fixes under `Unreleased` in `CHANGELOG.md`. Do not adjust
  completion scores or locked scope strings merely because a focused bug is fixed.
- Serialize Cargo work sharing `target`. A slow static-MKL build is not a failing
  test. For Windows linker failures, inspect disk space and active builds before
  considering a narrowly scoped cleanup.
- Locate installed tools before reinstalling them. If Task is unavailable, use
  the checked-in `scripts/ci/lme_ci.py` mapping. Avoid routine `uv sync`: it can
  remove the editable extension; `task python` restores the supported test setup.
- Keep hooks enabled. Pre-push preflight includes audits and metadata checks.
  Ordinary branch pushes do not trigger hosted validation; PRs do. Verify fresh
  PR head checks, including `pip-audit`, before an authorized merge, and report
  heavy ignored cases and optional-tool skips honestly.

Use the live repository default branch for delivery rather than assuming its name.
This skill does not authorize Git publication, worker delegation, or releases.
