"""
glmer_grouseticks.py — Poisson GLMM with lme_python.glmer()
============================================================

Fits a Poisson mixed model on the grouseticks dataset:

    TICKS ~ YEAR + HEIGHT + (1 | BROOD)

Demonstrates count predictions, cluster-robust standard errors,
and allow_new_levels for out-of-sample groups.

Run from the repository root:

    python python/examples/glmer_grouseticks.py
"""

import os
import sys
import math

try:
    import polars as pl
    import lme_python
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure lme_python is built and the virtual environment is active:")
    print("  cd python && maturin develop --release")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")
GROUSETICKS = os.path.join(DATA_DIR, "grouseticks.csv")


def main():
    df = pl.read_csv(GROUSETICKS)
    print(f"Loaded grouseticks: {df.height} rows, "
          f"{df['BROOD'].n_unique()} broods\n")

    FORMULA = "TICKS ~ YEAR + HEIGHT + (1 | BROOD)"

    # ── Fit Poisson GLMM ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Poisson GLMM fit (log link)")
    print("=" * 60)
    fit = lme_python.glmer(FORMULA, data=df, family_name="poisson")
    print(fit.summary())

    print(f"  Converged : {fit.converged}")
    # Poisson has no dispersion parameter
    print(f"  σ²        : {fit.sigma2}  (None → no dispersion for Poisson)")

    # ── Fixed-effect coefficients ─────────────────────────────────────────────
    print("\n--- Fixed effects (log-count scale) ---")
    for name, coef, se in zip(fit.fixed_names, fit.coefficients, fit.std_errors):
        irr = math.exp(coef)  # incidence rate ratio
        print(f"  {name:<12}  coef={coef:8.4f}  SE={se:6.4f}  IRR={irr:.4f}")

    # ── Link-scale vs response-scale predictions ──────────────────────────────
    print("\n--- Predictions (first 6 rows) ---")
    eta = fit.predict(df)           # log(μ)
    mu_pop = fit.predict_response(df)   # exp(η) — expected count, population level
    mu_cond = fit.predict_conditional_response(df, allow_new_levels=False)

    print(f"  {'TICKS':>5}  {'η (log μ)':>10}  {'μ̂ pop':>8}  {'μ̂ cond':>8}")
    for i in range(6):
        y_val = df["TICKS"][i]
        print(f"  {y_val:>5}  {eta[i]:>10.4f}  {mu_pop[i]:>8.4f}  {mu_cond[i]:>8.4f}")

    # Verify: response = exp(link)
    assert abs(math.exp(eta[0]) - mu_pop[0]) < 1e-6
    print("\n  ✓ predict_response == exp(predict) verified")

    # ── Cluster-robust standard errors ───────────────────────────────────────
    print("\n--- Cluster-robust SEs (clustered by BROOD) ---")
    fit.with_robust_se(df, cluster_col="BROOD")
    robust_se = fit.robust_se
    robust_p  = fit.robust_p_values
    for name, se_r, p_r in zip(fit.fixed_names, robust_se, robust_p):
        print(f"  {name:<12}  robust SE={se_r:.4f}  p={p_r:.4f}")

    # ── Out-of-sample prediction with allow_new_levels ────────────────────────
    print("\n--- Prediction for an unseen brood (allow_new_levels=True) ---")
    new_brood = pl.DataFrame({
        "TICKS": [0.0],
        "YEAR": [2001.0],
        "HEIGHT": [450.0],
        "BROOD": ["UNSEEN_BROOD"],
    })
    mu_new_cond = fit.predict_conditional_response(new_brood, allow_new_levels=True)
    mu_new_pop  = fit.predict_response(new_brood)
    print(f"  Population prediction : {mu_new_pop[0]:.4f}")
    print(f"  Conditional prediction: {mu_new_cond[0]:.4f}")
    print("  (Should match because unseen brood gets zero RE contribution)")

    print("\nDone.")


if __name__ == "__main__":
    main()
