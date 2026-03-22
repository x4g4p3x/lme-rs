"""
glmer_cbpp.py — Binomial GLMM with lme_python.glmer()
======================================================

Fits a binomial mixed model on the CBPP (contagious bovine
pleuropneumonia) dataset:

    y ~ period2 + period3 + period4 + (1 | herd)

Demonstrates probability predictions, response-scale vs link-scale
predictions, and parametric simulation.

Run from the repository root:

    python python/examples/glmer_cbpp.py
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
CBPP = os.path.join(DATA_DIR, "cbpp_binary.csv")


def main():
    df = pl.read_csv(CBPP)
    print(f"Loaded cbpp_binary: {df.height} rows\n")

    FORMULA = "y ~ period2 + period3 + period4 + (1 | herd)"

    # ── Fit binomial GLMM ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Binomial GLMM fit (logit link)")
    print("=" * 60)
    fit = lme_python.glmer(FORMULA, data=df, family_name="binomial")
    print(fit.summary())

    print(f"  Converged : {fit.converged}")
    print(f"  n         : {fit.num_obs}")
    print(f"  AIC       : {fit.aic:.2f}")

    # ── Fixed-effect coefficients ─────────────────────────────────────────────
    print("\n--- Fixed effects (log-odds scale) ---")
    for name, coef, se in zip(fit.fixed_names, fit.coefficients, fit.std_errors):
        # Convert log-odds to odds ratio for interpretability
        or_ = math.exp(coef)
        print(f"  {name:<16}  coef={coef:7.4f}  SE={se:6.4f}  OR={or_:.3f}")

    # ── Response-scale predictions (probabilities) ─────────────────────────────
    print("\n--- Predicted probabilities (first 8 observations) ---")
    # Link scale (linear predictor η)
    eta = fit.predict(df)
    # Response scale (probability = logistic(η))
    probs = fit.predict_response(df)
    # Conditional (includes herd random effect)
    probs_cond = fit.predict_conditional_response(df, allow_new_levels=False)

    print(f"  {'y':>3}  {'η (link)':>10}  {'P̂ (pop)':>10}  {'P̂ (cond)':>10}")
    for i in range(8):
        y_val = df["y"][i]
        print(f"  {y_val:>3.0f}  {eta[i]:>10.4f}  {probs[i]:>10.4f}  {probs_cond[i]:>10.4f}")

    # Sanity check: all probabilities should be in [0, 1]
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of [0,1]!"
    print("\n  ✓ All predicted probabilities are in [0, 1]")

    # ── Parametric simulation ─────────────────────────────────────────────────
    print("\n--- Parametric simulation (3 draws, showing first obs) ---")
    sims = fit.simulate(3)
    print(f"  Number of simulations : {len(sims)}")
    print(f"  Observations per sim  : {len(sims[0])}")
    for k, sim in enumerate(sims):
        # Binomial draws should be 0 or 1
        assert all(x in (0.0, 1.0) for x in sim), f"Simulation {k} has non-binary values!"
        print(f"  sim[{k}][0:5] = {list(sim[:5])}")
    print("  ✓ All simulated values are binary (0 or 1)")

    print("\nDone.")


if __name__ == "__main__":
    main()
