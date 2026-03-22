"""
lm_basics.py — Fixed-effects linear regression with lme_python.lm()
====================================================================

Demonstrates the formula-string entry point for ordinary least squares.
Run from the repository root:

    python python/examples/lm_basics.py

Or from the python/ directory:

    python examples/lm_basics.py
"""

import os
import sys

try:
    import polars as pl
    import lme_python
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure lme_python is built and the virtual environment is active:")
    print("  cd python && maturin develop --release")
    sys.exit(1)

# Locate the data file relative to this script so the example works
# regardless of which directory it is run from.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")
SLEEPSTUDY = os.path.join(DATA_DIR, "sleepstudy.csv")


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    df = pl.read_csv(SLEEPSTUDY)
    print(f"Loaded sleepstudy: {df.height} rows × {df.width} columns\n")

    # ── 1. Simple regression: Reaction ~ Days ────────────────────────────────
    print("=" * 60)
    print("Model 1: Reaction ~ Days  (intercept + slope, ignoring Subject)")
    print("=" * 60)

    fit = lme_python.lm("Reaction ~ Days", data=df)
    print(fit.summary())

    intercept, slope = fit.coefficients
    print(f"  Intercept : {intercept:.4f}")
    print(f"  Days      : {slope:.4f}")

    se = fit.std_errors
    print(f"  SE(Intercept): {se[0]:.4f}")
    print(f"  SE(Days)     : {se[1]:.4f}")

    print(f"\n  AIC: {fit.aic:.2f}  BIC: {fit.bic:.2f}  logLik: {fit.log_likelihood:.2f}")
    print(f"  n = {fit.num_obs}")

    # ── 2. Confidence intervals ───────────────────────────────────────────────
    print("\n--- Wald 95% confidence intervals ---")
    ci = fit.confint(0.95)
    names = fit.fixed_names
    for name, (lo, hi) in zip(names, ci):
        print(f"  {name:<20} [{lo:8.4f}, {hi:8.4f}]")

    # ── 3. Intercept-only model ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Model 2: Reaction ~ 1  (global mean only)")
    print("=" * 60)

    fit0 = lme_python.lm("Reaction ~ 1", data=df)
    print(f"  Global mean estimate: {fit0.coefficients[0]:.4f}")
    print(f"  (True mean: {df['Reaction'].mean():.4f})")

    # ── 4. Predictions ────────────────────────────────────────────────────────
    print("\n--- Population-level predictions for Days 0, 5, 10 ---")
    newdata = pl.DataFrame({"Reaction": [0.0, 0.0, 0.0], "Days": [0.0, 5.0, 10.0]})
    preds = fit.predict(newdata)
    for day, pred in zip([0, 5, 10], preds):
        print(f"  Days={day:2d}  predicted Reaction = {pred:.2f} ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
