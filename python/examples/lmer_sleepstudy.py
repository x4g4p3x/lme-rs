"""
lmer_sleepstudy.py — Linear mixed model with lme_python.lmer()
==============================================================

Fits the classic sleepstudy model:

    Reaction ~ Days + (Days | Subject)

and walks through REML/ML estimation, Satterthwaite p-values,
Wald confidence intervals, and prediction.

Run from the repository root:

    python python/examples/lmer_sleepstudy.py
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")
SLEEPSTUDY = os.path.join(DATA_DIR, "sleepstudy.csv")


def main():
    df = pl.read_csv(SLEEPSTUDY)
    print(f"Loaded sleepstudy: {df.height} rows, {df['Subject'].n_unique()} subjects\n")

    FORMULA = "Reaction ~ Days + (Days | Subject)"

    # ── 1. REML fit (default, best for variance components) ──────────────────
    print("=" * 60)
    print("REML fit")
    print("=" * 60)
    reml_fit = lme_python.lmer(FORMULA, data=df, reml=True)
    print(reml_fit.summary())
    print(f"  Converged : {reml_fit.converged}")
    print(f"  σ²        : {reml_fit.sigma2:.4f}")
    print(f"  n         : {reml_fit.num_obs}")

    # ── 2. Satterthwaite p-values ─────────────────────────────────────────────
    print("\n--- Satterthwaite denominator df and p-values ---")
    reml_fit.with_satterthwaite(df)
    dfs = reml_fit.satterthwaite_dfs
    pvals = reml_fit.satterthwaite_p_values
    for name, df_val, p in zip(reml_fit.fixed_names, dfs, pvals):
        print(f"  {name:<20} df={df_val:7.2f}  p={p:.4f}")

    # ── 3. Wald confidence intervals ──────────────────────────────────────────
    print("\n--- 95% Wald confidence intervals ---")
    ci = reml_fit.confint(0.95)
    for name, (lo, hi) in zip(reml_fit.fixed_names, ci):
        print(f"  {name:<20} [{lo:8.4f}, {hi:8.4f}]")

    # ── 4. ML fit (needed for likelihood ratio tests) ─────────────────────────
    print("\n" + "=" * 60)
    print("ML fit (reml=False)  — for model comparison")
    print("=" * 60)
    ml_fit = lme_python.lmer(FORMULA, data=df, reml=False)
    print(f"  AIC: {ml_fit.aic:.2f}  BIC: {ml_fit.bic:.2f}  logLik: {ml_fit.log_likelihood:.2f}")

    # ── 5. Random effects ─────────────────────────────────────────────────────
    print("\n--- Random effects (first 5 rows) ---")
    ranef = reml_fit.ranef  # list of (Grouping, Group, Effect, Value)
    print(f"  {'Grouping':<10} {'Group':<8} {'Effect':<16} {'Value':>8}")
    for row in ranef[:5]:
        grouping, group, effect, value = row
        print(f"  {grouping:<10} {group:<8} {effect:<16} {value:>8.4f}")

    # ── 6. Population-level vs conditional predictions ────────────────────────
    print("\n--- Predictions for Subject 308 at Days 0, 5, 10 ---")
    newdata = pl.DataFrame({
        "Reaction": [0.0, 0.0, 0.0],
        "Days": [0.0, 5.0, 10.0],
        "Subject": ["308", "308", "308"],
    })
    pop_preds = reml_fit.predict(newdata)
    cond_preds = reml_fit.predict_conditional(newdata, allow_new_levels=False)

    print(f"  {'Days':>4}  {'Population':>12}  {'Conditional':>12}")
    for day, pop, cond in zip([0, 5, 10], pop_preds, cond_preds):
        print(f"  {day:>4}  {pop:>12.4f}  {cond:>12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
