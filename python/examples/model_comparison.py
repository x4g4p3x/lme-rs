"""
model_comparison.py — Likelihood ratio tests and Type III ANOVA
===============================================================

Demonstrates:
  1. Nested model comparison with lme_python.anova() (LRT)
  2. Type III fixed-effects ANOVA table via fit.anova()
  3. Model selection with AIC / BIC

Run from the repository root:

    python python/examples/model_comparison.py
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


def fmt_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def main():
    df = pl.read_csv(SLEEPSTUDY)
    print(f"Loaded sleepstudy: {df.height} rows\n")

    # ── Fit three nested models with ML (required for LRT) ───────────────────
    # M0: random intercept only
    # M1: random intercept, fixed slope for Days
    # M2: random intercept + random slope (full model)
    print("Fitting nested models with ML (reml=False)...")
    m0 = lme_python.lmer("Reaction ~ 1 + (1 | Subject)",            data=df, reml=False)
    m1 = lme_python.lmer("Reaction ~ Days + (1 | Subject)",          data=df, reml=False)
    m2 = lme_python.lmer("Reaction ~ Days + (Days | Subject)",       data=df, reml=False)

    print(f"  M0: AIC={m0.aic:.2f}  logLik={m0.log_likelihood:.2f}")
    print(f"  M1: AIC={m1.aic:.2f}  logLik={m1.log_likelihood:.2f}")
    print(f"  M2: AIC={m2.aic:.2f}  logLik={m2.log_likelihood:.2f}")

    # ── 1. Likelihood ratio test: M0 vs M1 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("LRT: M0 vs M1  (does Days as a fixed effect improve fit?)")
    print("=" * 60)
    res01 = lme_python.anova(m0, m1)
    n0, n1, dev0, dev1, chi_sq, df_diff, p_val, fml0, fml1 = res01
    print(f"  Model 0 ({n0} params): deviance = {dev0:.4f}")
    print(f"  Model 1 ({n1} params): deviance = {dev1:.4f}")
    print(f"  χ²({df_diff}) = {chi_sq:.4f},  p = {p_val:.4e}  {fmt_stars(p_val)}")
    print(f"  → Adding Days as a fixed effect {'significantly' if p_val < 0.05 else 'does not significantly'} improves fit.")

    # ── 2. Likelihood ratio test: M1 vs M2 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("LRT: M1 vs M2  (does a random slope for Days improve fit?)")
    print("=" * 60)
    res12 = lme_python.anova(m1, m2)
    n0, n1, dev0, dev1, chi_sq, df_diff, p_val, fml0, fml1 = res12
    print(f"  Model 0 ({n0} params): deviance = {dev0:.4f}")
    print(f"  Model 1 ({n1} params): deviance = {dev1:.4f}")
    print(f"  χ²({df_diff}) = {chi_sq:.4f},  p = {p_val:.4e}  {fmt_stars(p_val)}")
    print(f"  → Adding random slopes {'significantly' if p_val < 0.05 else 'does not significantly'} improves fit.")

    # ── 3. Type III ANOVA table for the full model ────────────────────────────
    print("\n" + "=" * 60)
    print("Type III ANOVA table for M2  (Satterthwaite ddf)")
    print("=" * 60)
    # Use a REML fit for reporting; Satterthwaite requires the data
    m2_reml = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    m2_reml.with_satterthwaite(df)

    terms, num_df, den_df, f_val, p_val_anova, method = m2_reml.anova("satterthwaite")
    print(f"  Method: {method}")
    print(f"\n  {'Term':<16}  {'NumDf':>5}  {'DenDf':>8}  {'F':>8}  {'Pr(>F)':>10}  Sig")
    for term, nd, dd, fv, pv in zip(terms, num_df, den_df, f_val, p_val_anova):
        print(f"  {term:<16}  {nd:>5.0f}  {dd:>8.2f}  {fv:>8.4f}  {pv:>10.4e}  {fmt_stars(pv)}")

    print("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    # ── 4. AIC-based model ranking ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AIC ranking (lower is better)")
    print("=" * 60)
    models = [
        ("M0: intercept only",        m0),
        ("M1: fixed Days",            m1),
        ("M2: fixed+random Days",     m2),
    ]
    models_sorted = sorted(models, key=lambda x: x[1].aic)
    best_aic = models_sorted[0][1].aic
    print(f"  {'Model':<28}  {'AIC':>8}  {'ΔAIC':>7}")
    for label, m in models_sorted:
        print(f"  {label:<28}  {m.aic:>8.2f}  {m.aic - best_aic:>7.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
