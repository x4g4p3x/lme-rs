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
    import lme_python
    import polars as pl
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
    m0 = lme_python.lmer("Reaction ~ 1 + (1 | Subject)", data=df, reml=False)
    m1 = lme_python.lmer("Reaction ~ Days + (1 | Subject)", data=df, reml=False)
    m2 = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=False)

    print(f"  M0: AIC={m0.aic:.2f}  logLik={m0.log_likelihood:.2f}")
    print(f"  M1: AIC={m1.aic:.2f}  logLik={m1.log_likelihood:.2f}")
    print(f"  M2: AIC={m2.aic:.2f}  logLik={m2.log_likelihood:.2f}")

    # ── 1. Likelihood ratio test: M0 vs M1 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("LRT: M0 vs M1  (does Days as a fixed effect improve fit?)")
    print("=" * 60)
    res01 = lme_python.anova(m0, m1)
    print(f"  Model 0 ({res01.n_params_0} params): deviance = {res01.deviance_0:.4f}")
    print(f"  Model 1 ({res01.n_params_1} params): deviance = {res01.deviance_1:.4f}")
    stars01 = fmt_stars(res01.p_value)
    print(f"  χ²({res01.df}) = {res01.chi_sq:.4f},  p = {res01.p_value:.4e}  {stars01}")
    sig01 = "significantly" if res01.p_value < 0.05 else "does not significantly"
    print(f"  → Adding Days as a fixed effect {sig01} improves fit.")

    # ── 2. Likelihood ratio test: M1 vs M2 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("LRT: M1 vs M2  (does a random slope for Days improve fit?)")
    print("=" * 60)
    res12 = lme_python.anova(m1, m2)
    print(f"  Model 0 ({res12.n_params_0} params): deviance = {res12.deviance_0:.4f}")
    print(f"  Model 1 ({res12.n_params_1} params): deviance = {res12.deviance_1:.4f}")
    stars12 = fmt_stars(res12.p_value)
    print(f"  χ²({res12.df}) = {res12.chi_sq:.4f},  p = {res12.p_value:.4e}  {stars12}")
    sig12 = "significantly" if res12.p_value < 0.05 else "does not significantly"
    print(f"  → Adding random slopes {sig12} improves fit.")

    # ── 3. Type III ANOVA table for the full model ────────────────────────────
    print("\n" + "=" * 60)
    print("Type III ANOVA table for M2  (Satterthwaite ddf)")
    print("=" * 60)
    # Use a REML fit for reporting; Satterthwaite requires the data
    m2_reml = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    m2_reml.with_satterthwaite(df)

    tab = m2_reml.anova("satterthwaite")
    print(f"  Method: {tab.method}")
    print(f"\n  {'Term':<16}  {'NumDf':>5}  {'DenDf':>8}  {'F':>8}  {'Pr(>F)':>10}  Sig")
    for term, nd, dd, fv, pv in zip(tab.terms, tab.num_df, tab.den_df, tab.f_value, tab.p_value):
        print(f"  {term:<16}  {nd:>5.0f}  {dd:>8.2f}  {fv:>8.4f}  {pv:>10.4e}  {fmt_stars(pv)}")

    print("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    # ── 4. AIC-based model ranking ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AIC ranking (lower is better)")
    print("=" * 60)
    models = [
        ("M0: intercept only", m0),
        ("M1: fixed Days", m1),
        ("M2: fixed+random Days", m2),
    ]
    models_sorted = sorted(models, key=lambda x: x[1].aic)
    best_aic = models_sorted[0][1].aic
    print(f"  {'Model':<28}  {'AIC':>8}  {'ΔAIC':>7}")
    for label, m in models_sorted:
        print(f"  {label:<28}  {m.aic:>8.2f}  {m.aic - best_aic:>7.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
