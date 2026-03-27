"""
Plotting mini-project: fit `lme_python` models and visualize fitted values,
residuals, and predictions from the fitted object (no hand-entered numbers).

Run from the repository root:

    python python/examples/plotting_demo/plot_demo.py

Requires matplotlib and numpy (install alongside polars):

    pip install matplotlib numpy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
except ImportError as e:
    print(f"Import error: {e}")
    print("Install: pip install matplotlib numpy polars")
    print("Build the extension: cd python && maturin develop --release")
    sys.exit(1)

try:
    import lme_python
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure lme_python is built and the virtual environment is active:")
    print("  cd python && maturin develop --release")
    sys.exit(1)

from figure_specs import (
    DPI,
    GROUSETICKS_SCATTER,
    MPL_ADJUST_GROUSE,
    MPL_ADJUST_RESIDUAL,
    MPL_ADJUST_SPAGHETTI,
    SLEEP_RESIDUAL,
    SLEEP_SPAGHETTI,
)
from paths import repo_root, tests_data


def _out_dir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _figures_data_dir() -> Path:
    d = Path(__file__).resolve().parent / "figures_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_sleepstudy_lmm() -> Tuple[Path, Path]:
    """LMM: residual vs fitted + Days × Reaction with population and conditional curves."""
    df = pl.read_csv(tests_data("sleepstudy.csv"))
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)

    fitted = np.asarray(fit.fitted, dtype=float)
    resid = np.asarray(fit.residuals, dtype=float)
    days = df["Days"].to_numpy()
    reaction = df["Reaction"].to_numpy()

    # --- Figure 1: Residual vs fitted (diagnostic) ---
    fig1, ax1 = plt.subplots(figsize=SLEEP_RESIDUAL)
    ax1.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax1.scatter(fitted, resid, alpha=0.35, s=12, c="tab:blue")
    ax1.set_xlabel("Fitted (conditional)")
    ax1.set_ylabel("Residual (y − fitted)")
    ax1.set_title("Sleepstudy LMM — residual vs fitted")
    fig1.subplots_adjust(**MPL_ADJUST_RESIDUAL)
    p1 = _out_dir() / "sleepstudy_residuals_vs_fitted.png"
    fig1.savefig(p1, dpi=DPI)
    plt.close(fig1)

    with open(_figures_data_dir() / "sleepstudy_residuals_py.json", "w", encoding="utf-8") as _fj:
        json.dump({"fitted": fitted.tolist(), "residual": resid.tolist()}, _fj)

    # --- Figure 2: Spaghetti + population & conditional mean for one subject ---
    subjects = df["Subject"].unique().to_list()
    grid_days = np.linspace(float(days.min()), float(days.max()), 40)

    # Population curve: Xβ on the grid (Subject column required; use any fixed subject)
    subj = str(subjects[0])
    new_pop = pl.DataFrame(
        {
            "Reaction": [0.0] * len(grid_days),
            "Days": grid_days,
            "Subject": [subj] * len(grid_days),
        }
    )
    line_pop = np.asarray(fit.predict(new_pop), dtype=float)

    # Conditional curve for the same subject (includes random effects)
    line_cond = np.asarray(fit.predict_conditional(new_pop, allow_new_levels=False), dtype=float)

    fig2, ax2 = plt.subplots(figsize=SLEEP_SPAGHETTI)
    for s in subjects[::3]:  # subsample for readability
        sub = df.filter(pl.col("Subject") == s)
        ax2.plot(
            sub["Days"].to_numpy(),
            sub["Reaction"].to_numpy(),
            color="0.75",
            alpha=0.35,
            linewidth=0.9,
        )
    ax2.plot(grid_days, line_pop, color="tab:blue", linewidth=2.2, label="Population (×β)")
    ax2.plot(grid_days, line_cond, color="tab:orange", linewidth=2.0, linestyle="--", label=f"Conditional (×β+Zb), Subject {subj}")
    ax2.scatter(days, reaction, s=12, alpha=0.25, c="black", zorder=0)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Reaction (ms)")
    ax2.set_title("Sleepstudy — observed trajectories and prediction curves")
    ax2.legend(loc="upper left")
    fig2.subplots_adjust(**MPL_ADJUST_SPAGHETTI)
    p2 = _out_dir() / "sleepstudy_days_reaction_curves.png"
    fig2.savefig(p2, dpi=DPI)
    plt.close(fig2)

    with open(_figures_data_dir() / "sleepstudy_curves_py.json", "w", encoding="utf-8") as _fj:
        json.dump(
            {
                "grid_days": grid_days.tolist(),
                "line_pop": line_pop.tolist(),
                "line_cond": line_cond.tolist(),
                "subject": subj,
            },
            _fj,
        )

    return p1, p2


def plot_grouseticks_glmm() -> Path:
    """Poisson GLMM: observed counts vs model-based expected counts (response scale)."""
    df = pl.read_csv(tests_data("grouseticks.csv"))
    fit = lme_python.glmer(
        "TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
        data=df,
        family_name="poisson",
        n_agq=1,
    )
    y = np.asarray(df["TICKS"].to_numpy(), dtype=float)
    mu = np.asarray(fit.predict_response(df), dtype=float)

    fig, ax = plt.subplots(figsize=GROUSETICKS_SCATTER)
    max_val = max(float(y.max()), float(mu.max()) * 1.05) + 1.0
    ax.plot([0, max_val], [0, max_val], color="0.5", linewidth=0.9, linestyle="--", label="y = μ̂")
    ax.scatter(mu, y, alpha=0.35, s=14, c="tab:green")
    ax.set_xlabel("Fitted expected count (population, response scale)")
    ax.set_ylabel("Observed TICKS")
    ax.set_title("Grouseticks Poisson GLMM — observed vs fitted")
    ax.legend(loc="upper left")
    fig.subplots_adjust(**MPL_ADJUST_GROUSE)
    outp = _out_dir() / "grouseticks_observed_vs_fitted.png"
    fig.savefig(outp, dpi=DPI)
    plt.close(fig)

    with open(_figures_data_dir() / "grouseticks_py.json", "w", encoding="utf-8") as _fj:
        json.dump({"mu": mu.tolist(), "y": y.tolist()}, _fj)

    return outp


def main() -> None:
    print(f"Repository root (fixtures): {repo_root()}")
    print()
    p1, p2 = plot_sleepstudy_lmm()
    p3 = plot_grouseticks_glmm()
    print("Wrote:")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")


if __name__ == "__main__":
    main()
