"""
Build ``figures_overlay/*.png`` from exported Python JSON + R CSV so both layers share
one coordinate system (true alignment). Used by ``compare_plots.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from figure_specs import (
    DPI,
    GROUSETICKS_SCATTER,
    MPL_ADJUST_GROUSE,
    MPL_ADJUST_RESIDUAL,
    MPL_ADJUST_SPAGHETTI,
    SLEEP_RESIDUAL,
    SLEEP_SPAGHETTI,
)
from paths import tests_data

# Distinct colors: Python vs R (overlay only; side-by-side figures stay unchanged).
CLR_PY = "#1f77b4"
CLR_R = "#d95f02"


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def overlay_residual(here: Path, out_path: Path) -> bool:
    py_j = here / "figures_data" / "sleepstudy_residuals_py.json"
    r_csv = here / "figures_data" / "sleepstudy_residuals_r.csv"
    if not py_j.is_file() or not r_csv.is_file():
        return False
    py = _read_json(py_j)
    rtab = pl.read_csv(r_csv)
    f_py, r_py = np.asarray(py["fitted"]), np.asarray(py["residual"])
    f_r = rtab["fitted"].to_numpy()
    r_r = rtab["residual"].to_numpy()

    fig, ax = plt.subplots(figsize=SLEEP_RESIDUAL)
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)
    ax.scatter(f_py, r_py, s=14, alpha=0.45, c=CLR_PY, label="lme_python", linewidths=0)
    ax.scatter(f_r, r_r, s=12, alpha=0.55, c=CLR_R, label="lme4 (R)", marker="x", linewidths=0.6)
    ax.set_xlabel("Fitted (conditional)")
    ax.set_ylabel("Residual (y − fitted)")
    ax.set_title("Sleepstudy LMM — residual vs fitted (overlay, shared axes)")
    ax.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(**MPL_ADJUST_RESIDUAL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def overlay_spaghetti(here: Path, out_path: Path) -> bool:
    py_j = here / "figures_data" / "sleepstudy_curves_py.json"
    r_csv = here / "figures_data" / "sleepstudy_curves_r.csv"
    if not py_j.is_file() or not r_csv.is_file():
        return False
    py = _read_json(py_j)
    rtab = pl.read_csv(r_csv)
    grid_py = np.asarray(py["grid_days"])
    pop_py = np.asarray(py["line_pop"])
    cond_py = np.asarray(py["line_cond"])
    grid_r = rtab["Days"].to_numpy()
    pop_r = rtab["line_pop"].to_numpy()
    cond_r = rtab["line_cond"].to_numpy()

    df = pl.read_csv(tests_data("sleepstudy.csv"))
    days = df["Days"].to_numpy()
    reaction = df["Reaction"].to_numpy()
    subjects = df["Subject"].unique().to_list()

    fig, ax = plt.subplots(figsize=SLEEP_SPAGHETTI)
    for i in range(0, len(subjects), 3):
        s = subjects[i]
        sub = df.filter(pl.col("Subject") == s)
        ax.plot(
            sub["Days"].to_numpy(),
            sub["Reaction"].to_numpy(),
            color="0.75",
            alpha=0.35,
            linewidth=0.9,
        )
    ax.plot(grid_py, pop_py, color=CLR_PY, linewidth=2.2, linestyle="-", label="Pop. lme_python")
    ax.plot(grid_r, pop_r, color=CLR_R, linewidth=2.0, linestyle="-", label="Pop. lme4")
    ax.plot(grid_py, cond_py, color=CLR_PY, linewidth=1.8, linestyle="--", label="Cond. lme_python")
    ax.plot(grid_r, cond_r, color=CLR_R, linewidth=1.8, linestyle="--", label="Cond. lme4")
    ax.scatter(days, reaction, s=10, alpha=0.2, c="black", zorder=0)
    ax.set_xlabel("Days")
    ax.set_ylabel("Reaction (ms)")
    ax.set_title("Sleepstudy — trajectories + predictions (overlay, shared axes)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.subplots_adjust(**MPL_ADJUST_SPAGHETTI)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def overlay_grouse(here: Path, out_path: Path) -> bool:
    py_j = here / "figures_data" / "grouseticks_py.json"
    r_csv = here / "figures_data" / "grouseticks_r.csv"
    if not py_j.is_file() or not r_csv.is_file():
        return False
    py = _read_json(py_j)
    rtab = pl.read_csv(r_csv)
    mu_py = np.asarray(py["mu"])
    y_py = np.asarray(py["y"])
    mu_r = rtab["mu"].to_numpy()

    max_val = max(float(y_py.max()), float(mu_py.max()), float(mu_r.max())) * 1.05 + 1.0

    fig, ax = plt.subplots(figsize=GROUSETICKS_SCATTER)
    ax.plot([0, max_val], [0, max_val], color="0.5", linewidth=0.9, linestyle="--", zorder=0)
    ax.scatter(mu_py, y_py, alpha=0.4, s=16, c=CLR_PY, label="lme_python", linewidths=0)
    ax.scatter(mu_r, y_py, alpha=0.45, s=14, c=CLR_R, label="lme4 (R)", marker="x", linewidths=0.5)
    ax.set_xlabel("Fitted expected count (population, response scale)")
    ax.set_ylabel("Observed TICKS")
    ax.set_title("Grouseticks Poisson GLMM — observed vs fitted (overlay, shared axes)")
    ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(**MPL_ADJUST_GROUSE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def write_all_numeric_overlays(here: Path) -> Tuple[bool, bool, bool]:
    """Return (ok_residual, ok_spaghetti, ok_grouse)."""
    od = here / "figures_overlay"
    a = overlay_residual(here, od / "sleepstudy_residuals_vs_fitted.png")
    b = overlay_spaghetti(here, od / "sleepstudy_days_reaction_curves.png")
    c = overlay_grouse(here, od / "grouseticks_observed_vs_fitted.png")
    return a, b, c
