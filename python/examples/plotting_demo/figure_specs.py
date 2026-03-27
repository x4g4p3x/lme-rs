"""
Shared figure layout for Python (matplotlib), R (png), and overlay comparison.

`plot_r.R` uses the same pixel sizes (DPI × inches) so blended PNGs align.
"""

from __future__ import annotations

DPI = 150

# Matplotlib figsize (width, height) in inches — keep in sync with `plot_r.R`.
SLEEP_RESIDUAL = (6.5, 5.0)
SLEEP_SPAGHETTI = (7.5, 5.0)
GROUSETICKS_SCATTER = (6.5, 5.0)

OUTPUT_FILES = (
    "sleepstudy_residuals_vs_fitted.png",
    "sleepstudy_days_reaction_curves.png",
    "grouseticks_observed_vs_fitted.png",
)

# Matplotlib subplot margins (fraction of figure) — keep R `par(mar=...)` in `plot_r.R` in the same spirit.
MPL_ADJUST_RESIDUAL = {"left": 0.14, "right": 0.96, "bottom": 0.14, "top": 0.88}
MPL_ADJUST_SPAGHETTI = {"left": 0.12, "right": 0.97, "bottom": 0.12, "top": 0.78}
MPL_ADJUST_GROUSE = {"left": 0.14, "right": 0.96, "bottom": 0.14, "top": 0.88}

# Fraction of each edge trimmed before overlay (left, top, right, bottom), after R is resized to Python.
# Focuses on the data panel; tune if plot margins change.
OVERLAY_CROP_FRAC: dict[str, tuple[float, float, float, float]] = {
    "sleepstudy_residuals_vs_fitted.png": (0.125, 0.105, 0.035, 0.115),
    "sleepstudy_days_reaction_curves.png": (0.135, 0.165, 0.055, 0.115),
    "grouseticks_observed_vs_fitted.png": (0.125, 0.105, 0.035, 0.115),
}

# RGB multipliers for overlay (Python vs R) so both layers stay visible when aligned.
OVERLAY_PY_RGB = (0.32, 0.52, 1.0)
OVERLAY_R_RGB = (1.0, 0.38, 0.12)

# After cropping, search integer shifts of the R panel in [-N, N] px to minimize L1(luma) vs Python.
OVERLAY_MAX_SHIFT_PX = 10


def px_size(figsize_in: tuple[float, float]) -> tuple[int, int]:
    w, h = figsize_in
    return (int(w * DPI), int(h * DPI))
