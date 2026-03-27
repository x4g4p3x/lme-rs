# Plotting mini-project

This folder complements [`../verification_project/`](../verification_project/): instead of asserting reference scalars, it **fits** mixed models with `lme_python`, draws **matplotlib** figures from the fitted object, optionally reproduces the same views in **R (lme4)**, and produces **comparison** and **overlay** images.

| Goal | Where it lives |
|------|----------------|
| Exercise the Python API on `tests/data/` | `plot_demo.py` → `figures/` |
| Same fits in R | `plot_r.R` → `figures_r/` |
| Side-by-side PNGs (unchanged styling) | `compare_plots.py` → `figures_compare/` |
| **True** Python vs R overlay (shared axes, two colors) | `compare_plots.py` + `numeric_overlay.py` → `figures_overlay/` |

Exports used for the numeric overlay live in **`figures_data/`** (JSON from Python, CSV from R). All generated paths are gitignored.

---

## Quick start (full pipeline)

From the **repository root**, with `lme_python` built and R + **lme4** available:

```bash
python python/examples/plotting_demo/run_all.py
```

This runs, in order: **`plot_demo.py`** → **`plot_r.R`** → **`compare_plots.py`**. It prefers **`python/.venv`** for Python steps when that venv exists (so a global `python` launcher can still find `lme_python` installed by `maturin develop`).

---

## Setup (Python)

```bash
cd python
pip install matplotlib numpy pillow
maturin develop --release
```

**CPython 3.14:** PyO3 may reject 3.14 until explicitly supported. Use the stable ABI (see [`PYTHON_GUIDE.md`](../../PYTHON_GUIDE.md)):

```powershell
cd python
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"
maturin develop --release
```

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cd python && maturin develop --release
```

---

## Setup (R, for parity plots)

Install [R](https://www.r-project.org/) (e.g. Windows: `winget install RProject.R`). Install **lme4** once:

```r
install.packages("lme4")
```

`run_all.py` and `find_rscript.py` locate **`Rscript`** via `PATH`, the Windows registry (`SOFTWARE\R-core\R` → `InstallPath`), and `Program Files\R\R-*\bin`, so you do not have to add R to `PATH` manually.

---

## Run scripts individually

**Python figures only** (repo root):

```bash
python python/examples/plotting_demo/plot_demo.py
```

Writes **`figures/`** and **`figures_data/*_py.json`** for overlay.

**R figures only** (repo root):

```bash
Rscript python/examples/plotting_demo/plot_r.R
# or, if cwd differs:
Rscript python/examples/plotting_demo/plot_r.R "C:/path/to/lme-rs"
```

Writes **`figures_r/`** and **`figures_data/*_r.csv`**.

**Compare** (requires **`figures/`** and **`figures_r/`**; overlay needs complete **`figures_data/`** from both steps above):

```bash
python python/examples/plotting_demo/compare_plots.py
```

Optional debug: **`--overlay-raw`** adds a full-frame 50/50 alpha blend (usually misaligned).

---

## Output directories

| Directory | Contents |
|-----------|----------|
| `figures/` | Matplotlib PNGs from `lme_python`. |
| `figures_r/` | Base-graphics PNGs from **lme4** (same pixel sizes as `figure_specs.py` / `plot_r.R`). |
| `figures_data/` | JSON + CSV for numeric overlay (fitted values, curves, grouseticks `mu`/`y`). |
| `figures_compare/` | Side-by-side panels: lme_python \| lme4 (original appearance, R resized to match). |
| `figures_overlay/` | **Numeric overlay**: one matplotlib plot per file, **shared axes** — Python **blue**, R **orange** (`×` where appropriate). |
| `figures_overlay_raster/` | Only if `figures_data/` is incomplete: crop + shift + tinted raster blend (best-effort). |

---

## Models and API mapping

| Output basename | Model | Python | R (lme4) |
|-----------------|--------|--------|----------|
| `sleepstudy_residuals_vs_fitted.png` | LMM | `fitted`, `residuals` | `fitted()`, `residuals()` |
| `sleepstudy_days_reaction_curves.png` | LMM | `predict`, `predict_conditional` | `predict(..., re.form = NA / NULL)` |
| `grouseticks_observed_vs_fitted.png` | Poisson GLMM | `predict_response` | `predict(..., type = "response", re.form = NA)` |

Formulas match the verification project fixtures: sleepstudy `Reaction ~ Days + (Days | Subject)` (REML), grouseticks `TICKS ~ YEAR + HEIGHT + (1 | BROOD)` with `nAGQ = 1`.

---

## Layout (source files)

| File | Role |
|------|------|
| `figure_specs.py` | DPI, figure sizes, matplotlib `subplots_adjust`, overlay crop constants (raster fallback). |
| `paths.py` | Resolves repo root and `tests/data/`. |
| `plot_demo.py` | Fits models, writes `figures/` and `figures_data/*_py.json`. |
| `plot_r.R` | Fits **lme4**, writes `figures_r/` and `figures_data/*_r.csv`. |
| `numeric_overlay.py` | Builds **`figures_overlay/`** from `figures_data/` (shared-axes overlay). |
| `compare_plots.py` | **`figures_compare/`**; numeric overlay or raster fallback. |
| `find_rscript.py` | Resolves `Rscript` on Windows when not on `PATH`. |
| `run_all.py` | Runs the three stages with project venv + `find_rscript`. |
