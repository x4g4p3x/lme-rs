# Model diagnostics and comparison plots

[Example catalog](../../../docs/EXAMPLES.md) · [Verification project](../verification_project/README.md)

This optional demo fits models with `lme_python`, draws Matplotlib diagnostics,
and can compare them with R `lme4` output. Plots help inspect a model;
[the verification project](../verification_project/README.md) provides numeric
assertions.

## Python setup

Start in the repository root and build the extension in the project environment:

```bash
mise install
cd python
uv sync --extra dev --no-install-project
uv run --no-sync maturin develop --release
uv pip install matplotlib numpy pillow
```

The plotting packages are optional demo dependencies. Use `--no-sync` for the
commands below so the editable extension and these packages remain installed.

## Python figures only

From **`python/`**:

```bash
uv run --no-sync python examples/plotting_demo/plot_demo.py
```

Outputs go under this demo directory. Fixture paths are resolved by
[paths.py](paths.py).

## Optional R comparison

Install R and the `lme4` package. In an R session:

```r
install.packages("lme4")
```

Then, from **`python/`**, run the complete pipeline:

```bash
uv run --no-sync python examples/plotting_demo/run_all.py
```

The runner executes Python figures, R figures, then comparisons. It prefers
the repository's Python environment. [find_rscript.py](find_rscript.py)
locates Rscript using PATH and supported Windows installation locations.

To run just the R stage from the **repository root**:

```bash
Rscript python/examples/plotting_demo/plot_r.R
```

To rebuild comparisons from existing outputs, run from **`python/`**:

```bash
uv run --no-sync python examples/plotting_demo/compare_plots.py
```

## Read the outputs

| Directory | Contents |
|:----------|:---------|
| `figures/` | Python model diagnostics |
| `figures_r/` | R diagnostic figures |
| `figures_data/` | Python JSON and R CSV used for numeric overlays |
| `figures_compare/` | Side-by-side image panels |
| `figures_overlay/` | Shared-axis numeric overlays: Python blue, R orange |
| `figures_overlay_raster/` | Best-effort raster fallback when numeric data is incomplete |

These generated directories are ignored by Git. A raster overlay is an image
comparison, not evidence of numerical agreement. Optional `--overlay-raw`
produces a full-frame blend that may be misaligned.

## Models and prediction semantics

| Figure | Model/API |
|:-------|:----------|
| Sleepstudy residuals versus fitted | Random-slope REML LMM; `fitted` and `residuals` |
| Sleepstudy curves | `predict` for population effects; `predict_conditional` for known subjects |
| Grouseticks observations versus fitted means | Poisson GLMM; `predict_response` |

Formulas are `Reaction ~ Days + (Days | Subject)` and
`TICKS ~ YEAR + HEIGHT + (1 | BROOD)`. The GLMM uses Laplace (`n_agq=1`).
Check [the Python guide](../../PYTHON_GUIDE.md#predictions) before comparing
link-scale and response-scale quantities.

## Source map

- [plot_demo.py](plot_demo.py): Python fits and exported values.
- [plot_r.R](plot_r.R): R fits and exported values.
- [numeric_overlay.py](numeric_overlay.py): shared-axis overlays.
- [compare_plots.py](compare_plots.py): side-by-side and fallback comparisons.
- [figure_specs.py](figure_specs.py): shared sizing and layout settings.
- [run_all.py](run_all.py): pipeline orchestration.
