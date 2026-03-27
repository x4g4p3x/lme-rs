"""
Regenerate Python figures, R (lme4) figures, and blended overlays.

Requires: lme_python, matplotlib, numpy, pillow; R with lme4 (``Rscript`` on PATH or a normal Windows install).

Uses ``python/.venv`` when present (where ``maturin develop`` installs ``lme_python``), so the launcher ``python`` can differ from the venv that has the extension.

From the repository root:

    python python/examples/plotting_demo/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from find_rscript import find_rscript

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def project_python() -> Path:
    """Prefer ``python/.venv`` (maturin target) over the interpreter running this script."""
    if sys.platform == "win32":
        v = ROOT / "python" / ".venv" / "Scripts" / "python.exe"
    else:
        v = ROOT / "python" / ".venv" / "bin" / "python"
    if v.is_file():
        return v
    return Path(sys.executable)


def main() -> int:
    py = project_python()
    if py != Path(sys.executable):
        print(f"Using project venv Python: {py}", flush=True)

    subprocess.run([str(py), str(HERE / "plot_demo.py")], cwd=str(ROOT), check=True)

    rscript = find_rscript()
    if not rscript:
        print("Rscript not found (not on PATH, registry, or Program Files).", file=sys.stderr)
        print("If R is installed, add ...\\R\\R-x.y.z\\bin to PATH or reinstall R.", file=sys.stderr)
        print(f'Or run manually: "<path-to>\\Rscript.exe" "{HERE / "plot_r.R"}" "{ROOT}"', file=sys.stderr)
        return 2

    print(f"Using Rscript: {rscript}", flush=True)
    subprocess.run([rscript, str(HERE / "plot_r.R"), str(ROOT)], cwd=str(ROOT), check=True)
    subprocess.run([str(py), str(HERE / "compare_plots.py")], cwd=str(ROOT), check=True)
    print("\nDone: figures/, figures_r/, figures_data/, figures_compare/, figures_overlay/ (numeric overlay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
