"""
Compare Python (matplotlib) and R (base graphics) PNGs.

**Side-by-side** (``figures_compare/``): unchanged originals, resized to match — for a clean A|B view.

**Overlay** (``figures_overlay/``): when ``figures_data/*`` exists (JSON from ``plot_demo.py`` + CSV from
``plot_r.R``), builds a **numeric** matplotlib figure with **shared axes** — Python in blue, R in
orange (see ``numeric_overlay.py``). That is the reliable “real” overlay.

If those exports are missing, falls back to **raster** alignment (crop + shift + tint) into
``figures_overlay_raster/``.

Optional **--overlay-raw**: legacy full-frame 50/50 alpha blend (debug).

Requires pillow, numpy; numeric overlay also needs matplotlib and polars (same as ``plot_demo``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Union

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    print(f"Import error: {e}")
    print("Install: pip install pillow numpy")
    sys.exit(1)

from figure_specs import (
    OUTPUT_FILES,
    OVERLAY_CROP_FRAC,
    OVERLAY_MAX_SHIFT_PX,
    OVERLAY_PY_RGB,
    OVERLAY_R_RGB,
)

try:
    from numeric_overlay import write_all_numeric_overlays
except ImportError:
    write_all_numeric_overlays = None  # type: ignore[misc, assignment]


def _have_numeric_overlay_data(here: Path) -> bool:
    fd = here / "figures_data"
    pairs = [
        ("sleepstudy_residuals_py.json", "sleepstudy_residuals_r.csv"),
        ("sleepstudy_curves_py.json", "sleepstudy_curves_r.csv"),
        ("grouseticks_py.json", "grouseticks_r.csv"),
    ]
    return all((fd / a).is_file() and (fd / b).is_file() for a, b in pairs)


def _title_font(size: int = 17) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    candidates: list[str] = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates.extend(
        [
            os.path.join(windir, "Fonts", "segoeui.ttf"),
            os.path.join(windir, "Fonts", "arial.ttf"),
        ]
    )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _offset_rgb(
    img: Image.Image,
    dx: int,
    dy: int,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Translate RGB image by (dx, dy), padding with ``fill``."""
    img = img.convert("RGB")
    w, h = img.size
    out = Image.new("RGB", (w, h), fill)
    out.paste(img, (dx, dy))
    return out


def _best_shift_py_r(py_c: Image.Image, r_c: Image.Image, max_shift: int) -> tuple[int, int]:
    """Integer shift (dx, dy) applied to R to minimize mean |luma_py - luma_r| (white fill)."""
    py = np.asarray(py_c.convert("L"), dtype=np.float32)
    r0 = np.asarray(r_c.convert("L"), dtype=np.float32)
    if py.shape != r0.shape:
        r0 = np.asarray(r_c.resize(py_c.size, Image.Resampling.LANCZOS).convert("L"), dtype=np.float32)
    h, w = py.shape
    iy, ix = np.meshgrid(np.arange(h, dtype=np.int32), np.arange(w, dtype=np.int32), indexing="ij")
    best_dx, best_dy = 0, 0
    best_err = float("inf")
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            jx = ix - dx
            jy = iy - dy
            valid = (jx >= 0) & (jx < w) & (jy >= 0) & (jy < h)
            rs = np.full((h, w), 255.0, dtype=np.float32)
            rs[valid] = r0[jy[valid], jx[valid]]
            err = float(np.mean(np.abs(py - rs)))
            if err < best_err:
                best_err = err
                best_dx, best_dy = dx, dy
    return best_dx, best_dy


def _crop_frac(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    """Remove fractional margins from left, top, right, bottom (each in [0,1))."""
    w, h = img.size
    x0 = int(round(w * left))
    y0 = int(round(h * top))
    x1 = int(round(w * (1.0 - right)))
    y1 = int(round(h * (1.0 - bottom)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop {left, top, right, bottom} for size {w}x{h}")
    return img.crop((x0, y0, x1, y1))


def _diff_stats_full(py_path: Path, r_path: Path) -> tuple[float, float]:
    """Mean and max |RGB| after resizing R to Python dimensions (full frame)."""
    a = Image.open(py_path).convert("RGBA")
    b = Image.open(r_path).convert("RGBA")
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    diff = np.abs(aa[..., :3] - bb[..., :3])
    return float(np.mean(diff)), float(np.max(diff))


def _diff_stats_cropped(
    py_path: Path,
    r_path: Path,
    crop: Tuple[float, float, float, float],
) -> tuple[float, float]:
    """Mean and max |RGB| on cropped panels after the same shift alignment as the overlay."""
    a = Image.open(py_path).convert("RGBA")
    b = Image.open(r_path).convert("RGBA")
    if b.size != a.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    l, t, r, b_ = crop
    ca = _crop_frac(a, l, t, r, b_).convert("RGB")
    cb = _crop_frac(b, l, t, r, b_).convert("RGB")
    if cb.size != ca.size:
        cb = cb.resize(ca.size, Image.Resampling.LANCZOS)
    dx, dy = _best_shift_py_r(ca, cb, OVERLAY_MAX_SHIFT_PX)
    cb = _offset_rgb(cb, dx, dy)
    aa = np.asarray(ca, dtype=np.float32)
    bb = np.asarray(cb, dtype=np.float32)
    diff = np.abs(aa[..., :3] - bb[..., :3])
    return float(np.mean(diff)), float(np.max(diff))


def _side_by_side(py_path: Path, r_path: Path, out_path: Path, gap: int = 24, header: int = 40) -> None:
    py = Image.open(py_path).convert("RGBA")
    r = Image.open(r_path).convert("RGBA")
    if r.size != py.size:
        r = r.resize(py.size, Image.Resampling.LANCZOS)
    w, h = py.size
    total_w = w * 2 + gap
    total_h = h + header
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    canvas.paste(py.convert("RGB"), (0, header))
    canvas.paste(r.convert("RGB"), (w + gap, header))
    draw = ImageDraw.Draw(canvas)
    font = _title_font(17)
    draw.text((12, 10), "lme_python", fill=(33, 33, 33), font=font)
    draw.text((w + gap + 12, 10), "lme4 (R)", fill=(33, 33, 33), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _aligned_color_overlay(
    py_path: Path,
    r_path: Path,
    out_path: Path,
    crop: Tuple[float, float, float, float],
) -> None:
    """Crop panels, align R to Python by small shift search, tint, blend 50/50."""
    py = Image.open(py_path).convert("RGBA")
    r = Image.open(r_path).convert("RGBA")
    if r.size != py.size:
        r = r.resize(py.size, Image.Resampling.LANCZOS)

    l, t, rgt, b = crop
    py_c = _crop_frac(py, l, t, rgt, b).convert("RGB")
    r_c = _crop_frac(r, l, t, rgt, b).convert("RGB")
    if r_c.size != py_c.size:
        r_c = r_c.resize(py_c.size, Image.Resampling.LANCZOS)

    dx, dy = _best_shift_py_r(py_c, r_c, OVERLAY_MAX_SHIFT_PX)
    r_c = _offset_rgb(r_c, dx, dy)

    pa = np.asarray(py_c, dtype=np.float32) / 255.0
    ra = np.asarray(r_c, dtype=np.float32) / 255.0

    py_t = pa * np.array(OVERLAY_PY_RGB, dtype=np.float32)
    r_t = ra * np.array(OVERLAY_R_RGB, dtype=np.float32)
    comb = np.clip(0.5 * py_t + 0.5 * r_t, 0.0, 1.0)
    out = (comb * 255.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(out_path)


def _blend_raw_fullframe(py_path: Path, r_path: Path, out_path: Path) -> None:
    a = Image.open(py_path).convert("RGBA")
    b = Image.open(r_path).convert("RGBA")
    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    blended = Image.blend(a, b, 0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python vs R figure PNGs.")
    parser.add_argument(
        "--overlay-raw",
        action="store_true",
        help="Also write an un-cropped 50%% alpha blend (usually misaligned).",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    py_dir = here / "figures"
    r_dir = here / "figures_r"
    compare_dir = here / "figures_compare"
    overlay_dir = here / "figures_overlay"
    raster_dir = here / "figures_overlay_raster"

    if not py_dir.is_dir():
        print(f"Missing {py_dir} — run plot_demo.py first.", file=sys.stderr)
        sys.exit(1)
    if not r_dir.is_dir():
        print(f"Missing {r_dir} — generate R figures first (install R + lme4, then plot_r.R).", file=sys.stderr)
        print("  Windows: winget install RProject.R  then: Rscript python/examples/plotting_demo/plot_r.R", file=sys.stderr)
        sys.exit(1)

    default_crop = (0.12, 0.10, 0.04, 0.12)

    print("Side-by-side (identical styling to source PNGs):")
    for name in OUTPUT_FILES:
        py_p = py_dir / name
        r_p = r_dir / name
        if not py_p.is_file() or not r_p.is_file():
            print(f"  skip (missing): {name}", file=sys.stderr)
            continue
        out_p = compare_dir / name
        _side_by_side(py_p, r_p, out_p)
        mae, mmax = _diff_stats_full(py_p, r_p)
        print(f"  {name}")
        print(f"    -> {out_p}")
        print(f"    full-frame RGB |d| mean / max: {mae:.2f} / {mmax:.2f}")

    use_numeric = (
        write_all_numeric_overlays is not None
        and _have_numeric_overlay_data(here)
    )
    if use_numeric:
        print("\nNumeric overlay (shared axes; Python=blue, R=orange) -> figures_overlay/:")
        ok = write_all_numeric_overlays(here)
        for name, o in zip(OUTPUT_FILES, ok):
            print(f"  {name}  ok={o}")
    else:
        print(
            "\nNo complete figures_data/ (re-run plot_demo.py and plot_r.R). "
            "Raster fallback -> figures_overlay_raster/:",
            file=sys.stderr,
        )
        for name in OUTPUT_FILES:
            py_p = py_dir / name
            r_p = r_dir / name
            if not py_p.is_file() or not r_p.is_file():
                continue
            crop = OVERLAY_CROP_FRAC.get(name, default_crop)
            out_p = raster_dir / name
            _aligned_color_overlay(py_p, r_p, out_p, crop)
            mae_c, mmax_c = _diff_stats_cropped(py_p, r_p, crop)
            print(f"  {name}")
            print(f"    -> {out_p}")
            print(f"    cropped-panel RGB |d| mean / max: {mae_c:.2f} / {mmax_c:.2f}")

    if args.overlay_raw:
        print("\nOptional raw full-frame 50/50 blend (debug) -> figures_overlay/:")
        for name in OUTPUT_FILES:
            py_p = py_dir / name
            r_p = r_dir / name
            if not py_p.is_file() or not r_p.is_file():
                continue
            out_p = overlay_dir / (name.replace(".png", "_raw_blend.png"))
            _blend_raw_fullframe(py_p, r_p, out_p)
            print(f"  -> {out_p}")


if __name__ == "__main__":
    main()
