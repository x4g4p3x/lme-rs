#!/usr/bin/env python3
"""Fair-ish external timings: nlmer, post-fit inference, and Python FFI.

Times only the model-fit / inference call after data is loaded. Julia is not
required. R (`lme4`, optional `lmerTest`) and `lme_python` are used when present.

    python3 scripts/run_external_timings.py --repeats 5 --output \\
        benchmarks/external-nlmm-inference-python-timings-2026-08-14.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUST_EXAMPLE = "bench_external_timings"
R_SCRIPT = ROOT / "comparisons" / "bench_external_timings.R"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "external-nlmm-inference-python-timings.json"
CASES = [
    "sleepstudy_lmer",
    "sleepstudy_satterthwaite",
    "sleepstudy_kenward_roger",
    "orange_nlmer",
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"==> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, check=True, capture_output=True)


def rust_report(case: str, warmups: int, repeats: int) -> dict[str, Any]:
    completed = _run(
        [
            "cargo",
            "run",
            "--release",
            "--locked",
            "--example",
            RUST_EXAMPLE,
            "--",
            "--case",
            case,
            "--warmups",
            str(warmups),
            "--repeats",
            str(repeats),
        ]
    )
    text = completed.stdout
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"rust example produced no JSON for {case}: {text[-500:]}")
    return json.loads(text[start:])


def rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def r_has_package(pkg: str) -> bool:
    probe = subprocess.run(
        ["Rscript", "-e", f"suppressPackageStartupMessages(library({pkg}))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def r_report(case: str, warmups: int, repeats: int) -> dict[str, Any] | None:
    if not rscript_available() or not r_has_package("lme4"):
        return None
    completed = _run(
        [
            "Rscript",
            str(R_SCRIPT),
            "--case",
            case,
            "--warmups",
            str(warmups),
            "--repeats",
            str(repeats),
        ]
    )
    text = completed.stdout
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"R script produced no JSON for {case}: {text[-500:]}")
    return json.loads(text[start:])


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "mean_seconds": statistics.fmean(ordered),
        "median_seconds": median,
    }


def python_lmer_report(warmups: int, repeats: int) -> dict[str, Any] | None:
    venv_site = ROOT / "python" / ".venv" / "lib"
    if venv_site.is_dir():
        for path in venv_site.glob("python*/site-packages"):
            site = str(path)
            if site not in sys.path:
                sys.path.insert(0, site)
    try:
        import lme_python
        import polars as pl
    except ImportError as exc:
        print(f"skip python FFI: {exc}", flush=True)
        return None

    path = ROOT / "tests" / "data" / "sleepstudy.csv"
    df = pl.read_csv(path)
    formula = "Reaction ~ Days + (Days | Subject)"

    def once() -> None:
        lme_python.lmer(formula, df, True)

    for _ in range(warmups):
        once()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        once()
        samples.append(time.perf_counter() - started)
    return {
        "implementation": "python_lme_python",
        "case": "sleepstudy_lmer",
        "family": "python_ffi",
        "formula": formula,
        "n_obs": df.height,
        "warmups": warmups,
        "repeats": repeats,
        "samples_seconds": samples,
        "summary": summarize(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Do not time lme_python (useful before maturin develop)",
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Do not time R even when Rscript/lme4 are installed",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Smoke: Rust timings only",
    )
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help="Comma-separated case list",
    )
    args = parser.parse_args()
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = [item for item in cases if item not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}")

    reports: list[dict[str, Any]] = []
    skipped: list[str] = []

    for case in cases:
        reports.append(rust_report(case, args.warmups, args.repeats))
        if args.rust_only or args.skip_r:
            continue
        r_payload = r_report(case, args.warmups, args.repeats)
        if r_payload is None:
            skipped.append(f"R:{case}")
        elif r_payload.get("skipped"):
            skipped.append(str(r_payload["skipped"]))
        else:
            reports.append(r_payload)

    if not args.rust_only and not args.skip_python and "sleepstudy_lmer" in cases:
        py_payload = python_lmer_report(args.warmups, args.repeats)
        if py_payload is None:
            skipped.append("python:sleepstudy_lmer")
        else:
            reports.append(py_payload)

    families = sorted({report["family"] for report in reports})
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "warmups": args.warmups,
        "repeats": args.repeats,
        "families_present": families,
        "skipped": skipped,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(reports)} reports, skipped={skipped})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or "")
        raise SystemExit(exc.returncode) from exc
