#!/usr/bin/env python3
"""Rust (`LME_PERF_DIAG=1`) vs Julia (`optsum.feval`) fit breakdown on fair fixtures."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "benchmark-results" / "fair-rust-julia-data"
PERF_EXAMPLE = "bench_perf_breakdown"
RUST_GEN_EXAMPLE = "bench_fair_rust_julia"
JULIA_SCRIPT = REPO_ROOT / "comparisons" / "bench_fair_julia_perf.jl"


@dataclass(frozen=True)
class FairCase:
    name: str
    generator: str
    formula: str
    reml: bool
    params: dict[str, Any]


FAIR_CASES: dict[str, FairCase] = {
    "random_intercept_10k": FairCase(
        name="random_intercept_10k",
        generator="random_intercept",
        formula="y ~ x + (1 | group)",
        reml=False,
        params={"n_obs": 10_000, "n_groups": 100},
    ),
    "crossed_20k": FairCase(
        name="crossed_20k",
        generator="crossed",
        formula="y ~ x + (1 | plate) + (1 | sample)",
        reml=False,
        params={"n_obs": 20_000, "n_plates": 250, "n_samples": 100},
    ),
    "nested_10k": FairCase(
        name="nested_10k",
        generator="nested",
        formula="y ~ x + (1 | batch/cask)",
        reml=False,
        params={"n_batches": 200, "casks_per_batch": 10, "reps_per_cask": 5},
    ),
}


def rust_binary(name: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return REPO_ROOT / "target" / "release" / "examples" / f"{name}{suffix}"


def resolve_julia(explicit: str | None) -> str | None:
    if explicit:
        path = Path(explicit)
        return str(path) if path.exists() else explicit
    env = os.environ.get("JULIA_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("julia")
    if found:
        return found
    if os.name == "nt":
        programs = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
        if programs.is_dir():
            matches = sorted(programs.glob("Julia-*/bin/julia.exe"))
            if matches:
                return str(matches[-1])
    return None


def build_perf_example() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--example", PERF_EXAMPLE],
        cwd=REPO_ROOT,
        check=True,
    )


def generate_args(case: FairCase, output: Path) -> list[str]:
    cmd = [
        str(rust_binary(RUST_GEN_EXAMPLE)),
        "generate",
        "--kind",
        case.generator,
        "--output",
        str(output),
    ]
    if case.generator == "random_intercept":
        cmd.extend(
            [
                "--n-obs",
                str(case.params["n_obs"]),
                "--n-groups",
                str(case.params["n_groups"]),
            ]
        )
    elif case.generator == "crossed":
        cmd.extend(
            [
                "--n-obs",
                str(case.params["n_obs"]),
                "--n-plates",
                str(case.params["n_plates"]),
                "--n-samples",
                str(case.params["n_samples"]),
            ]
        )
    elif case.generator == "nested":
        cmd.extend(
            [
                "--n-batches",
                str(case.params["n_batches"]),
                "--casks-per-batch",
                str(case.params["casks_per_batch"]),
                "--reps-per-cask",
                str(case.params["reps_per_cask"]),
            ]
        )
    else:
        raise ValueError(f"cannot generate data for {case.name}")
    return cmd


def data_path(case: FairCase, data_dir: Path) -> Path:
    return data_dir / f"{case.name}.csv"


def ensure_data(case: FairCase, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_path(case, data_dir)
    if csv_path.exists():
        return csv_path
    if not rust_binary(RUST_GEN_EXAMPLE).exists():
        subprocess.run(
            ["cargo", "build", "--release", "--locked", "--example", RUST_GEN_EXAMPLE],
            cwd=REPO_ROOT,
            check=True,
        )
    subprocess.run(generate_args(case, csv_path), cwd=REPO_ROOT, check=True)
    return csv_path


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = (completed.stdout or "").strip()
    if not text:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or "empty command output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.splitlines()[-1])


def rust_breakdown(case: FairCase, csv_path: Path, warmups: int) -> dict[str, Any]:
    cmd = [
        str(rust_binary(PERF_EXAMPLE)),
        "--case",
        case.name,
        "--data",
        str(csv_path),
        "--formula",
        case.formula,
        "--reml",
        "true" if case.reml else "false",
        "--warmups",
        str(warmups),
    ]
    return run_json(cmd)


def julia_breakdown(
    julia_bin: str, case: FairCase, csv_path: Path, warmups: int
) -> dict[str, Any]:
    cmd = [
        julia_bin,
        str(JULIA_SCRIPT),
        "--data",
        str(csv_path),
        "--case",
        case.name,
        "--formula",
        case.formula,
        "--reml",
        "true" if case.reml else "false",
        "--warmups",
        str(warmups),
    ]
    return run_json(cmd)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def case_n_obs(case: FairCase) -> int:
    if "n_obs" in case.params:
        return int(case.params["n_obs"])
    if case.generator == "nested":
        return (
            int(case.params["n_batches"])
            * int(case.params["casks_per_batch"])
            * int(case.params["reps_per_cask"])
        )
    raise ValueError(f"cannot infer n_obs for {case.name}")


def compare_rust_julia(
    rust: dict[str, Any] | None, julia: dict[str, Any] | None
) -> dict[str, Any] | None:
    if rust is None or julia is None:
        return None
    rust_evals = rust.get("deviance_eval_count")
    julia_evals = julia.get("optsum", {}).get("optimizer_feval")
    rust_mean = rust.get("mean_deviance_eval_seconds")
    julia_mean = julia.get("mean_feval_seconds")
    rust_fit = rust.get("fit_wall_seconds")
    julia_fit = julia.get("fit_wall_seconds")
    return {
        "rust_deviance_eval_count": rust_evals,
        "julia_optimizer_feval": julia_evals,
        "rust_over_julia_feval": ratio(rust_evals, julia_evals),
        "julia_over_rust_feval": ratio(julia_evals, rust_evals),
        "rust_mean_eval_seconds": rust_mean,
        "julia_mean_feval_seconds": julia_mean,
        "rust_over_julia_mean_eval": ratio(rust_mean, julia_mean),
        "julia_over_rust_mean_eval": ratio(julia_mean, rust_mean),
        "rust_fit_wall_seconds": rust_fit,
        "julia_fit_wall_seconds": julia_fit,
        "rust_over_julia_fit": ratio(rust_fit, julia_fit),
        "julia_over_rust_fit": ratio(julia_fit, rust_fit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="crossed_20k,nested_10k,random_intercept_10k",
        help="Comma-separated fair benchmark cases.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--implementations",
        default="rust,julia",
        help="Comma-separated: rust, julia",
    )
    parser.add_argument(
        "--julia",
        default=None,
        help="Path to julia executable (else JULIA_BIN, PATH, Windows default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    implementations = {part.strip() for part in args.implementations.split(",") if part.strip()}

    julia_bin = resolve_julia(args.julia)
    if "julia" in implementations and not julia_bin:
        print(
            "Julia not found; skipping Julia breakdown (set PATH, JULIA_BIN, or --julia).",
            file=sys.stderr,
        )
        implementations.discard("julia")

    if not implementations:
        print("No implementations to run.", file=sys.stderr)
        return 1

    if "rust" in implementations and not args.skip_build:
        build_perf_example()

    reports: list[dict[str, Any]] = []
    for name in [c.strip() for c in args.cases.split(",") if c.strip()]:
        if name not in FAIR_CASES:
            print(f"unknown case: {name}", file=sys.stderr)
            return 1
        case = FAIR_CASES[name]
        csv_path = ensure_data(case, data_dir)

        rust_report = None
        julia_report = None
        failures: list[dict[str, str]] = []

        if "rust" in implementations:
            try:
                rust_report = rust_breakdown(case, csv_path, args.warmups)
            except Exception as exc:
                failures.append({"implementation": "rust", "error": str(exc)})

        if "julia" in implementations and julia_bin:
            try:
                julia_report = julia_breakdown(julia_bin, case, csv_path, args.warmups)
            except Exception as exc:
                failures.append({"implementation": "julia", "error": str(exc)})

        entry = {
            "case": case.name,
            "formula": case.formula,
            "reml": case.reml,
            "n_obs": case_n_obs(case),
            "rust": rust_report,
            "julia": julia_report,
            "comparison": compare_rust_julia(rust_report, julia_report),
            "failures": failures,
        }
        reports.append(entry)
        print(json.dumps(entry, indent=2))

    out_path = REPO_ROOT / "benchmark-results" / "perf-breakdown.json"
    out_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
