#!/usr/bin/env python3
"""Fair Rust vs Julia benchmark (fit-only timing, shared CSV fixtures).

Unlike scripts/run_cross_language_benchmarks.py, this harness:
- generates identical synthetic CSVs from the same RNG recipes as benches/bench_math.rs
- loads data once per language before timing
- records only the model fit call (no process startup, no CSV I/O in timed section)
- warms up Julia JIT before measured repeats
- optional Rust ``prepare_lmer`` / ``fit_prepared`` phases (LMM cases)

Requires: built bench_fair_rust_julia example; Julia with CSV, DataFrames, JSON,
MixedModels, and GLM (for GLMM cases). See BENCHMARK_COVERAGE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_EXAMPLE = "bench_fair_rust_julia"
JULIA_SCRIPT = REPO_ROOT / "comparisons" / "bench_fair_julia_timing.jl"
DEFAULT_DATA_DIR = REPO_ROOT / "benchmark-results" / "fair-rust-julia-data"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark-results" / "fair-rust-julia-benchmarks.json"


@dataclass(frozen=True)
class FairCase:
    name: str
    generator: str
    formula: str
    reml: bool
    params: dict[str, int]
    data_source: str | None = None
    model: str = "lmm"
    tier: str = "A"
    reference: str = "MixedModels.jl fit"


def lmm_case(**kwargs: Any) -> FairCase:
    return FairCase(model="lmm", **kwargs)


FAIR_CASES: dict[str, FairCase] = {
    "sleepstudy_reml": lmm_case(
        name="sleepstudy_reml",
        generator="fixture",
        formula="Reaction ~ Days + (Days | Subject)",
        reml=True,
        params={},
        data_source="tests/data/sleepstudy.csv",
        reference="MixedModels.jl; random slopes (real fixture)",
    ),
    "sleepstudy_weighted_reml": FairCase(
        name="sleepstudy_weighted_reml",
        generator="fixture",
        formula="Reaction ~ Days + (Days | Subject)",
        reml=True,
        params={},
        data_source="tests/data/sleepstudy.csv",
        model="lmm_weighted",
        reference="MixedModels.jl wts=…",
    ),
    "penicillin_crossed_reml": lmm_case(
        name="penicillin_crossed_reml",
        generator="fixture",
        formula="diameter ~ 1 + (1 | plate) + (1 | sample)",
        reml=True,
        params={},
        data_source="tests/data/penicillin.csv",
        reference="MixedModels.jl; real crossed intercept",
    ),
    "pastes_nested_reml": lmm_case(
        name="pastes_nested_reml",
        generator="fixture",
        formula="strength ~ 1 + (1 | batch/cask)",
        reml=True,
        params={},
        data_source="tests/data/pastes.csv",
        reference="MixedModels.jl; real nested intercept",
    ),
    "random_intercept_10k": lmm_case(
        name="random_intercept_10k",
        generator="random_intercept",
        formula="y ~ x + (1 | group)",
        reml=False,
        params={"n_obs": 10_000, "n_groups": 100},
    ),
    "random_intercept_50k": lmm_case(
        name="random_intercept_50k",
        generator="random_intercept",
        formula="y ~ x + (1 | group)",
        reml=False,
        params={"n_obs": 50_000, "n_groups": 500},
    ),
    "random_intercept_100k": lmm_case(
        name="random_intercept_100k",
        generator="random_intercept",
        formula="y ~ x + (1 | group)",
        reml=False,
        params={"n_obs": 100_000, "n_groups": 1_000},
    ),
    "crossed_20k": lmm_case(
        name="crossed_20k",
        generator="crossed",
        formula="y ~ x + (1 | plate) + (1 | sample)",
        reml=False,
        params={"n_obs": 20_000, "n_plates": 250, "n_samples": 100},
    ),
    "nested_10k": lmm_case(
        name="nested_10k",
        generator="nested",
        formula="y ~ x + (1 | batch/cask)",
        reml=False,
        params={"n_batches": 200, "casks_per_batch": 10, "reps_per_cask": 5},
    ),
    "cbpp_binomial_ml": FairCase(
        name="cbpp_binomial_ml",
        generator="fixture",
        formula="y ~ period2 + period3 + period4 + (1 | herd)",
        reml=False,
        params={},
        data_source="tests/data/cbpp_binary.csv",
        model="glmm_binomial",
        reference="MixedModels.jl GLMM (Laplace); not lme4 AGQ-in-θ",
    ),
    "grouseticks_poisson_ml": FairCase(
        name="grouseticks_poisson_ml",
        generator="fixture",
        formula="TICKS ~ YEAR + HEIGHT + (1 | BROOD)",
        reml=False,
        params={},
        data_source="tests/data/grouseticks.csv",
        model="glmm_poisson",
        reference="MixedModels.jl GLMM (Laplace)",
    ),
}

LMM_CORE_CASES = (
    "sleepstudy_reml",
    "random_intercept_10k",
    "random_intercept_50k",
    "random_intercept_100k",
    "crossed_20k",
    "nested_10k",
)
DEFAULT_CASES = ",".join(FAIR_CASES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help="Comma-separated fair benchmark cases.",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--julia",
        default=None,
        help="Path to julia executable (else JULIA_BIN, else PATH).",
    )
    parser.add_argument(
        "--skip-rust-build",
        action="store_true",
        help="Skip `cargo build --release --example bench_fair_rust_julia`.",
    )
    parser.add_argument(
        "--implementations",
        default="rust,julia",
        help="Comma-separated implementations to run.",
    )
    parser.add_argument(
        "--with-phases",
        action="store_true",
        help="Record Rust prepare_lmer and fit_prepared (LMM cases only).",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=1.5,
        help="Target rust/julia median ratio for cold_fit (axis 3 threshold; default 1.5).",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def rust_binary() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "target" / "release" / "examples" / f"{RUST_EXAMPLE}{suffix}"


def run_capture(command: list[str], timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (completed.stdout or completed.stderr).strip()


def maybe_version(command: list[str]) -> str | None:
    try:
        return run_capture(command, timeout=60)
    except Exception:
        return None


def git_sha() -> str | None:
    return maybe_version(["git", "rev-parse", "HEAD"])


def machine_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def build_rust_example() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--example", RUST_EXAMPLE],
        cwd=REPO_ROOT,
        check=True,
    )


def generate_args(case: FairCase, output: Path) -> list[str]:
    cmd = [
        str(rust_binary()),
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
    if case.data_source:
        return REPO_ROOT / case.data_source
    return data_dir / f"{case.name}.csv"


def ensure_data(case: FairCase, data_dir: Path, timeout: int) -> Path:
    path = data_path(case, data_dir)
    if case.generator == "fixture":
        if not path.exists():
            raise FileNotFoundError(f"missing fixture CSV for {case.name}: {path}")
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(generate_args(case, path), cwd=REPO_ROOT, check=True, timeout=timeout)
    return path


def supports_phases(case: FairCase) -> bool:
    return case.model in ("lmm", "lmm_weighted")


def metric_median(result: dict[str, Any], metric: str) -> float:
    block = result.get(metric)
    if isinstance(block, dict) and "summary" in block:
        return float(block["summary"]["median_seconds"])
    if metric == "cold_fit" and "summary" in result:
        return float(result["summary"]["median_seconds"])
    raise KeyError(f"metric {metric} missing from result")


def rust_time_command(
    case: FairCase,
    csv_path: Path,
    warmups: int,
    repeats: int,
    with_phases: bool,
) -> list[str]:
    cmd = [
        str(rust_binary()),
        "time",
        "--case",
        case.name,
        "--data",
        str(csv_path),
        "--formula",
        case.formula,
        "--model",
        case.model,
        "--reml",
        "true" if case.reml else "false",
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
    ]
    if with_phases and supports_phases(case):
        cmd.append("--with-phases")
    return cmd


def julia_time_command(
    julia_bin: str,
    case: FairCase,
    csv_path: Path,
    warmups: int,
    repeats: int,
) -> list[str]:
    return [
        julia_bin,
        str(JULIA_SCRIPT),
        "--data",
        str(csv_path),
        "--case",
        case.name,
        "--formula",
        case.formula,
        "--model",
        case.model,
        "--reml",
        "true" if case.reml else "false",
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
    ]


def run_timing(command: list[str], timeout: int) -> dict[str, Any]:
    stdout = run_capture(command, timeout=timeout)
    line = stdout.splitlines()[-1]
    return json.loads(line)


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def compare_metric(
    rust: dict[str, Any] | None,
    julia: dict[str, Any] | None,
    metric: str,
    target_ratio: float,
) -> dict[str, Any] | None:
    if rust is None or julia is None:
        return None
    rust_med = metric_median(rust, metric)
    julia_med = metric_median(julia, metric)
    rust_over = ratio(rust_med, julia_med)
    return {
        "metric": metric,
        "rust_median_seconds": rust_med,
        "julia_median_seconds": julia_med,
        "rust_over_julia_median": rust_over,
        "julia_over_rust_median": ratio(julia_med, rust_med),
        "faster_implementation": (
            "rust"
            if rust_med < julia_med
            else "julia"
            if julia_med < rust_med
            else "tie"
        ),
        "meets_target": rust_over is not None and rust_over <= target_ratio,
    }


def compare_case(
    case: FairCase,
    rust: dict[str, Any] | None,
    julia: dict[str, Any] | None,
    target_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cold = compare_metric(rust, julia, "cold_fit", target_ratio)
    if cold is not None:
        rows.append(cold)
    if rust is not None and rust.get("fit_prepared") is not None and julia is not None:
        rust_med = metric_median(rust, "fit_prepared")
        julia_med = metric_median(julia, "cold_fit")
        rust_over = ratio(rust_med, julia_med)
        rows.append(
            {
                "metric": "fit_prepared_vs_julia_fit",
                "rust_median_seconds": rust_med,
                "julia_median_seconds": julia_med,
                "rust_over_julia_median": rust_over,
                "julia_over_rust_median": ratio(julia_med, rust_med),
                "faster_implementation": (
                    "rust"
                    if rust_med < julia_med
                    else "julia"
                    if julia_med < rust_med
                    else "tie"
                ),
                "meets_target": rust_over is not None and rust_over <= target_ratio,
            }
        )
    if rust is not None and rust.get("prepare_lmer") is not None:
        prep = rust["prepare_lmer"]["summary"]["median_seconds"]
        rows.append(
            {
                "metric": "prepare_lmer_rust_only",
                "rust_median_seconds": prep,
                "julia_median_seconds": None,
                "rust_over_julia_median": None,
                "julia_over_rust_median": None,
                "faster_implementation": "n/a",
                "meets_target": None,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    cases = split_csv(args.cases)
    implementations = set(split_csv(args.implementations))
    unknown = [case for case in cases if case not in FAIR_CASES]
    if unknown:
        print(f"Unknown cases: {', '.join(unknown)}", file=sys.stderr)
        return 1

    julia_bin = resolve_julia(args.julia)
    if "julia" in implementations and not julia_bin:
        print(
            "Julia not found. Set PATH, JULIA_BIN, or pass --julia.",
            file=sys.stderr,
        )
        return 1

    glmm_cases = [name for name in cases if FAIR_CASES[name].model.startswith("glmm")]
    if "julia" in implementations and julia_bin and glmm_cases:
        try:
            run_capture([julia_bin, "-e", "using GLM"], timeout=120)
        except Exception as exc:
            print(
                "GLM.jl is required for GLMM fair cases "
                f"({', '.join(glmm_cases)}). Install with: "
                'julia -e \'using Pkg; Pkg.add("GLM")\'',
                file=sys.stderr,
            )
            print(f"Probe failed: {exc}", file=sys.stderr)
            return 1

    if "rust" in implementations:
        if not args.skip_rust_build:
            build_rust_example()
        if not rust_binary().exists():
            print(f"Missing Rust example binary: {rust_binary()}", file=sys.stderr)
            return 1

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for case_name in cases:
        case = FAIR_CASES[case_name]
        print(f"Fair benchmark: {case_name} ...")
        try:
            csv_path = ensure_data(case, data_dir, args.timeout)
        except Exception as exc:
            failures.append({"case": case_name, "stage": "data", "error": str(exc)})
            continue

        rust_result = None
        julia_result = None

        if "rust" in implementations:
            try:
                rust_result = run_timing(
                    rust_time_command(
                        case, csv_path, args.warmups, args.repeats, args.with_phases
                    ),
                    args.timeout,
                )
                results.append(rust_result)
            except Exception as exc:
                failures.append(
                    {
                        "case": case_name,
                        "implementation": "rust",
                        "error": str(exc),
                    }
                )

        if "julia" in implementations and julia_bin:
            try:
                julia_result = run_timing(
                    julia_time_command(
                        julia_bin, case, csv_path, args.warmups, args.repeats
                    ),
                    args.timeout,
                )
                results.append(julia_result)
            except Exception as exc:
                failures.append(
                    {
                        "case": case_name,
                        "implementation": "julia",
                        "error": str(exc),
                    }
                )

        case_rows = compare_case(case, rust_result, julia_result, args.target_ratio)
        if case_rows:
            comparisons.append(
                {
                    "case": case_name,
                    "model": case.model,
                    "reference": case.reference,
                    "metrics": case_rows,
                }
            )
            for row in case_rows:
                metric = row["metric"]
                rust_med = row["rust_median_seconds"]
                julia_med = row["julia_median_seconds"]
                if julia_med is not None:
                    ratio_s = row["rust_over_julia_median"]
                    print(
                        f"  {metric}: rust={rust_med:.4f}s julia={julia_med:.4f}s "
                        f"ratio={ratio_s:.2f} target<={args.target_ratio}"
                    )
                else:
                    print(f"  {metric}: rust={rust_med:.4f}s")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "methodology": {
            "description": "Shared CSV fixtures; data loaded once; model fit timed per metric; Julia JIT warmed up.",
            "coverage_doc": "BENCHMARK_COVERAGE.md",
            "data_generation": "comparisons/bench_fair_rust_julia.rs generate (matches benches/bench_math.rs RNG recipes)",
            "rust_engine": "lme-rs lmer/glmer/lmer_weighted; optional prepare_lmer + fit_prepared",
            "julia_engine": "MixedModels.jl fit / GeneralizedLinearMixedModel (GLM.jl)",
            "target_ratio_cold_fit": args.target_ratio,
            "note": "Different optimizers and likelihood paths; compare throughput, not coefficient identity.",
        },
        "machine_info": machine_info(),
        "runtime_versions": {
            "rustc": maybe_version(["rustc", "--version"]),
            "julia": maybe_version([julia_bin, "--version"]) if julia_bin else None,
        },
        "config": {
            "cases": cases,
            "implementations": sorted(implementations),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timeout_seconds": args.timeout,
            "with_phases": args.with_phases,
            "target_ratio": args.target_ratio,
            "data_dir": str(data_dir),
        },
        "results": results,
        "comparisons": comparisons,
        "failures": failures,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")

    if failures:
        print(f"{len(failures)} fair benchmark stages failed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
