#!/usr/bin/env python3
"""Run cross-language example benchmarks and emit JSON results."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CASES = {
    "sleepstudy": {
        "rust_example": "sleepstudy",
        "python_script": "examples/sleepstudy.py",
        "r_script": "examples/sleepstudy.R",
        "julia_script": "examples/sleepstudy.jl",
    },
    "pastes": {
        "rust_example": "lmm_pastes",
        "python_script": "examples/lmm_pastes.py",
        "r_script": "examples/lmm_pastes.R",
        "julia_script": "examples/lmm_pastes.jl",
    },
    "cbpp": {
        "rust_example": "glmm_cbpp",
        "python_script": "examples/glmm_cbpp.py",
        "r_script": "examples/glmm_cbpp.R",
        "julia_script": "examples/glmm_cbpp.jl",
    },
    "grouseticks": {
        "rust_example": "glmm_grouseticks",
        "python_script": "examples/glmm_grouseticks.py",
        "r_script": "examples/glmm_grouseticks.R",
        "julia_script": "examples/glmm_grouseticks.jl",
    },
}


@dataclass(frozen=True)
class BenchmarkCommand:
    implementation: str
    case: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-language example benchmarks and save JSON output."
    )
    parser.add_argument(
        "--cases",
        default="sleepstudy,pastes,cbpp,grouseticks",
        help="Comma-separated benchmark cases to run.",
    )
    parser.add_argument(
        "--implementations",
        default="rust,python,r,julia",
        help="Comma-separated implementations to run.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Number of warmup runs per command.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured runs per command.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-run timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        default="benchmark-results/cross-language-benchmarks.json",
        help="Output JSON path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--skip-rust-build",
        action="store_true",
        help="Skip `cargo build --release --examples` before timing Rust binaries.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return stdout or stderr


def maybe_version(command: list[str]) -> str | None:
    try:
        return run_capture(command)
    except Exception:
        return None


def git_sha() -> str | None:
    return maybe_version(["git", "rev-parse", "HEAD"])


def machine_info() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def runtime_versions() -> dict[str, str | None]:
    return {
        "rustc": maybe_version(["rustc", "--version"]),
        "python": maybe_version([sys.executable, "--version"]),
        "Rscript": maybe_version(["Rscript", "--version"]),
        "julia": maybe_version(["julia", "--version"]),
    }


def rust_binary_path(example_name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "target" / "release" / "examples" / f"{example_name}{suffix}"


def available_implementations(requested: Iterable[str]) -> dict[str, bool]:
    requested_set = set(requested)
    return {
        "rust": "rust" in requested_set,
        "python": "python" in requested_set,
        "r": "r" in requested_set and shutil.which("Rscript") is not None,
        "julia": "julia" in requested_set and shutil.which("julia") is not None,
    }


def prepare_rust_examples() -> None:
    subprocess.run(
        ["cargo", "build", "--release", "--examples"],
        cwd=REPO_ROOT,
        check=True,
    )


def build_commands(cases: list[str], implementations: list[str]) -> list[BenchmarkCommand]:
    available = available_implementations(implementations)
    commands: list[BenchmarkCommand] = []
    for case in cases:
        spec = BENCHMARK_CASES[case]
        if available["rust"]:
            rust_binary = rust_binary_path(spec["rust_example"])
            if rust_binary.exists():
                commands.append(
                    BenchmarkCommand("rust", case, [str(rust_binary)])
                )
        if available["python"]:
            commands.append(
                BenchmarkCommand(
                    "python",
                    case,
                    [sys.executable, str(REPO_ROOT / spec["python_script"])],
                )
            )
        if available["r"]:
            commands.append(
                BenchmarkCommand(
                    "r",
                    case,
                    ["Rscript", str(REPO_ROOT / spec["r_script"])],
                )
            )
        if available["julia"]:
            commands.append(
                BenchmarkCommand(
                    "julia",
                    case,
                    ["julia", str(REPO_ROOT / spec["julia_script"])],
                )
            )
    return commands


def summarize(samples: list[float]) -> dict[str, float]:
    result = {
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
    }
    if len(samples) > 1:
        result["stdev_seconds"] = statistics.stdev(samples)
    return result


def timed_run(command: list[str], timeout: int) -> float:
    started = time.perf_counter()
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )
    return time.perf_counter() - started


def capture_failure_output(command: list[str], timeout: int) -> str:
    try:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        details = stderr or stdout or str(exc)
        return details
    except Exception as exc:
        return str(exc)
    return "Command unexpectedly succeeded during failure capture."


def benchmark_command(
    item: BenchmarkCommand, warmups: int, repeats: int, timeout: int
) -> dict[str, object]:
    for _ in range(warmups):
        timed_run(item.command, timeout)

    samples = [timed_run(item.command, timeout) for _ in range(repeats)]
    return {
        "case": item.case,
        "implementation": item.implementation,
        "command": item.command,
        "samples_seconds": samples,
        "summary": summarize(samples),
    }


def main() -> int:
    args = parse_args()
    cases = split_csv(args.cases)
    implementations = split_csv(args.implementations)

    unknown_cases = [case for case in cases if case not in BENCHMARK_CASES]
    if unknown_cases:
        print(f"Unknown benchmark cases: {', '.join(unknown_cases)}", file=sys.stderr)
        return 1

    if "rust" in implementations and not args.skip_rust_build:
        prepare_rust_examples()

    commands = build_commands(cases, implementations)
    if not commands:
        print("No runnable benchmark commands were found.", file=sys.stderr)
        return 1

    results = []
    failures = []
    for item in commands:
        print(f"Benchmarking {item.case} [{item.implementation}] ...")
        try:
            results.append(
                benchmark_command(item, args.warmups, args.repeats, args.timeout)
            )
        except Exception as exc:
            failures.append(
                {
                    "case": item.case,
                    "implementation": item.implementation,
                    "command": item.command,
                    "error": str(exc),
                    "details": capture_failure_output(item.command, args.timeout),
                }
            )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "machine_info": machine_info(),
        "runtime_versions": runtime_versions(),
        "config": {
            "cases": cases,
            "implementations": implementations,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timeout_seconds": args.timeout,
        },
        "results": results,
        "failures": failures,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    if failures:
        print(f"{len(failures)} benchmark runs failed.", file=sys.stderr)
        for failure in failures:
            command = " ".join(failure["command"])
            print(
                f"- {failure['case']} [{failure['implementation']}]: {failure['error']}",
                file=sys.stderr,
            )
            print(f"  command: {command}", file=sys.stderr)
            print(f"  details: {failure['details']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
