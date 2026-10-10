#!/usr/bin/env python3
"""Bounded toolchain probes; build artifacts before measuring, and run sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {
    key: "1"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def abi_worker() -> dict:
    import importlib.metadata
    import math

    import lme_python as lme
    import polars as pl
    from lme_python import lme_python as native

    x = [float(i) for i in range(100)]
    y = [1 + 0.4 * value + (((i * 7) % 11) - 5) * 0.1 for i, value in enumerate(x)]
    data = pl.DataFrame({"x": x, "y": y})
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    slope = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    slope /= math.fsum((a - mean_x) ** 2 for a in x)
    expected = [mean_y - slope * mean_x, slope]
    fit = lme.lm("y ~ x", data)
    assert all(abs(a - b) < 1e-10 for a, b in zip(fit.coefficients, expected, strict=True))
    results = {}
    start = time.perf_counter()
    for _ in range(200):
        fitted = lme.lm("y ~ x", data)
    results["lm_200"] = time.perf_counter() - start
    assert fitted.coefficients == fit.coefficients
    start = time.perf_counter()
    for _ in range(50000):
        coefficients = fit.coefficients
    results["coefficients_50000"] = time.perf_counter() - start
    assert coefficients == fit.coefficients
    start = time.perf_counter()
    for _ in range(1000):
        prediction = fit.predict(data)
    results["predict_1000"] = time.perf_counter() - start
    assert max(abs(a - (expected[0] + expected[1] * b)) for a, b in zip(prediction, x)) < 1e-9
    return {
        "seconds": results,
        "coefficients": coefficients,
        "python": sys.version,
        "module": str(native.__file__),
        "native_sha256": digest(Path(native.__file__)),
        "dependencies": {name: importlib.metadata.version(name) for name in ("polars", "numpy")},
    }


def paired_measurements(commands: dict[str, list[str]], directory: Path, *, worker=False) -> dict:
    env = {**os.environ, **THREAD_ENV}
    records = []
    labels = list(commands)
    for pair in range(-2, 10):
        order = labels if pair % 2 == 0 else list(reversed(labels))
        for label in order:
            start = time.perf_counter()
            result = subprocess.run(
                commands[label], cwd=ROOT, env=env, capture_output=True, text=True
            )
            elapsed = time.perf_counter() - start
            (directory / f"{pair + 2:02d}-{label}.log").write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            record = {
                "pair": pair,
                "warmup": pair < 0,
                "variant": label,
                "elapsed_seconds": elapsed,
                "exit_code": result.returncode,
            }
            if worker and result.returncode == 0:
                record["worker"] = json.loads(result.stdout)
            records.append(record)
            if result.returncode != 0:
                return {"status": "failed", "records": records, "commands": commands}
    summary = {}
    for label in labels:
        kept = [r for r in records if r["variant"] == label and not r["warmup"]]
        metrics = list(kept[0]["worker"]["seconds"]) if worker else ["elapsed_seconds"]
        summary[label] = {}
        for metric in metrics:
            values = [r["worker"]["seconds"][metric] if worker else r[metric] for r in kept]
            summary[label][metric] = {
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "samples": len(values),
            }
    return {"status": "passed", "records": records, "summary": summary, "commands": commands}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["abi3", "nextest", "link", "abi-worker"])
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--abi3-python", type=Path)
    parser.add_argument("--nextest", type=Path)
    parser.add_argument("--suite", choices=["unit", "consolidated"], default="unit")
    parser.add_argument("--link-args", type=Path)
    parser.add_argument("--lld", type=Path)
    parser.add_argument("--lib-path", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.experiment == "abi-worker":
        print(json.dumps(abi_worker()))
        return 0
    if args.output is None:
        parser.error("--output is required")
    directory = args.output.parent / args.output.stem
    directory.mkdir(parents=True, exist_ok=True)
    if args.experiment == "abi3":
        if args.native_python is None or args.abi3_python is None:
            parser.error("abi3 requires both Python executables")
        commands = {
            label: [str(executable), str(Path(__file__).resolve()), "abi-worker"]
            for label, executable in (("native", args.native_python), ("abi3", args.abi3_python))
        }
        report = paired_measurements(commands, directory, worker=True)
    elif args.experiment == "nextest":
        if args.nextest is None:
            parser.error("nextest requires its executable")
        selection = ["--lib"]
        if args.suite == "consolidated":
            selection += ["--features", "ci-consolidated-tests", "--test", "ci_consolidated"]
        commands = {
            "cargo": ["cargo", "test", "--locked", *selection, "--", "--test-threads=2"],
            "nextest": [
                str(args.nextest),
                "nextest",
                "run",
                "--locked",
                *selection,
                "--test-threads",
                "2",
                "--no-fail-fast",
                "--status-level",
                "fail",
            ],
        }
        report = paired_measurements(commands, directory)
    else:
        if args.link_args is None or args.lld is None:
            parser.error("link requires captured linker arguments and the LLD executable")
        captured = json.loads(args.link_args.read_text(encoding="utf-8"))
        output = directory.resolve() / "sleepstudy.exe"
        arguments = [f"/OUT:{output}" if a.upper().startswith("/OUT:") else a for a in captured[1:]]
        # Cargo normally supplies LIB through MSVC discovery; standalone relinks
        # receive the same explicit library search paths for both linkers.
        arguments.extend(f"/LIBPATH:{path}" for path in args.lib_path)
        commands = {
            "msvc": [str(Path(captured[0])), *arguments],
            "lld": [str(args.lld), "-flavor", "link", *arguments],
        }
        report = paired_measurements(commands, directory)
        report["input_sha256"] = {a: digest(Path(a)) for a in captured[1:] if Path(a).is_file()}
        report["correctness"] = {}
        # Compare actual execution of both linked artifacts separately from timing.
        for label, command in commands.items():
            link = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if link.returncode == 0:
                run = subprocess.run(
                    [str(output)],
                    cwd=ROOT,
                    env={**os.environ, **THREAD_ENV},
                    capture_output=True,
                    text=True,
                )
                report["correctness"][label] = {
                    "exit_code": run.returncode,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                }
            else:
                report["correctness"][label] = {
                    "link_exit_code": link.returncode,
                    "stderr": link.stdout + link.stderr,
                }
    report.update(
        {
            "experiment": args.experiment,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "thread_environment": THREAD_ENV,
            "script_sha256": digest(Path(__file__)),
            "warmups_per_variant": 2,
            "requested_samples_per_variant": 10,
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary", {})}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
