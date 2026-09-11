#!/usr/bin/env python3
"""Compare Argmin and Basin using independent, alternating fair-harness processes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ["ABBA", "BAAB", "ABBA"]
MARGIN = 1.05


def summarize(reports: list[dict]) -> list[dict]:
    """Use execution blocks as the resampling unit; qualify every measured fit."""
    if len(reports) != 12 or [(r["block"], r["position"], r["backend"]) for r in reports] != [
        (block, position, {"A": "argmin", "B": "basin"}[letter])
        for block, order in enumerate(ORDERS)
        for position, letter in enumerate(order)
    ]:
        raise ValueError("Expected complete ABBA / BAAB / ABBA execution order")
    names = [row["case"] for row in reports[0]["report"]["results"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected nonempty, unique cases")
    for run in reports:
        if [row["case"] for row in run["report"]["results"]] != names:
            raise ValueError("Case coverage changed between processes")
        reference_report = reports[0]["report"]
        for field in ["source_content_sha256", "cargo_lock_sha256", "harness_sha256"]:
            value = run["report"].get("provenance", {}).get(field)
            if not value or value != reference_report.get("provenance", {}).get(field):
                raise ValueError(f"Missing or changed provenance: {field}")
        if run["report"].get("fixtures") != reference_report.get("fixtures"):
            raise ValueError("Fixtures changed between processes")
    rng = random.Random(260911)
    rows = []
    for index, name in enumerate(names):
        selected = [(run, run["report"]["results"][index]) for run in reports]
        reference = selected[0][1]["fit_checks"][0]
        errors = set()
        metrics = [metric for metric in ["cold_fit", "fit_prepared"] if metric in selected[0][1]]
        for run, result in selected:
            if result.get("optimizer_backend") != run["backend"]:
                errors.add("backend label mismatch")
            for metric in metrics:
                samples = result[metric]["samples_seconds"]
                if len(samples) < 10 or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError("Expected at least ten positive finite timing samples")
                key = "fit_checks" if metric == "cold_fit" else "prepared_fit_checks"
                checks = result.get(key, [])
                if len(checks) != len(samples):
                    errors.add("missing measured-fit checks")
                for check in checks:
                    if check.get("converged") is not True:
                        errors.add("nonconverged measured fit")
                    if (
                        not math.isfinite(check["objective"])
                        or not check["coefficients"]
                        or any(not math.isfinite(value) for value in check["coefficients"])
                    ):
                        errors.add("invalid numerical fit")
                    if not math.isclose(
                        check["objective"], reference["objective"], rel_tol=1e-8, abs_tol=1e-6
                    ):
                        errors.add("objective mismatch")
                    if len(check["coefficients"]) != len(reference["coefficients"]) or any(
                        not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
                        for a, b in zip(check["coefficients"], reference["coefficients"])
                    ):
                        errors.add("coefficient mismatch")
        row = dict(case=name, fit_agreement=not errors, reasons=sorted(errors))
        for metric in metrics:
            by_backend = {
                backend: [
                    statistics.median(result[metric]["samples_seconds"])
                    for run, result in selected
                    if run["backend"] == backend
                ]
                for backend in ["argmin", "basin"]
            }
            block_ratios = []
            for block in range(3):
                medians = {
                    backend: statistics.median(
                        [
                            statistics.median(result[metric]["samples_seconds"])
                            for run, result in selected
                            if run["backend"] == backend and run["block"] == block
                        ]
                    )
                    for backend in ["argmin", "basin"]
                }
                block_ratios.append(medians["basin"] / medians["argmin"])
            samples = sorted(statistics.median(rng.choices(block_ratios, k=3)) for _ in range(2000))
            low, high = samples[49], samples[1949]
            row[metric] = dict(
                argmin_seconds=statistics.median(by_backend["argmin"]),
                basin_seconds=statistics.median(by_backend["basin"]),
                basin_over_argmin=statistics.median(block_ratios),
                block_ratios=block_ratios,
                ratio_interval_95=[low, high],
                competitive=not errors and high <= MARGIN,
            )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-results/optimizer-comparison.json")
    )
    parser.add_argument(
        "--argmin-target", type=Path, help="Reuse a prebuilt Argmin target directory"
    )
    parser.add_argument("--basin-target", type=Path, help="Reuse a prebuilt Basin target directory")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent / (args.output.stem + "-runs")
    raw_dir.mkdir(exist_ok=True)
    reports = []
    for block, order in enumerate(ORDERS):
        for position, letter in enumerate(order):
            backend = {"A": "argmin", "B": "basin"}[letter]
            path = (raw_dir / f"{block}-{position}-{backend}.json").resolve()
            command = [
                sys.executable,
                "scripts/run_fair_rust_julia_benchmark.py",
                "--implementations",
                "rust",
                "--with-phases",
                "--warmups",
                "3",
                "--repeats",
                "11",
                "--threads",
                "1",
                "--output",
                str(path),
            ]
            if backend == "basin":
                command += ["--rust-features", "basin"]
            env = os.environ.copy()
            target = getattr(args, backend + "_target")
            if target:
                env["CARGO_TARGET_DIR"] = str(target.resolve())
                command += ["--skip-rust-build"]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            reports.append(
                dict(
                    block=block,
                    position=position,
                    backend=backend,
                    report=json.loads(path.read_text(encoding="utf-8")),
                )
            )
    payload = dict(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        protocol=dict(
            block_orders=ORDERS,
            warmups=3,
            repeats=11,
            threads=1,
            processes_per_backend=6,
            samples_per_case_backend_metric=66,
            noninferiority_margin=MARGIN,
            bootstrap_resamples=2000,
            interval_unit="paired execution block; three blocks on one workstation",
        ),
        caveat=(
            "Intervals over three blocks are limited workstation evidence; use other suites "
            "and supported-platform checks before changing defaults."
        ),
        cases=summarize(reports),
        runs=reports,
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
