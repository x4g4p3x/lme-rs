#!/usr/bin/env python3
"""Compare fixed-effect coefficients across R, Julia, Rust, and Python.

Loads expected tolerances from ``tests/data/golden_parity_manifest.json`` and
runs language-specific parity exporters under ``comparisons/parity/``.

Example::

    python scripts/verify_cross_language_parity.py
    python scripts/verify_cross_language_parity.py --implementations r,rust
"""

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
MANIFEST = REPO_ROOT / "tests" / "data" / "golden_parity_manifest.json"

# Maps parity exporter case id -> golden manifest case id.
PARITY_CASES: dict[str, str] = {
    "glmm_probit": "cbpp_binomial_probit",
    "glmm_weighted": "cbpp_binomial_weighted",
    "glmm_offset": "grouseticks_poisson_offset",
    "sleepstudy_offset": "sleepstudy_offset_reml",
}


@dataclass(frozen=True)
class ExpectedCoef:
    name: str
    value: float
    tolerance: float


@dataclass(frozen=True)
class ParityCase:
    case_id: str
    golden_id: str
    expected: list[ExpectedCoef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=",".join(PARITY_CASES),
        help="Comma-separated parity case ids.",
    )
    parser.add_argument(
        "--implementations",
        default="r,rust,julia,python",
        help="Comma-separated implementations to run (r, rust, julia, python).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-command timeout in seconds.",
    )
    parser.add_argument(
        "--skip-rust-build",
        action="store_true",
        help="Skip `cargo build --release --example parity_export`.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_manifest() -> dict[str, Any]:
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_parity_cases(selected: list[str]) -> list[ParityCase]:
    manifest = load_manifest()
    by_id = {case["id"]: case for case in manifest["cases"]}
    cases: list[ParityCase] = []
    for case_id in selected:
        golden_id = PARITY_CASES.get(case_id)
        if golden_id is None:
            raise SystemExit(f"unknown parity case: {case_id}")
        golden = by_id.get(golden_id)
        if golden is None:
            raise SystemExit(f"golden manifest missing case: {golden_id}")
        expected = [
            ExpectedCoef(
                name=item["name"],
                value=float(item["value"]),
                tolerance=float(item["tolerance"]),
            )
            for item in golden["expected"]["coefficients"]
        ]
        cases.append(ParityCase(case_id=case_id, golden_id=golden_id, expected=expected))
    return cases


def find_rscript() -> str | None:
    env = os.environ.get("RSCRIPT")
    if env and Path(env).exists():
        return env
    found = shutil.which("Rscript")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def run_command(command: list[str], timeout: int, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def parse_payload(stdout: str) -> dict[str, Any]:
    # Exporters may print warnings before JSON; take the last non-empty line.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON object found in exporter output:\n{stdout}")


def run_r(case_id: str, timeout: int) -> dict[str, float]:
    rscript = find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found (set RSCRIPT or add R to PATH)")
    script = REPO_ROOT / "comparisons" / "parity" / f"{case_id}.R"
    stdout = run_command([rscript, str(script)], timeout=timeout, cwd=REPO_ROOT)
    payload = parse_payload(stdout)
    return {str(k): float(v) for k, v in payload["coefficients"].items()}


def run_julia(case_id: str, timeout: int) -> dict[str, float]:
    julia = shutil.which("julia")
    if julia is None:
        raise RuntimeError("julia not found on PATH")
    script = REPO_ROOT / "comparisons" / "parity" / f"{case_id}.jl"
    stdout = run_command([julia, str(script)], timeout=timeout, cwd=REPO_ROOT)
    payload = parse_payload(stdout)
    return {str(k): float(v) for k, v in payload["coefficients"].items()}


def run_rust(case_id: str, timeout: int, skip_build: bool) -> dict[str, float]:
    if not skip_build:
        run_command(
            ["cargo", "build", "--release", "--locked", "--example", "parity_export"],
            timeout=timeout,
            cwd=REPO_ROOT,
        )
    binary = REPO_ROOT / "target" / "release" / "examples" / "parity_export"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")
    stdout = run_command([str(binary), case_id], timeout=timeout, cwd=REPO_ROOT)
    payload = parse_payload(stdout)
    return {str(k): float(v) for k, v in payload["coefficients"].items()}


def run_python(case_id: str) -> dict[str, float]:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("polars is required for python parity checks") from exc

    try:
        import lme_python
    except ImportError as exc:
        raise RuntimeError(
            "lme_python extension not installed; run `task python` or maturin develop"
        ) from exc

    if case_id == "glmm_probit":
        df = pl.read_csv(REPO_ROOT / "tests/data/cbpp_binary.csv")
        fit = lme_python.glmer(
            "y ~ period2 + period3 + period4 + (1 | herd)",
            data=df,
            family_name="binomial",
            link_name="probit",
        )
    elif case_id == "glmm_weighted":
        df = pl.read_csv(REPO_ROOT / "tests/data/cbpp_binary_weighted.csv")
        fit = lme_python.glmer_weighted(
            "y ~ period2 + period3 + period4 + (1 | herd)",
            data=df,
            family_name="binomial",
            weights=df["prior_w"].to_list(),
        )
    elif case_id == "glmm_offset":
        df = pl.read_csv(REPO_ROOT / "tests/data/grouseticks.csv")
        fit = lme_python.glmer(
            "TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD)",
            data=df,
            family_name="poisson",
        )
    elif case_id == "sleepstudy_offset":
        df = pl.read_csv(REPO_ROOT / "tests/data/sleepstudy.csv")
        fit = lme_python.lmer(
            "Reaction ~ Days + offset(OffsetDays10) + (Days | Subject)",
            data=df,
            reml=True,
        )
    else:
        raise RuntimeError(f"no python parity handler for {case_id}")

    names = list(fit.fixed_names)
    coef = list(fit.coefficients)
    return dict(zip(names, coef))


def assert_against_expected(
    case: ParityCase,
    implementation: str,
    actual: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    for check in case.expected:
        if check.name not in actual:
            failures.append(f"{check.name}: missing from {implementation} output")
            continue
        diff = abs(actual[check.name] - check.value)
        if diff > check.tolerance:
            failures.append(
                f"{check.name}: {implementation}={actual[check.name]:.6g} "
                f"expected={check.value:.6g} tol={check.tolerance} diff={diff:.6g}"
            )
    return failures


def max_pairwise_diff(
    results: dict[str, dict[str, float]],
    names: list[str],
) -> float:
    impls = list(results.keys())
    max_diff = 0.0
    for i, left in enumerate(impls):
        for right in impls[i + 1 :]:
            for name in names:
                if name in results[left] and name in results[right]:
                    max_diff = max(max_diff, abs(results[left][name] - results[right][name]))
    return max_diff


def main() -> int:
    args = parse_args()
    cases = load_parity_cases(split_csv(args.cases))
    implementations = split_csv(args.implementations)

    all_failures: list[str] = []
    for case in cases:
        print(f"\n=== {case.case_id} (golden: {case.golden_id}) ===")
        results: dict[str, dict[str, float]] = {}
        for impl in implementations:
            try:
                if impl == "r":
                    results[impl] = run_r(case.case_id, args.timeout)
                elif impl == "julia":
                    results[impl] = run_julia(case.case_id, args.timeout)
                elif impl == "rust":
                    results[impl] = run_rust(
                        case.case_id, args.timeout, args.skip_rust_build
                    )
                elif impl == "python":
                    results[impl] = run_python(case.case_id)
                else:
                    raise RuntimeError(f"unknown implementation: {impl}")
            except RuntimeError as exc:
                print(f"  [{impl}] SKIP: {exc}")
                continue

            failures = assert_against_expected(case, impl, results[impl])
            if failures:
                print(f"  [{impl}] FAIL vs golden reference")
                for line in failures:
                    print(f"    - {line}")
                all_failures.extend(f"{case.case_id}/{impl}: {line}" for line in failures)
            else:
                print(f"  [{impl}] OK vs golden reference")

        if len(results) >= 2:
            names = [check.name for check in case.expected]
            cross = max_pairwise_diff(results, names)
            cross_tol = max(check.tolerance for check in case.expected)
            status = "OK" if cross <= cross_tol else "WARN"
            print(
                f"  [cross-language] {status}: max pairwise diff={cross:.6g} "
                f"(limit={cross_tol}) among {', '.join(results)}"
            )
            if cross > cross_tol:
                all_failures.append(
                    f"{case.case_id}/cross-language: max diff {cross:.6g} > {cross_tol}"
                )

    if all_failures:
        print("\nParity verification failed:")
        for line in all_failures:
            print(f"  - {line}")
        return 1

    print("\nAll requested parity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
