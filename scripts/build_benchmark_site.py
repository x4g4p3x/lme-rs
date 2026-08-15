#!/usr/bin/env python3
"""Build chart-friendly JSON for the public benchmark dashboard.

Default inputs are the checked-in engineering references:
fair Rust vs MixedModels.jl (tier A) and the Julia-free external timings.
Optional CI overlays add GitHub-hosted runner fair results and whole-script
example timings; those are not the completion baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAIR = (
    REPO_ROOT / "benchmarks" / "fair-rust-julia-reference-2026-07-22-full-tier-a.json"
)
DEFAULT_EXTERNAL = (
    REPO_ROOT
    / "benchmarks"
    / "external-nlmm-inference-python-timings-2026-08-14.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "benchmarks" / "data"
SITE_SCHEMA_VERSION = 2
REPO_BLOB = "https://github.com/x4g4p3x/lme-rs/blob/master"
IMPL_LABELS = {
    "rust": "Rust",
    "julia": "Julia",
    "python": "Python",
    "r": "R",
    "r_lme4": "R lme4",
    "r_lmerTest": "R lmerTest",
    "python_lme_python": "Python FFI",
}
FAMILY_LABELS = {
    "lmm_fit": "LMM fit",
    "nlmm_fit": "NLMM fit",
    "post_fit_inference": "Post-fit inference",
    "python_ffi": "Python FFI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data files for the benchmark dashboard."
    )
    parser.add_argument(
        "--fair-json",
        default=str(DEFAULT_FAIR),
        help="Checked-in fair Rust vs Julia reference JSON (completion baseline).",
    )
    parser.add_argument(
        "--external-json",
        default=str(DEFAULT_EXTERNAL),
        help="Checked-in nlmer / inference / Python FFI timings JSON.",
    )
    parser.add_argument(
        "--ci-fair-json",
        default="",
        help="Optional GitHub Actions fair-harness JSON (not the completion baseline).",
    )
    parser.add_argument(
        "--cross-language-json",
        default="",
        help="Optional whole-script cross-language JSON (startup/JIT inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory where latest.json should be written.",
    )
    parser.add_argument("--run-url", default="")
    parser.add_argument("--release-url", default="")
    parser.add_argument("--criterion-asset-name", default="")
    parser.add_argument("--cross-language-asset-name", default="")
    parser.add_argument("--fair-asset-name", default="")
    parser.add_argument("--ref-name", default="")
    parser.add_argument(
        "--repo-blob-url",
        default=REPO_BLOB,
        help="Base URL for methodology/coverage links.",
    )
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float | None:
    filtered = [value for value in values if value > 0]
    if not filtered:
        return None
    return math.exp(sum(math.log(value) for value in filtered) / len(filtered))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def display_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def bar_entries(named_seconds: list[tuple[str, float]]) -> list[dict[str, Any]]:
    positive = [(name, seconds) for name, seconds in named_seconds if seconds > 0]
    max_seconds = max((seconds for _, seconds in positive), default=0.0)
    fastest = min(positive, key=lambda item: item[1])[0] if positive else None
    entries = []
    for name, seconds in named_seconds:
        entries.append(
            {
                "implementation": name,
                "label": IMPL_LABELS.get(name, name),
                "median_seconds": seconds,
                "is_fastest": name == fastest,
                "width_fraction": (seconds / max_seconds) if max_seconds else 0.0,
            }
        )
    return entries


def index_results(results: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        key = (str(result.get("implementation")), str(result.get("case")))
        indexed[key] = result
    return indexed


def metric_block(metric: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    rust_seconds = metric.get("rust_median_seconds")
    julia_seconds = metric.get("julia_median_seconds")
    named: list[tuple[str, float]] = []
    if rust_seconds is not None:
        named.append(("rust", float(rust_seconds)))
    if julia_seconds is not None:
        named.append(("julia", float(julia_seconds)))
    return {
        "rust_median_seconds": rust_seconds,
        "julia_median_seconds": julia_seconds,
        "rust_over_julia_median": metric.get("rust_over_julia_median"),
        "faster_implementation": metric.get("faster_implementation"),
        "meets_target": metric.get("meets_target"),
        "entries": bar_entries(named),
    }


def transform_fair(payload: dict[str, Any], *, label: str, source_path: str) -> dict[str, Any]:
    results_by_key = index_results(payload.get("results") or [])
    cases = []
    cold_ratios: list[float] = []
    prepared_ratios: list[float] = []
    cold_passes = 0
    cold_total = 0
    prepared_passes = 0
    prepared_total = 0

    for comparison in payload.get("comparisons") or []:
        case_name = str(comparison["case"])
        metrics = {str(item["metric"]): item for item in comparison.get("metrics") or []}
        rust_row = results_by_key.get(("rust", case_name), {})
        julia_row = results_by_key.get(("julia", case_name), {})
        cold = metric_block(metrics.get("cold_fit"))
        prepared = metric_block(metrics.get("fit_prepared_vs_julia_fit"))
        prepare = metrics.get("prepare_lmer_rust_only") or {}

        if cold and cold.get("rust_over_julia_median") is not None:
            ratio = float(cold["rust_over_julia_median"])
            cold_ratios.append(ratio)
            cold_total += 1
            if cold.get("meets_target"):
                cold_passes += 1
        if prepared and prepared.get("rust_over_julia_median") is not None:
            ratio = float(prepared["rust_over_julia_median"])
            prepared_ratios.append(ratio)
            prepared_total += 1
            if prepared.get("meets_target"):
                prepared_passes += 1

        cases.append(
            {
                "case": case_name,
                "formula": rust_row.get("formula") or julia_row.get("formula"),
                "model": comparison.get("model") or rust_row.get("model"),
                "n_obs": rust_row.get("n_obs") or julia_row.get("n_obs"),
                "reference": comparison.get("reference"),
                "cold_fit": cold,
                "fit_prepared": prepared,
                "prepare_lmer_seconds": prepare.get("rust_median_seconds"),
            }
        )

    config = payload.get("config") or {}
    return {
        "label": label,
        "source_path": source_path,
        "generated_at": payload.get("generated_at"),
        "git_sha": payload.get("git_sha"),
        "methodology": payload.get("methodology") or {},
        "machine_info": payload.get("machine_info") or {},
        "runtime_versions": payload.get("runtime_versions") or {},
        "config": {
            "cases": config.get("cases") or [item["case"] for item in cases],
            "implementations": config.get("implementations") or ["rust", "julia"],
            "warmups": config.get("warmups"),
            "repeats": config.get("repeats"),
            "with_phases": config.get("with_phases"),
            "target_ratio": config.get("target_ratio")
            or (payload.get("methodology") or {}).get("target_ratio_cold_fit"),
        },
        "summary": {
            "cold_fit_passes": cold_passes,
            "cold_fit_cases": cold_total,
            "prepared_passes": prepared_passes,
            "prepared_cases": prepared_total,
            "geometric_mean_rust_over_julia_cold_fit": geometric_mean(cold_ratios),
            "median_rust_over_julia_cold_fit": (
                statistics.median(cold_ratios) if cold_ratios else None
            ),
            "geometric_mean_rust_over_julia_prepared": geometric_mean(prepared_ratios),
            "median_rust_over_julia_prepared": (
                statistics.median(prepared_ratios) if prepared_ratios else None
            ),
        },
        "cases": cases,
        "failures": payload.get("failures") or [],
    }


def transform_cross_language(
    payload: dict[str, Any], *, source_path: str
) -> dict[str, Any]:
    config = payload["config"]
    cases = list(config["cases"])
    implementations = list(config["implementations"])
    results_by_case: dict[str, dict[str, dict[str, float]]] = {case: {} for case in cases}
    for result in payload.get("results") or []:
        summary = result["summary"]
        results_by_case[result["case"]][result["implementation"]] = {
            "median_seconds": summary["median_seconds"],
            "mean_seconds": summary["mean_seconds"],
            "min_seconds": summary["min_seconds"],
            "max_seconds": summary["max_seconds"],
        }

    case_cards = []
    ratios_by_implementation: dict[str, list[float]] = {
        implementation: []
        for implementation in implementations
        if implementation != "rust"
    }
    for case in cases:
        case_results = results_by_case.get(case, {})
        rust_seconds = None
        if "rust" in case_results:
            rust_seconds = float(case_results["rust"]["median_seconds"])
        named = [
            (implementation, float(item["median_seconds"]))
            for implementation, item in case_results.items()
        ]
        entries = []
        for implementation in implementations:
            result = case_results.get(implementation)
            if result is None:
                continue
            median_seconds = float(result["median_seconds"])
            relative_to_rust = None
            if rust_seconds and rust_seconds > 0:
                relative_to_rust = median_seconds / rust_seconds
                if implementation != "rust":
                    ratios_by_implementation[implementation].append(relative_to_rust)
            bar = next(
                item for item in bar_entries(named) if item["implementation"] == implementation
            )
            entries.append(
                {
                    **bar,
                    "mean_seconds": float(result["mean_seconds"]),
                    "min_seconds": float(result["min_seconds"]),
                    "max_seconds": float(result["max_seconds"]),
                    "relative_to_rust": relative_to_rust,
                }
            )
        fastest = next((item["implementation"] for item in entries if item["is_fastest"]), None)
        case_cards.append(
            {
                "case": case,
                "fastest_implementation": fastest,
                "entries": entries,
            }
        )

    implementation_summary = []
    for implementation in implementations:
        if implementation == "rust":
            implementation_summary.append(
                {
                    "implementation": implementation,
                    "label": IMPL_LABELS.get(implementation, implementation),
                    "geometric_mean_relative_to_rust": 1.0,
                }
            )
            continue
        implementation_summary.append(
            {
                "implementation": implementation,
                "label": IMPL_LABELS.get(implementation, implementation),
                "geometric_mean_relative_to_rust": geometric_mean(
                    ratios_by_implementation[implementation]
                ),
            }
        )

    return {
        "label": "Whole-script example timings",
        "source_path": source_path,
        "generated_at": payload.get("generated_at"),
        "git_sha": payload.get("git_sha"),
        "machine_info": payload.get("machine_info") or {},
        "runtime_versions": payload.get("runtime_versions") or {},
        "config": {
            "cases": cases,
            "implementations": implementations,
            "warmups": config.get("warmups"),
            "repeats": config.get("repeats"),
            "timeout_seconds": config.get("timeout_seconds"),
        },
        "implementation_summary": implementation_summary,
        "cases": case_cards,
        "failures": payload.get("failures") or [],
        "caveat": (
            "Times whole example scripts, including process startup and Julia JIT. "
            "Small fixtures favor a prebuilt Rust binary; do not use these bars as "
            "fit-only throughput vs MixedModels.jl."
        ),
    }


def transform_external(payload: dict[str, Any], *, source_path: str) -> dict[str, Any]:
    families: dict[str, dict[str, dict[str, Any]]] = {}
    for report in payload.get("reports") or []:
        family = str(report.get("family") or "other")
        case = str(report["case"])
        families.setdefault(family, {}).setdefault(case, {})[str(report["implementation"])] = (
            report
        )

    family_cards = []
    for family, cases in families.items():
        case_cards = []
        for case, impls in cases.items():
            named = [
                (implementation, float(report["summary"]["median_seconds"]))
                for implementation, report in impls.items()
            ]
            sample = next(iter(impls.values()))
            case_cards.append(
                {
                    "case": case,
                    "formula": sample.get("formula"),
                    "n_obs": sample.get("n_obs"),
                    "entries": bar_entries(named),
                }
            )
        family_cards.append(
            {
                "family": family,
                "label": FAMILY_LABELS.get(family, family),
                "cases": case_cards,
            }
        )

    return {
        "label": "nlmer, inference, and Python FFI",
        "source_path": source_path,
        "generated_at": payload.get("generated_at"),
        "host": payload.get("host") or {},
        "warmups": payload.get("warmups"),
        "repeats": payload.get("repeats"),
        "families": family_cards,
        "skipped": payload.get("skipped") or [],
    }


def asset_urls(release_url: str, ref_name: str, names: dict[str, str]) -> dict[str, str | None]:
    urls: dict[str, str | None] = {
        "criterion": None,
        "cross_language": None,
        "fair": None,
    }
    if not release_url or not ref_name:
        return urls
    release_base = release_url.rsplit("/tag/", 1)[0]
    download_root = f"{release_base}/download/{ref_name}/"
    for key, filename in names.items():
        if filename:
            urls[key] = f"{download_root}{filename}"
    return urls


def build_site_payload(args: argparse.Namespace) -> dict[str, Any]:
    fair_path = Path(args.fair_json)
    if not fair_path.is_absolute():
        fair_path = REPO_ROOT / fair_path
    fair = transform_fair(
        load_json(fair_path),
        label="Engineering reference (workstation)",
        source_path=display_source(fair_path),
    )
    if not fair["cases"]:
        raise SystemExit(f"No fair-harness cases found in {fair_path}")

    external = None
    if args.external_json:
        external_path = Path(args.external_json)
        if not external_path.is_absolute():
            external_path = REPO_ROOT / external_path
        if external_path.exists():
            external = transform_external(
                load_json(external_path),
                source_path=display_source(external_path),
            )

    ci_fair = None
    if args.ci_fair_json:
        ci_path = Path(args.ci_fair_json)
        if not ci_path.is_absolute():
            ci_path = REPO_ROOT / ci_path
        ci_fair = transform_fair(
            load_json(ci_path),
            label="GitHub-hosted runner",
            source_path=display_source(ci_path),
        )

    cross_language = None
    if args.cross_language_json:
        cross_path = Path(args.cross_language_json)
        if not cross_path.is_absolute():
            cross_path = REPO_ROOT / cross_path
        cross_language = transform_cross_language(
            load_json(cross_path),
            source_path=display_source(cross_path),
        )

    summary = fair["summary"]
    names = {
        "criterion": args.criterion_asset_name,
        "cross_language": args.cross_language_asset_name,
        "fair": args.fair_asset_name,
    }
    return {
        "schema_version": SITE_SCHEMA_VERSION,
        "generated_at": fair.get("generated_at"),
        "ref_name": args.ref_name or None,
        "headline": {
            "title": "Fair fit-only vs MixedModels.jl",
            "source_label": fair["label"],
            "cold_fit_target": fair["config"].get("target_ratio") or 1.0,
            "cold_fit_passes": summary["cold_fit_passes"],
            "cold_fit_cases": summary["cold_fit_cases"],
            "geometric_mean_rust_over_julia_cold_fit": summary[
                "geometric_mean_rust_over_julia_cold_fit"
            ],
            "median_rust_over_julia_cold_fit": summary["median_rust_over_julia_cold_fit"],
        },
        "links": {
            "run_url": args.run_url or None,
            "release_url": args.release_url or None,
            "methodology_url": f"{args.repo_blob_url}/BENCHMARKS.md",
            "coverage_url": f"{args.repo_blob_url}/BENCHMARK_COVERAGE.md",
            "repo_url": "https://github.com/x4g4p3x/lme-rs",
            "assets": {
                "criterion": args.criterion_asset_name or None,
                "cross_language": args.cross_language_asset_name or None,
                "fair": args.fair_asset_name or None,
            },
            "asset_urls": asset_urls(args.release_url, args.ref_name, names),
        },
        "fair": fair,
        "ci_fair": ci_fair,
        "cross_language": cross_language,
        "external": external,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_site_payload(args)
    (output_dir / "latest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
