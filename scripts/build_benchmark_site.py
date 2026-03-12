#!/usr/bin/env python3
"""Build chart-friendly benchmark site data from cross-language benchmark JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data files for the benchmark dashboard."
    )
    parser.add_argument(
        "--cross-language-json",
        required=True,
        help="Path to the cross-language benchmark JSON input.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the generated dashboard data should be written.",
    )
    parser.add_argument(
        "--run-url",
        default="",
        help="GitHub Actions run URL for the current benchmark run.",
    )
    parser.add_argument(
        "--release-url",
        default="",
        help="GitHub release URL, if the run corresponds to a release tag.",
    )
    parser.add_argument(
        "--criterion-asset-name",
        default="",
        help="Criterion archive asset filename for the matching release.",
    )
    parser.add_argument(
        "--cross-language-asset-name",
        default="",
        help="Cross-language JSON asset filename for the matching release.",
    )
    parser.add_argument(
        "--ref-name",
        default="",
        help="Git ref name for the benchmark run.",
    )
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float | None:
    filtered = [value for value in values if value > 0]
    if not filtered:
        return None
    return math.exp(sum(math.log(value) for value in filtered) / len(filtered))


def main() -> int:
    args = parse_args()

    input_path = Path(args.cross_language_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    config = payload["config"]
    cases = config["cases"]
    implementations = config["implementations"]

    results_by_case: dict[str, dict[str, dict[str, float | str]]] = {
        case: {} for case in cases
    }
    for result in payload["results"]:
        summary = result["summary"]
        results_by_case[result["case"]][result["implementation"]] = {
            "median_seconds": summary["median_seconds"],
            "mean_seconds": summary["mean_seconds"],
            "min_seconds": summary["min_seconds"],
            "max_seconds": summary["max_seconds"],
        }

    case_cards = []
    ratios_by_implementation: dict[str, list[float]] = {
        implementation: [] for implementation in implementations if implementation != "rust"
    }

    for case in cases:
        entries = []
        case_results = results_by_case.get(case, {})
        rust_seconds = None
        if "rust" in case_results:
            rust_seconds = float(case_results["rust"]["median_seconds"])

        max_seconds = max(
            (float(item["median_seconds"]) for item in case_results.values()),
            default=0.0,
        )
        fastest_impl = None
        if case_results:
            fastest_impl = min(
                case_results.items(),
                key=lambda item: float(item[1]["median_seconds"]),
            )[0]

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

            entries.append(
                {
                    "implementation": implementation,
                    "median_seconds": median_seconds,
                    "mean_seconds": float(result["mean_seconds"]),
                    "min_seconds": float(result["min_seconds"]),
                    "max_seconds": float(result["max_seconds"]),
                    "relative_to_rust": relative_to_rust,
                    "is_fastest": implementation == fastest_impl,
                    "width_fraction": (median_seconds / max_seconds) if max_seconds else 0.0,
                }
            )

        case_cards.append(
            {
                "case": case,
                "fastest_implementation": fastest_impl,
                "max_seconds": max_seconds,
                "entries": entries,
            }
        )

    implementation_summary = []
    for implementation in implementations:
        if implementation == "rust":
            implementation_summary.append(
                {
                    "implementation": implementation,
                    "geometric_mean_relative_to_rust": 1.0,
                }
            )
            continue

        implementation_summary.append(
            {
                "implementation": implementation,
                "geometric_mean_relative_to_rust": geometric_mean(
                    ratios_by_implementation[implementation]
                ),
            }
        )

    asset_urls = {
        "criterion": None,
        "cross_language": None,
    }
    if args.release_url and args.ref_name:
        release_base = args.release_url.rsplit("/tag/", 1)[0]
        download_root = f"{release_base}/download/{args.ref_name}/"
        if args.criterion_asset_name:
            asset_urls["criterion"] = f"{download_root}{args.criterion_asset_name}"
        if args.cross_language_asset_name:
            asset_urls["cross_language"] = f"{download_root}{args.cross_language_asset_name}"

    site_payload = {
        "generated_at": payload["generated_at"],
        "git_sha": payload["git_sha"],
        "ref_name": args.ref_name,
        "run_url": args.run_url or None,
        "release_url": args.release_url or None,
        "assets": {
            "criterion": args.criterion_asset_name or None,
            "cross_language": args.cross_language_asset_name or None,
        },
        "asset_urls": asset_urls,
        "machine_info": {
            "platform": payload["machine_info"]["platform"],
            "machine": payload["machine_info"]["machine"],
            "cpu_count": payload["machine_info"]["cpu_count"],
        },
        "runtime_versions": payload["runtime_versions"],
        "config": {
            "cases": cases,
            "implementations": implementations,
            "warmups": config["warmups"],
            "repeats": config["repeats"],
            "timeout_seconds": config["timeout_seconds"],
        },
        "implementation_summary": implementation_summary,
        "cases": case_cards,
        "failures": payload["failures"],
    }

    (output_dir / "latest.json").write_text(
        json.dumps(site_payload, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
