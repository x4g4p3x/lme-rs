"""Post-study raw-count verification and compact evidence ledger; never refits."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import lmm_random_slope as pilot


def decision(fit, method, truth):
    if fit.get("converged") is not True:
        return None
    value = fit.get("methods", {}).get(method, {})
    if not value or "error" in value:
        return None
    if (
        not all(
            isinstance(value.get(k), (int, float)) and math.isfinite(value[k])
            for k in ("beta", "se", "p", "critical")
        )
        or value["se"] <= 0
    ):
        return None
    lower = value["beta"] - value["critical"] * value["se"]
    upper = value["beta"] + value["critical"] * value["se"]
    return value["p"] < 0.05, lower <= truth <= upper


def report(directory, optimum_path, output):
    manifest = pilot.verify_run(directory)
    summary = pilot.load(directory / "summary.json")
    optimum = pilot.load(optimum_path)
    anchor = pilot.shared.digest(directory / "summary.json")
    if optimum["source_summary_sha256"] != anchor:
        raise ValueError("diagnostic and study summary differ")
    if summary["manifest_sha256"] != pilot.shared.digest(directory / "manifest.json"):
        raise ValueError("manifest differs from source summary")
    for name, expected in summary["artifacts"].items():
        if pilot.shared.digest(directory / name) != expected:
            raise ValueError(f"study artifact changed: {name}")
    n = manifest["replicates"]
    lookup = {(r["case"], r["method"]): r for r in summary["summaries"]}
    counts, paired_counts = {}, Counter()
    for case in pilot.load(directory / "designs.json"):
        folder = directory / "cases" / case["id"]
        pairs = pilot.ordered(folder / "paired.jsonl", case["id"], n, False)
        for engine in ("candidate", "reference"):
            for method in pilot.METHODS:
                values = [decision(r[engine], method, case["beta"]) for r in pairs]
                valid = [v for v in values if v is not None]
                row = lookup[case["id"], f"{engine}:{method}"]
                actual = (len(valid), sum(v[0] for v in valid), sum(v[1] for v in valid))
                expected = (n - row["failed"], row["rejection"]["count"], row["coverage"]["count"])
                if actual != expected:
                    raise ValueError("independent raw binary counts differ from study summary")
                counts[case["id"], engine, method] = actual
        for pair in pairs:
            for method in pilot.METHODS:
                a = decision(pair["candidate"], method, case["beta"])
                b = decision(pair["reference"], method, case["beta"])
                if a is not None and b is not None:
                    paired_counts[f"{method}:evaluable"] += 1
                    paired_counts[f"{method}:reject"] += a[0] != b[0]
                    paired_counts[f"{method}:cover"] += a[1] != b[1]
    stored_paired = Counter()
    for row in summary["paired"]:
        stored_paired.update(row["binary"])
    if stored_paired != paired_counts:
        raise ValueError("independent binary disagreement counts differ")
    fit_totals = {e: Counter() for e in ("candidate", "reference")}
    for row in summary["fit_diagnostics"]:
        for engine in fit_totals:
            fit_totals[engine].update(row["engines"][engine])
    metric_flags, affected_cases, flagged_pairs = Counter(), set(), 0
    case_flags = {}
    for row in summary["paired"]:
        count = 0
        flags = Counter()
        for pair in row["numeric"]:
            bad = [v for v in pair["checks"] if not v["passed"]]
            flags.update(v["metric"] for v in bad)
            count += bool(bad)
        case_flags[row["case"]] = {
            "flagged_pairs": count,
            "metric_flags": flags,
            "binary_disagreements": row["binary"],
        }
        flagged_pairs += count
        metric_flags.update(flags)
        if count:
            affected_cases.add(row["case"])
    optimality = {}
    for engine in ("candidate", "reference"):
        rows = optimum["comparisons"]
        worst = max(rows, key=lambda r: r["engines"][engine]["criterion_gap"])
        optimality[engine] = {
            "balanced_attempts": len(rows),
            "above_optimum": sum(r["engines"][engine]["above_optimum"] for r in rows),
            "below_oracle": sum(r["engines"][engine]["below_oracle"] for r in rows),
            "maximum_gap": worst["engines"][engine]["criterion_gap"],
            "worst_case": worst["case"],
            "worst_replicate": worst["replicate"],
        }
    pilot.shared.write_json(
        output,
        {
            "schema_version": 1,
            "protocol": pilot.PROTOCOL,
            "scope": "exploratory pilot; no calibration decision; immutable original outcomes",
            "source_summary_sha256": anchor,
            "manifest_sha256": pilot.shared.digest(directory / "manifest.json"),
            "freeze_sha256": manifest["freeze_sha256"],
            "poststudy_optimum_sha256": pilot.shared.digest(optimum_path),
            "reporter_sha256": pilot.shared.digest(Path(__file__)),
            "independent_count_audit": (
                "passed for candidate/reference rejection, coverage and disagreements"
            ),
            "candidate": manifest["freeze"]["environment"],
            "engine_commit": pilot.ENGINE_COMMIT,
            "attempted_per_engine": manifest["attempted"],
            "designs": [
                {k: v for k, v in c.items() if k not in ("x", "group")}
                for c in pilot.load(directory / "designs.json")
            ],
            "fit_totals": fit_totals,
            "paired_binary": paired_counts,
            "numeric": {
                "flagged_pairs": flagged_pairs,
                "affected_cases": len(affected_cases),
                "metric_flags": metric_flags,
                "by_case": case_flags,
            },
            "poststudy_global_optimum": optimality,
            "case_method_evidence": summary["summaries"],
            "execution": {
                e: pilot.load(directory / f"{e}-execution.json") for e in ("candidate", "reference")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--optimum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.directory, args.optimum, args.output)
