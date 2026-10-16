"""Verify v2 counts from retained pairs; report numerical gates without refitting."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import lmm_random_slope_v2 as pilot
from report_lmm_random_slope import decision


def valid_decision(fit, method, truth):
    if pilot.covariance_diagnostic(fit.get("covariance"), fit.get("sigma2"))["state"] == "invalid":
        return None
    return decision(fit, method, truth)


def pair_counts(candidate, reference):
    if len(candidate) != len(reference):
        raise ValueError("paired inventories differ")
    common = [(a, b) for a, b in zip(candidate, reference) if a is not None and b is not None]
    return {
        "evaluable": len(common),
        "reject": sum(a[0] != b[0] for a, b in common),
        "cover": sum(a[1] != b[1] for a, b in common),
    }


def validate_balanced(row):
    if not isinstance(row["evaluable"], bool):
        raise ValueError("balanced evaluability must be boolean")
    if not row["evaluable"]:
        return
    metrics = [c["metric"] for c in row["checks"]]
    expected = {"criterion", "sigma2", "covariance_00", "covariance_01", "covariance_11"}
    if (
        len(metrics) != len(expected)
        or set(metrics) != expected
        or not isinstance(row["rank_match"], bool)
    ):
        raise ValueError("incomplete balanced metric/rank inventory")
    for check in row["checks"]:
        delta, limit = check["difference"], check["limit"]
        if (
            not math.isfinite(delta)
            or not math.isfinite(limit)
            or limit < 0
            or not isinstance(check["passed"], bool)
            or check["passed"] != (abs(delta) <= limit)
        ):
            raise ValueError("inconsistent balanced numerical check")


def report(directory, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = pilot.verify_run(directory)
    summary = pilot.load(directory / "summary.json")
    if summary["manifest_sha256"] != pilot.shared.digest(directory / "manifest.json"):
        raise ValueError("summary manifest identity differs")
    for name, expected in summary["artifacts"].items():
        if pilot.shared.digest(directory / name) != expected:
            raise ValueError("retained study artifact changed")
    lookup = {(r["case"], r["method"]): r for r in summary["summaries"]}
    seen, converged, binary = set(), Counter(), Counter()
    paired_lookup = {r["case"]: r for r in summary["paired"]}
    for case in pilot.load(directory / "designs.json"):
        pairs = pilot.ordered(
            directory / "cases" / case["id"] / "paired.jsonl",
            case["id"],
            manifest["replicates"],
            False,
        )
        for pair in pairs:
            seen.add((case["id"], pair["replicate"]))
            for engine in ("candidate", "reference"):
                converged[engine] += pair[engine].get("converged") is True
        decisions = {}
        for engine in ("candidate", "reference"):
            for method in pilot.METHODS:
                values = [valid_decision(p[engine], method, case["beta"]) for p in pairs]
                decisions[engine, method] = values
                values = [v for v in values if v is not None]
                actual = (len(values), sum(v[0] for v in values), sum(v[1] for v in values))
                recorded = lookup[case["id"], f"{engine}:{method}"]
                expected = (
                    recorded["rejection"]["evaluable"],
                    recorded["rejection"]["count"],
                    recorded["coverage"]["count"],
                )
                if actual != expected:
                    raise ValueError("independent rejection/coverage count audit failed")
        case_binary = Counter()
        for method in pilot.METHODS:
            case_binary.update(
                {
                    f"{method}:{key}": v
                    for key, v in pair_counts(
                        decisions["candidate", method], decisions["reference", method]
                    ).items()
                }
            )
        if case_binary != Counter(paired_lookup[case["id"]]["binary"]):
            raise ValueError("independent paired disagreement audit failed")
        binary.update(case_binary)
    if len(seen) != manifest["attempted"]:
        raise ValueError("missing/duplicate case inventory")
    expected_balanced = {
        (c["id"], i)
        for c in pilot.load(directory / "designs.json")
        if len(set(c["sizes"])) == 1
        for i in range(manifest["replicates"])
    }
    observed_balanced = [(r["case"], r["replicate"]) for r in summary["balanced_checks"]]
    if (
        len(observed_balanced) != len(expected_balanced)
        or set(observed_balanced) != expected_balanced
    ):
        raise ValueError("missing/duplicate balanced oracle inventory")
    balanced = {}
    for engine in ("candidate", "reference"):
        rows = [r["engines"][engine] for r in summary["balanced_checks"]]
        for row in rows:
            validate_balanced(row)
        evaluable = [r for r in rows if r["evaluable"]]
        criteria = [next(c for c in r["checks"] if c["metric"] == "criterion") for r in evaluable]
        balanced[engine] = {
            "attempted": len(rows),
            "evaluable": len(evaluable),
            "above_optimum": sum(c["difference"] > c["limit"] for c in criteria),
            "below_oracle": sum(c["difference"] < -c["limit"] for c in criteria),
            "variance_flagged_fits": sum(
                any(not c["passed"] for c in r["checks"] if c["metric"] != "criterion")
                for r in evaluable
            ),
            "rank_mismatches": sum(r["rank_match"] is False for r in evaluable),
            "maximum_absolute_criterion_gap": max(
                (abs(c["difference"]) for c in criteria), default=None
            ),
        }
    flags = [
        dict(case=r["case"], **p)
        for r in summary["paired"]
        for p in r["numeric"]
        if any(not c["passed"] for c in p["checks"])
    ]
    fit_totals = {e: Counter() for e in ("candidate", "reference")}
    for row in summary["fit_diagnostics"]:
        for engine in fit_totals:
            fit_totals[engine].update(row["engines"][engine])
    for engine in fit_totals:
        if fit_totals[engine]["converged"] != converged[engine]:
            raise ValueError("independent convergence audit failed")
    pilot.shared.write_json(
        output,
        {
            "schema_version": 1,
            "scope": "fresh exploratory pilot; no confirmatory calibration decision",
            "protocol": pilot.PROTOCOL,
            "namespace": manifest["namespace"],
            "engine_commit": pilot.ENGINE_COMMIT,
            "candidate": manifest["freeze"]["environment"],
            "control": pilot.CONTROL,
            "attempted": manifest["attempted"],
            "manifest_sha256": pilot.shared.digest(directory / "manifest.json"),
            "summary_sha256": pilot.shared.digest(directory / "summary.json"),
            "raw_count_audit": (
                "passed independently for candidate/reference rejection and coverage"
            ),
            "fit_totals": fit_totals,
            "balanced_checks": balanced,
            "paired_binary": dict(binary),
            "numeric_flagged_pairs": len(flags),
            "numeric_flags": flags,
            "summaries": summary["summaries"],
            "execution": {
                e: pilot.load(directory / f"{e}-execution.json") for e in ("candidate", "reference")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.directory, args.output)
