"""Retain fixed R optimizer variants on the nine flagged v2 references."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import lmm_random_slope_v2 as pilot

VARIANTS = [
    {"id": "baseline", "start": None, "control": {"maxfun": 100000}},
    {
        "id": "tight_default",
        "start": None,
        "control": {"maxfun": 100000, "rhobeg": 0.1, "rhoend": 1e-10},
    },
    {
        "id": "tight_positive",
        "start": [0.5, 0.25, 0.5],
        "control": {"maxfun": 100000, "rhobeg": 0.1, "rhoend": 1e-10},
    },
    {
        "id": "tight_negative",
        "start": [0.5, -0.25, 0.5],
        "control": {"maxfun": 100000, "rhobeg": 0.1, "rhoend": 1e-10},
    },
]
ROOT = Path(__file__).resolve().parents[1]
R_SOURCE = ROOT / "comparisons/scientific/lmm_random_slope_reference_controls.R"


def select_best(values):
    eligible = [v for v in values if v["eligible"]]
    order = {v["id"]: i for i, v in enumerate(VARIANTS)}
    return (
        min(eligible, key=lambda a: (a["fit"]["criterion"], order[a["variant"]]))
        if eligible
        else None
    )


def select(source):
    manifest = pilot.verify_run(source)
    summary = pilot.load(source / "summary.json")
    for name, expected in summary["artifacts"].items():
        if pilot.shared.digest(source / name) != expected:
            raise ValueError("original artifact changed")
    keys = [
        (row["case"], row["replicate"])
        for row in summary["balanced_checks"]
        if any(
            c["metric"] == "criterion" and c["difference"] > c["limit"]
            for c in row["engines"]["reference"]["checks"]
        )
    ]
    if len(keys) != 9 or len(set(keys)) != 9:
        raise ValueError("expected the nine retained above-oracle reference keys")
    designs = {c["id"]: c for c in pilot.load(source / "designs.json")}
    rows = []
    for name, index in keys:
        path = source / "cases" / name
        inputs = pilot.ordered(path / "inputs.jsonl.gz", name, manifest["replicates"])
        pairs = pilot.ordered(path / "paired.jsonl", name, manifest["replicates"], False)
        rows.append(
            {
                "case": designs[name],
                "replicate": index,
                "y": inputs[index]["y"],
                "retained_reference": pairs[index]["reference"],
            }
        )
    return rows


def run(source, destination, rscript):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("diagnostics must remain separate from the completed study")
    rows = select(source)
    destination.mkdir(parents=True)
    pilot.shared.write_json(destination / "inputs.json", rows)
    pilot.shared.write_json(destination / "variants.json", VARIANTS)
    # Written before any model fit: selection, fixed attempt bank and source identity.
    pilot.shared.write_json(
        destination / "receipt.json",
        {
            "scope": (
                "selected post-study reference diagnostics; "
                "no replacement pilot or calibration claim"
            ),
            "selection": "all nine R objectives above balanced oracle tolerance in immutable v2",
            "ranking": (
                "minimum criterion among converged finite fits with valid covariance; "
                "fixed variant order breaks exact ties; inference never used"
            ),
            "hashes": {
                p.relative_to(ROOT).as_posix(): pilot.shared.digest(p)
                for p in [Path(__file__), R_SOURCE, ROOT / "scientific/renv.lock"]
            },
            "inputs_sha256": pilot.shared.digest(destination / "inputs.json"),
            "variants_sha256": pilot.shared.digest(destination / "variants.json"),
            "source_summary_sha256": pilot.shared.digest(source / "summary.json"),
        },
    )
    environment = dict(os.environ, **pilot.shared.THREAD_LIMITS)
    with (destination / "r-output.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [str(rscript), str(R_SOURCE), str(destination)],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    attempts = [
        json.loads(line) for line in (destination / "attempts.jsonl").read_text().splitlines()
    ]
    expected = {(r["case"]["id"], r["replicate"], v["id"]) for r in rows for v in VARIANTS}
    actual = {(a["case"], a["replicate"], a["variant"]) for a in attempts}
    if len(attempts) != len(expected) or actual != expected:
        raise ValueError("missing/duplicate/unexpected diagnostic attempt")
    summaries = []
    for row in rows:
        values = [
            a
            for a in attempts
            if (a["case"], a["replicate"]) == (row["case"]["id"], row["replicate"])
        ]
        for value in values:
            value["numerical"] = pilot.balanced_checks(row["case"], row["y"], value["fit"])
            value["eligible"] = (
                value["numerical"]["evaluable"] and value["fit"].get("converged") is True
            )
            value["numerical_pass"] = (
                value["eligible"]
                and value["numerical"]["rank_match"]
                and all(c["passed"] for c in value["numerical"]["checks"])
            )
        chosen = select_best(values)
        summaries.append(
            {
                "case": row["case"]["id"],
                "replicate": row["replicate"],
                "selected": chosen["variant"] if chosen else None,
                "selected_numerical_pass": chosen["numerical_pass"] if chosen else False,
                "retained_reference": row["retained_reference"],
                "attempts": values,
            }
        )
    pilot.shared.write_json(
        destination / "analysis.json",
        {
            "scope": (
                "selected post-study optimizer diagnostics; "
                "cannot estimate future recovery probability"
            ),
            "receipt_sha256": pilot.shared.digest(destination / "receipt.json"),
            "attempts_sha256": pilot.shared.digest(destination / "attempts.jsonl"),
            "environment_sha256": pilot.shared.digest(destination / "reference-environment.json"),
            "summaries": summaries,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rscript", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.destination, args.rscript)
