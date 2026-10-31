"""Independent fixed-point likelihood diagnostics for three retained failures.

Post-study diagnostic evidence only; never reads or writes outcome databases.
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

from diagnose_lmm_stopping import BalancedLikelihood, digest, write_new

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scientific/STOPPING_CONTROL_PILOT_V1_COUNTEREXAMPLES.json"
BITS = ROOT / "tests/data/stopping_control_counterexample_bits.json"


def analyze(raw_path, binary, output):
    fixtures = json.loads(FIXTURE.read_bytes())["counterexamples"]
    raw = json.loads(raw_path.read_bytes())
    if len(raw) != len(fixtures):
        raise ValueError("diagnostic case inventory differs")
    result = []
    for original, row in zip(fixtures, raw):
        if row["case"] != original["case"]["id"] or row["replicate"] != original["replicate"]:
            raise ValueError("diagnostic case/replicate differs")
        points = row["evaluations"]
        if {p["label"] for p in points} != {
            "retained",
            "zero_column",
            "repaired",
            *[
                f"retained_axis_{axis}_{sign}"
                for axis in range(3)
                for sign in (-1, 1)
                if original["candidate"]["theta"][axis]
                + sign * math.ulp(1.0) ** 0.25 * (1 + abs(original["candidate"]["theta"][axis]))
                >= 0
                or axis == 1
            ],
        }:
            raise ValueError("fixed-point inventory differs")
        by_label = {p["label"]: p for p in points}
        retained = original["candidate"]["theta"]
        if (
            len(points) != len(by_label)
            or by_label["retained"]["theta"] != retained
            or by_label["zero_column"]["theta"] != [0.0, 0.0, retained[2]]
            or by_label["repaired"]["theta"] != row["fit"]["theta"]
            or row["control"] != {"tolerance": 1e-10, "max_iterations": 5000}
        ):
            raise ValueError("fixed-point values/control differ")
        precisions = []
        for digits in (60, 80):
            with localcontext() as context:
                context.prec = digits
                likelihood = BalancedLikelihood(original["case"], original["y"])
                evaluated = []
                for point in points:
                    reference = likelihood.criterion(point["theta"])
                    errors = {
                        key: float(Decimal.from_float(point[key]) - reference)
                        for key in ("cached", "full")
                    }
                    limit = 64 * math.ulp(1.0) * max(1, abs(float(reference)))
                    evaluated.append(
                        {
                            **point,
                            "independent": str(reference),
                            "errors": errors,
                            "diagnostic_limit": limit,
                            "passed": all(abs(v) <= limit for v in errors.values()),
                        }
                    )
                costs = {p["label"]: Decimal(p["independent"]) for p in evaluated}
                precisions.append(
                    {
                        "digits": digits,
                        "points": evaluated,
                        "retained_minus_repaired": str(costs["retained"] - costs["repaired"]),
                        "origin_minus_retained": str(costs["zero_column"] - costs["retained"]),
                    }
                )
        result.append(
            {
                "case": row["case"],
                "replicate": row["replicate"],
                "fit": row["fit"],
                "independent": precisions,
            }
        )
    sources = [
        "scripts/diagnose_lmm_covariance_precision.py",
        "scripts/diagnose_lmm_stopping.py",
        "examples/diagnose_covariance_precision.rs",
        "src/optimizer.rs",
        "src/optimizer/basin_backend.rs",
        "Cargo.lock",
    ]
    write_new(
        output,
        {
            "scope": "post-study fixed-point diagnostic; no fresh validation or calibration claim",
            "fixture_sha256": digest(FIXTURE),
            "bits_sha256": digest(BITS),
            "raw_sha256": digest(raw_path),
            "binary_sha256": digest(binary),
            "sources": {name: digest(ROOT / name) for name in sources},
            "all_fixed_points_passed": all(
                p["passed"] for row in result for run in row["independent"] for p in run["points"]
            ),
            "cases": result,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-path", "binary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    analyze(**vars(parser.parse_args()))
