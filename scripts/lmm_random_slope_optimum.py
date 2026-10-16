"""Post-pilot balanced constrained-REML oracle; reads records, never refits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import lmm_random_slope as pilot


def optimum(group_x, observations):
    """Common centered full-rank Z=[1,x], G>1,m>2 and positive within RSS."""
    g, m = len(observations), len(group_x)
    x = group_x
    sxx = math.fsum(t * t for t in x)
    if (
        g <= 1
        or m <= 2
        or sxx <= 0
        or abs(sum(x)) > 1e-10
        or any(len(row) != m for row in observations)
    ):
        raise ValueError("oracle requires balanced centered G>1,m>2 design")
    coefficients = [
        [math.fsum(row) / m, math.fsum(a * b for a, b in zip(x, row)) / sxx] for row in observations
    ]
    beta = [math.fsum(row[j] for row in coefficients) / g for j in range(2)]
    ssw = math.fsum(
        (y - c[0] - c[1] * v) ** 2
        for row, c in zip(observations, coefficients)
        for v, y in zip(x, row)
    )
    if ssw <= 0:
        raise ValueError("positive within-group RSS required")
    q, w = g - 1, g * (m - 2)
    a = m * math.fsum((c[0] - beta[0]) ** 2 for c in coefficients) / q
    b = (
        math.sqrt(m * sxx)
        * math.fsum((c[0] - beta[0]) * (c[1] - beta[1]) for c in coefficients)
        / q
    )
    d = sxx * math.fsum((c[1] - beta[1]) ** 2 for c in coefficients) / q
    radius = math.hypot((a - d) / 2, b)
    low, high = (a + d) / 2 - radius, (a + d) / 2 + radius
    sigma2 = ssw / w
    binding = 0
    if sigma2 > low:
        binding = 1
        sigma2 = (ssw + q * low) / (w + q)
        if sigma2 > high:
            binding = 2
            sigma2 = (ssw + q * (low + high)) / (w + 2 * q)
    if binding == 0:
        r00, r01, r11 = a - sigma2, b, d - sigma2
    elif binding == 2:
        r00 = r01 = r11 = 0.0
    else:
        scale = (high - sigma2) / (high - low)
        r00, r01, r11 = scale * (a - low), scale * b, scale * (d - low)
    covariance = [[r00 / m, r01 / math.sqrt(m * sxx)], [r01 / math.sqrt(m * sxx), r11 / sxx]]
    eigenvalues = [max(sigma2, low), max(sigma2, high)]
    criterion = (
        (g * m - 2) * math.log(2 * math.pi)
        + math.log(g * g * m * sxx)
        + w * math.log(sigma2)
        + ssw / sigma2
        + q * sum(math.log(v) + lam / v for v, lam in zip(eigenvalues, [low, high]))
    )
    fixed_covariance = [
        [(covariance[0][0] + sigma2 / m) / g, covariance[0][1] / g],
        [covariance[1][0] / g, (covariance[1][1] + sigma2 / sxx) / g],
    ]
    return {
        "beta": beta[1],
        "coefficients": beta,
        "sigma2": sigma2,
        "covariance": covariance,
        "fixed_covariance": fixed_covariance,
        "criterion": criterion,
        "binding_directions": binding,
        "ssw": ssw,
        "between_eigenvalues": [low, high],
    }


def diagnose(directory, output):
    manifest = pilot.verify_run(directory)
    summary = pilot.load(directory / "summary.json")
    for name, expected in summary["artifacts"].items():
        if pilot.shared.digest(directory / name) != expected:
            raise ValueError("original artifact differs from the pilot summary")
    comparisons = []
    for case in pilot.load(directory / "designs.json"):
        if len(set(case["sizes"])) != 1:
            continue
        m = case["sizes"][0]
        x = case["x"][:m]
        if any(case["x"][i : i + m] != x for i in range(0, len(case["x"]), m)):
            raise ValueError("oracle requires common group predictor")
        folder = directory / "cases" / case["id"]
        inputs = pilot.ordered(folder / "inputs.jsonl.gz", case["id"], manifest["replicates"])
        candidates = pilot.ordered(
            folder / "candidate.jsonl.gz", case["id"], manifest["replicates"]
        )
        references = pilot.ordered(
            folder / "paired.jsonl", case["id"], manifest["replicates"], False
        )
        for row, candidate, pair in zip(inputs, candidates, references):
            y = row["y"]
            expected = optimum(x, [y[i : i + m] for i in range(0, len(y), m)])
            engines = {}
            for label, actual in (
                ("candidate", candidate["candidate"]),
                ("reference", pair["reference"]),
            ):
                difference = actual["criterion"] - expected["criterion"]
                atol, rtol = pilot.GATES["criterion"]
                limit = atol + rtol * abs(expected["criterion"])
                engines[label] = {
                    "criterion_gap": difference,
                    "limit": limit,
                    "above_optimum": difference > limit,
                    "below_oracle": difference < -limit,
                    "sigma2_difference": actual["sigma2"] - expected["sigma2"],
                }
            comparisons.append(
                {
                    "case": case["id"],
                    "replicate": row["replicate"],
                    "oracle": expected,
                    "engines": engines,
                }
            )
    pilot.shared.write_json(
        output,
        {
            "scope": "post-pilot balanced global REML diagnostic; no refits or substitutions",
            "source_summary_sha256": pilot.shared.digest(directory / "summary.json"),
            "diagnostic_source_sha256": pilot.shared.digest(Path(__file__)),
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.directory, args.output)
