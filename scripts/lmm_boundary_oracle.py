"""Exact balanced between-group REML oracle, independent of fitting libraries.

The optional candidate adapter checks published 0.2.7 on five deterministic
fixtures. These are diagnostic fixtures, not additional confirmatory draws.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path

PI = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164")


def criterion(result, sigma2, v):
    """Full-design -2 REML log likelihood, including log|X'X|."""
    sigma2, v = Q(sigma2), Q(v)
    if sigma2 <= 0 or v < sigma2:
        raise ValueError("inadmissible covariance")
    with localcontext() as ctx:
        ctx.prec = 70

        def dec(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        return float(
            (result["n"] - 2) * (2 * PI).ln()
            + dec(result["xtx_det"]).ln()
            + result["w"] * dec(sigma2).ln()
            + dec(result["ssw"]) / dec(sigma2)
            + result["q"] * dec(v).ln()
            + dec(result["ssb"]) / dec(v)
        )


def oracle(group_x, observations):
    x = list(map(Q, group_x))
    y = [list(map(Q, row)) for row in observations]
    g = len(x)
    if g <= 2 or len(y) != g or len(y[0]) < 2 or any(len(row) != len(y[0]) for row in y):
        raise ValueError("requires balanced G>2, m>1 design")
    m = len(y[0])
    n = g * m
    w = g * (m - 1)
    q = g - 2
    xbar = sum(x) / g
    sxx = sum((value - xbar) ** 2 for value in x)
    if sxx == 0:
        raise ValueError("constant predictor")
    means = [sum(row) / m for row in y]
    ybar = sum(means) / g
    beta1 = sum((a - xbar) * (b - ybar) for a, b in zip(x, means)) / sxx
    beta0 = ybar - beta1 * xbar
    ssw = sum((value - mean) ** 2 for row, mean in zip(y, means) for value in row)
    ssb = m * sum((mean - beta0 - beta1 * value) ** 2 for mean, value in zip(means, x))
    if ssw <= 0:
        raise ValueError("degenerate within-group variation")
    msw, msb = ssw / w, ssb / q
    boundary = msb <= msw
    sigma2 = (ssw + ssb) / (w + q) if boundary else msw
    v = sigma2 if boundary else msb
    result = {
        "n": n,
        "g": g,
        "m": m,
        "w": w,
        "q": q,
        "sxx": sxx,
        "xtx_det": m * m * g * sxx,
        "beta0": beta0,
        "beta1": beta1,
        "ssw": ssw,
        "ssb": ssb,
        "msw": msw,
        "msb": msb,
        "sigma2": sigma2,
        "v": v,
        "tau2": (v - sigma2) / m,
        "boundary": boundary,
        "slope_variance": v / (m * sxx),
        "group_means_variance": msb / (m * sxx),
    }
    result["criterion"] = criterion(result, sigma2, v)
    return result


def fixtures():
    result = []
    for name, x, residual in [
        ("strict_boundary", list(map(Q, [-3, -1, 1, 3])), [Q(1, 2), Q(-1, 2), Q(-1, 2), Q(1, 2)]),
        ("strict_interior", list(map(Q, [-3, -1, 1, 3])), list(map(Q, [1, -1, -1, 1]))),
        *[
            (name, list(map(Q, [-1, -1, 1, 1])), [scale, -scale, Q(0), Q(0)])
            for name, scale in [
                ("exact_boundary", Q(1)),
                ("near_boundary_below", Q(999, 1000)),
                ("near_boundary_above", Q(1001, 1000)),
            ]
        ],
    ]:
        means = [2 + Q(4, 5) * value + error for value, error in zip(x, residual)]
        y = [[mean - 1, mean + 1] for mean in means]
        expected = oracle(x, y)
        result.append(
            {
                "id": name,
                "group_x": list(map(float, x)),
                "observations": [list(map(float, row)) for row in y],
                "expected": {k: float(v) if isinstance(v, Q) else v for k, v in expected.items()},
                "exact": {k: str(v) for k, v in expected.items() if isinstance(v, Q)},
            }
        )
    return result


def compare(record, expected):
    checks = []
    for name in ["beta0", "beta1", "sigma2", "tau2", "slope_variance", "criterion"]:
        actual = record.get(name)
        # Declared before fixture execution; precision for optimized fits, not MC bands.
        limit = 1e-7 if name == "criterion" else 1e-7 + 1e-5 * abs(expected[name])
        checks.append(
            {
                "metric": name,
                "actual": actual,
                "expected": expected[name],
                "limit": limit,
                "passed": isinstance(actual, (float, int))
                and math.isfinite(actual)
                and abs(actual - expected[name]) <= limit,
            }
        )
    for name in ["kr_variance", "kr_df"]:
        target = expected["slope_variance"] if name == "kr_variance" else expected["q"]
        actual = record.get(name)
        limit = 1e-7 + 1e-5 * abs(target)
        checks.append(
            {
                "metric": name,
                "actual": actual,
                "expected": target,
                "limit": limit,
                "passed": isinstance(actual, (float, int))
                and math.isfinite(actual)
                and abs(actual - target) <= limit,
            }
        )
    return checks


def candidate_checks(cases):
    import lme_python as lme
    import polars as pl
    from lme_python import lme_python as native

    if importlib.metadata.version("lme-python") != "0.2.7" or not Path(
        native.__file__
    ).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("requires isolated published 0.2.7")
    results = []
    for case in cases:
        y = [v for row in case["observations"] for v in row]
        x = [v for v in case["group_x"] for _ in range(2)]
        data = pl.DataFrame(
            {"y": y, "x": x, "group": [f"g{i}" for i in range(4) for _ in range(2)]}
        )
        fit = lme.lmer(
            "y ~ x + (1 | group)",
            data,
            reml=True,
            control=lme.FitControl(tolerance=1e-10, max_iterations=5000),
        )
        record = {
            "case": case["id"],
            "converged": fit.converged,
            "beta0": fit.coefficients[0],
            "beta1": fit.coefficients[1],
            "sigma2": fit.sigma2,
            "tau2": fit.sigma2 * fit.theta[0] ** 2,
            "slope_variance": fit.std_errors[1] ** 2,
            "criterion": fit.reml_criterion,
        }
        fit.with_satterthwaite(data)
        sat = fit.test_contrast([[0, 1]], "satterthwaite")
        record.update(satterthwaite_df=sat.den_df, satterthwaite_p=sat.p_value)
        fit.with_kenward_roger(data)
        kr = fit.test_contrast([[0, 1]], "kenward_roger")
        unit = fit.test_contrast_vs(
            [[0, 1]], [fit.coefficients[0], fit.coefficients[1] - 1], "kenward_roger"
        )
        if unit.num_df != 1 or not math.isclose(unit.den_df, kr.den_df, rel_tol=1e-10):
            raise ValueError("unit-contrast rank/df mismatch")
        record.update(kr_variance=1 / unit.f_value, kr_df=kr.den_df, kr_p=kr.p_value)
        record["checks"] = compare(record, case["expected"])
        results.append(record)
    return {
        "candidate_version": "0.2.7",
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "control": {"tolerance": 1e-10, "max_iterations": 5000},
        "results": results,
        "passed": all(r["converged"] and all(c["passed"] for c in r["checks"]) for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    cases = fixtures()
    report = candidate_checks(cases) if args.candidate else {"fixtures": cases}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write("\n")
    if args.candidate and not report["passed"]:
        raise SystemExit("candidate fixture checks failed; evidence retained")


if __name__ == "__main__":
    main()
