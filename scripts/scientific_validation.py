"""Independent numerical contracts and selected paper comparisons.

The oracle uses Python integer/rational arithmetic and Decimal, never the
candidate's matrix routines or another fitted statistical library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path

PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286")


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def inverse_det(a):
    n = len(a)
    rows = [[Q(v) for v in row] + [Q(i == j) for j in range(n)] for i, row in enumerate(a)]
    determinant = Q(1)
    for col in range(n):
        pivot = next((i for i in range(col, n) if rows[i][col]), None)
        if pivot is None:
            raise ValueError("singular oracle matrix")
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            determinant *= -1
        value = rows[col][col]
        determinant *= value
        rows[col] = [v / value for v in rows[col]]
        for i in range(n):
            if i != col:
                value = rows[i][col]
                rows[i] = [v - value * p for v, p in zip(rows[i], rows[col])]
    return [row[n:] for row in rows], determinant


def decimal(q):
    return Decimal(q.numerator) / Decimal(q.denominator)


def gls_oracle(x, y, variance, reml):
    vi, det_v = inverse_det(variance)
    xtvi = multiply(transpose(x), vi)
    covariance, det_precision = inverse_det(multiply(xtvi, x))
    beta = multiply(covariance, multiply(xtvi, [[v] for v in y]))
    prediction = multiply(x, beta)
    residual = [[v - pred[0]] for v, pred in zip(y, prediction)]
    quadratic = multiply(transpose(residual), multiply(vi, residual))[0][0]
    with localcontext() as context:
        context.prec = 70
        df = len(y) - (len(x[0]) if reml else 0)
        criterion = Decimal(df) * (2 * PI).ln() + decimal(det_v).ln() + decimal(quadratic)
        if reml:
            criterion += decimal(det_precision).ln()
    return [v[0] for v in beta], covariance, float(criterion)


def balanced_case(reml):
    x = [[Q(1), value] for _ in range(4) for value in map(Q, ["-1.5", "-.5", ".5", "1.5"])]
    y = [
        Q(10 + shift) + 2 * row[1] + error
        for shift in [-3, -1, 1, 3]
        for row, error in zip(x[:4], map(Q, ["-.3", ".9", "-.9", ".3"]))
    ]
    sigma2 = Q(36, 55) if reml else Q(3, 5)
    tau2 = Q(20, 3) - sigma2 / 4 if reml else Q(5) - sigma2 / 4
    variance = [
        [(tau2 if i // 4 == j // 4 else Q(0)) + (sigma2 if i == j else Q(0)) for j in range(16)]
        for i in range(16)
    ]
    beta, covariance, criterion = gls_oracle(x, y, variance, reml)
    return x, y, sigma2, tau2, beta, covariance, criterion


def self_test():
    inverse, determinant = inverse_det([[2, 1], [1, 3]])
    assert determinant == 5
    assert inverse == [[Q(3, 5), Q(-1, 5)], [Q(-1, 5), Q(2, 5)]]
    try:
        inverse_det([[1, 1], [1, 1]])
    except ValueError:
        pass
    else:
        raise AssertionError("singular matrix accepted")
    for reml in [False, True]:
        _, _, sigma2, _, beta, covariance, _ = balanced_case(reml)
        assert beta == [10, 2]
        assert covariance[0][0] == (Q(5, 3) if reml else Q(5, 4))
        assert covariance[1][1] == sigma2 / 20
        assert covariance[0][1] == 0
    print("independent oracle: exact inverse, singular control and balanced identities passed")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CaseChecks:
    def __init__(self, contract):
        self.contract = contract
        self.values = []
        self.details = {}

    def close(self, name, actual, expected, rtol=None, atol=None):
        actual, expected = float(actual), float(expected)
        rtol = self.contract["rtol"] if rtol is None else rtol
        atol = self.contract["atol"] if atol is None else atol
        limit = atol + rtol * abs(expected)
        passed = math.isfinite(actual) and abs(actual - expected) <= limit
        self.values.append(
            {
                "metric": name,
                "actual": actual if math.isfinite(actual) else str(actual),
                "expected": expected,
                "limit": limit,
                "passed": passed,
            }
        )

    def vector(self, name, actual, expected, **tolerances):
        if len(actual) != len(expected):
            raise ValueError(f"{name} length differs: {len(actual)} != {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.close(f"{name}[{i}]", a, e, **tolerances)


def nist_rows(path):
    # Metadata also contains 'Data:'. Only the final heading starts observations.
    lines = path.read_text().rsplit("Data:", 1)[1].splitlines()[1:]
    return [[float(v) for v in line.split()] for line in lines if line.strip()]


def numerical_case(case_id, checks, lme, pl, run_dir):
    if case_id not in {
        "ols-rational-v1",
        "nist-longley-v1",
        "nist-wampler1-v1",
        "balanced-lmm-ml-v1",
        "balanced-lmm-reml-v1",
    }:
        raise ValueError(f"undeclared numerical case: {case_id}")
    if case_id == "ols-rational-v1":
        x = [[Q(1), Q(i)] for i in range(-2, 4)]
        y = [3 + 2 * row[1] + Q(error, 10) for row, error in zip(x, [1, -2, 1, 1, -2, 1])]
        precision_inv, _ = inverse_det(multiply(transpose(x), x))
        beta = multiply(precision_inv, multiply(transpose(x), [[v] for v in y]))
        fit = lme.lm([float(v) for v in y], [[float(v) for v in row] for row in x])
        sigma2 = Q(3, 100)
        checks.vector("coefficients", fit.coefficients, [v[0] for v in beta])
        checks.close("sigma2", fit.sigma2, sigma2)
        checks.vector(
            "std_errors",
            fit.std_errors,
            [math.sqrt(float(sigma2 * precision_inv[i][i])) for i in range(2)],
        )
        variance_sum = sigma2 * sum(sum(row) for row in precision_inv)
        contrast = fit.test_contrast([[1, 1]], "residual")
        checks.close(
            "joint_coefficient_variance",
            sum(fit.coefficients) ** 2 / contrast.f_value,
            variance_sum,
        )
        return
    if case_id.startswith("nist-"):
        name = "Longley" if "longley" in case_id else "Wampler1"
        rows = nist_rows(run_dir / f"{name}.dat")
        x = [[1.0] + row[1:] for row in rows]
        if name == "Wampler1":
            x = [[row[1] ** power for power in range(6)] for row in rows]
        fit = lme.lm([row[0] for row in rows], x)
        if name == "Longley":
            checks.vector(
                "coefficients",
                fit.coefficients,
                [
                    -3482258.63459582,
                    15.0618722713733,
                    -0.0358191792925910,
                    -2.02022980381683,
                    -1.03322686717359,
                    -0.0511041056535807,
                    1829.15146461355,
                ],
            )
            checks.vector(
                "std_errors",
                fit.std_errors,
                [
                    890420.383607373,
                    84.9149257747669,
                    0.0334910077722432,
                    0.488399681651699,
                    0.214274163161675,
                    0.226073200069370,
                    455.478499142212,
                ],
            )
            checks.close("RSS", fit.sigma2 * 9, 836424.055505915)
        else:
            checks.vector("coefficients", fit.coefficients, [1] * 6, rtol=0, atol=1e-8)
            checks.close("sigma2", fit.sigma2, 0, rtol=0, atol=1e-14)
        return
    reml = "-reml-" in case_id
    x, y, sigma2, tau2, beta, covariance, criterion = balanced_case(reml)
    data = pl.DataFrame(
        {
            "y": list(map(float, y)),
            "x": [float(row[1]) for row in x],
            "g": [f"g{i // 4}" for i in range(16)],
        }
    )
    fit = lme.lmer("y ~ x + (1 | g)", data, reml=reml)
    if not fit.converged:
        raise ValueError("balanced fit did not converge")
    checks.vector("coefficients", fit.coefficients, beta)
    checks.close("sigma2", fit.sigma2, sigma2)
    checks.close("tau2", fit.sigma2 * fit.theta[0] ** 2, tau2)
    checks.vector(
        "std_errors", fit.std_errors, [math.sqrt(float(covariance[i][i])) for i in range(2)]
    )
    checks.close("criterion", fit.reml_criterion if reml else fit.deviance, criterion)
    fit.with_satterthwaite(data)
    contrast = fit.test_contrast([[1, 1]], "satterthwaite")
    checks.close(
        "joint_coefficient_variance",
        sum(fit.coefficients) ** 2 / contrast.f_value,
        sum(sum(row) for row in covariance),
    )
    if reml:
        for label, row, expected_df in [("intercept", [1, 0], 3), ("slope", [0, 1], 11)]:
            inference = fit.test_contrast([row], "satterthwaite")
            checks.close(
                f"satterthwaite.{label}.den_df",
                inference.den_df,
                expected_df,
                rtol=0,
                atol=checks.contract["satterthwaite_df_atol"],
            )


def canonical_name(name):
    if name.startswith("TVset:Picture"):
        tv, picture = name.removeprefix("TVset:Picture").split(":")
        return f"TVset{tv}:Picture{picture}"
    return re.sub(r"[\[\]]", "", name).replace("T.", "")


def paper_case(case_id, checks, lme, pl, run_dir, references):
    name = case_id.split("-")[0]
    cases = {
        "bates": ("sleepstudy.csv", "Reaction ~ Days + (Days | Subject)", ["Subject"]),
        "kuznetsova": (
            "TVbo.csv",
            "Sharpnessofmovement ~ TVset * Picture + (1 | Assessor) "
            "+ (1 | Assessor:TVset) + (1 | Assessor:Picture)",
            ["TVset", "Picture", "Assessor"],
        ),
        "halekoh": (
            "beets.csv",
            "sugpct ~ block + sow + harvest + (1 | block:harvest)",
            ["block", "sow", "harvest"],
        ),
    }
    filename, formula, factors = cases[name]
    data = pl.read_csv(run_dir / filename).with_columns(pl.col(factors).cast(pl.String))
    fit = lme.lmer(formula, data, reml=True, control=lme.FitControl(**checks.contract["control"]))
    checks.details.update(
        formula=formula,
        num_obs=data.height,
        reml=True,
        control=checks.contract["control"],
        fixed_names=fit.fixed_names,
        converged=fit.converged,
        iterations=fit.iterations,
    )
    if not fit.converged:
        raise ValueError("paper example fit did not converge")
    ref = references[name] if name == "bates" else references[name]["fit"]
    if ref["optimizer_messages"]:
        raise ValueError("R reference model reported convergence messages")
    order = [canonical_name(label) for label in fit.fixed_names]
    checks.vector("coefficients", fit.coefficients, [ref["coefficients"][label] for label in order])
    checks.vector("std_errors", fit.std_errors, [ref["std_errors"][label] for label in order])
    checks.close("sigma2", fit.sigma2, ref["sigma2"])
    # R orders RE blocks by decreasing group count; this API retains formula
    # order. Match structural labels, not the raw positions in the two arrays.
    expected_theta = ref["theta"] if isinstance(ref["theta"], list) else [ref["theta"]]
    if name == "kuznetsova":
        by_group = dict(zip(ref["theta_names"], expected_theta))
        expected_theta = [
            by_group[f"{group}.(Intercept)"]
            for group in ["Assessor", "Assessor:TVset", "Assessor:Picture"]
        ]
    checks.vector("theta", fit.theta, expected_theta)
    checks.close("criterion", fit.reml_criterion, ref["criterion"], rtol=1e-8, atol=1e-5)
    checks.vector("conditional_fitted", fit.fitted, ref["fitted"])
    if name == "bates":
        return
    fit.with_satterthwaite(data)
    fit.with_kenward_roger(data)
    if name == "kuznetsova":
        for method, kind, key in [
            ("satterthwaite", "III", "satterthwaite_type3"),
            ("kenward_roger", "II", "kenward_roger_type2"),
        ]:
            table = fit.anova(ddf_method=method, anova_type=kind)
            expected = {record["term"]: record for record in references[name][key]}
            if set(table.terms) != set(expected):
                raise ValueError("ANOVA hypothesis terms differ")
            for i, term in enumerate(table.terms):
                for metric in ["num_df", "den_df", "f_value", "p_value"]:
                    tolerance = {
                        "num_df": {"rtol": 0, "atol": 0},
                        "p_value": {"rtol": 0, "atol": 1e-5},
                    }.get(metric, {})
                    checks.close(
                        f"{key}.{term}.{metric}",
                        getattr(table, metric)[i],
                        expected[term][metric],
                        **tolerance,
                    )
    else:
        result = fit.linear_hypothesis("harvest", "kenward_roger")
        for metric in ["num_df", "den_df", "f_value", "p_value"]:
            tolerance = {
                "num_df": {"rtol": 0, "atol": 0},
                "p_value": {"rtol": 0, "atol": 1e-5},
            }.get(metric, {})
            checks.close(
                f"harvest_kr.{metric}",
                getattr(result, metric),
                references[name]["harvest_kr"][metric],
                **tolerance,
            )


def verify(registry_path, run_dir, output, expected_version):
    import lme_python as lme
    import polars as pl
    from lme_python import lme_python as native

    registry = json.loads(registry_path.read_text())
    case_ids = [case["id"] for case in registry["cases"]]
    if registry["schema_version"] != 1 or not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("registry needs schema 1 and nonempty, unique case IDs")
    actual_version = importlib.metadata.version("lme-python")
    if actual_version != expected_version or expected_version != registry["baseline_version"]:
        raise ValueError(
            f"version mismatch: candidate={actual_version}, requested={expected_version}"
        )
    Path(native.__file__).resolve().relative_to(Path(sys.prefix).resolve())
    source_hashes = {}
    for name, source in registry["sources"].items():
        digest = sha256(run_dir / name)
        if digest != source["sha256"].lower():
            raise ValueError(f"source hash mismatch: {name}")
        source_hashes[name] = digest
    references = json.loads((run_dir / "paper-references.json").read_text())
    if references["packages"] != registry["reference_packages"]:
        raise ValueError("R reference package versions differ from the declared environment")
    if references["namespace_versions"] != registry["reference_namespaces"]:
        raise ValueError("R loaded dependency versions differ from the declared environment")
    results = []
    for contract in registry["cases"]:
        checks = CaseChecks(contract)
        result = {"id": contract["id"], "contract": contract}
        try:
            if "doi" in contract:
                paper_case(contract["id"], checks, lme, pl, run_dir, references)
            else:
                numerical_case(contract["id"], checks, lme, pl, run_dir)
            result["status"] = (
                "passed" if checks.values and all(v["passed"] for v in checks.values) else "failed"
            )
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["checks"] = checks.values
        result["details"] = checks.details
        results.append(result)
        print(f"{result['status']}: {result['id']} ({len(checks.values)} comparisons)")
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine_version": actual_version,
        "native_sha256": sha256(Path(native.__file__)),
        "harness_sha256": sha256(Path(__file__)),
        "registry_sha256": sha256(registry_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "source_sha256": source_hashes,
        "reference_environment": {
            key: references[key]
            for key in ["r_version", "packages", "namespace_versions", "contrasts", "scope"]
        },
        "python_dependencies": dict(
            sorted(
                (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
            )
        ),
        "r_session_sha256": sha256(run_dir / "R-session.txt"),
        "data_sha256": {
            name: sha256(run_dir / name) for name in ["sleepstudy.csv", "TVbo.csv", "beets.csv"]
        },
        "reference_sha256": sha256(run_dir / "paper-references.json"),
        "passed": all(case["status"] == "passed" for case in results),
        "cases": results,
        "limits": [
            "selected deterministic examples only",
            "no calibration study performed",
            "external scientific review pending",
            "no GLMM/NLMM certification",
        ],
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine-version", default="0.2.7")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return 0
    if not all([args.registry, args.run_dir, args.output]):
        parser.error("--registry, --run-dir and --output are required")
    return verify(args.registry, args.run_dir, args.output, args.engine_version)


if __name__ == "__main__":
    raise SystemExit(main())
