"""Exploratory LMM pilot: immutable inputs, paired implementations and failure bounds.

Only the pilot namespace is executable. Confirmation needs a separately frozen
revision; this script never promotes a pilot result to a validation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROTOCOL = "lmm-calibration-v2"
FAMILY = 48
BOOTSTRAPS = 1000
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def emit(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def scaled(n):
    values = [2 * i / (n - 1) - 1 for i in range(n)]
    mean = statistics.mean(values)
    scale = math.sqrt(statistics.mean((x - mean) ** 2 for x in values))
    return [(x - mean) / scale for x in values]


def cases():
    result = []
    factors = itertools.product(
        [6, 12, 30], [3, 10], [0, 0.2, 0.7], ["within", "between"], [0, 0.5]
    )
    for g, m, rho, location, beta in factors:
        result.append(make_case(g, [m] * g, rho, location, beta, "balanced"))
    for g, rho, location in itertools.product([6, 12], [0.2, 0.7], ["within", "between"]):
        result.append(make_case(g, [2, 4, 8] * (g // 3), rho, location, 0, "imbalanced"))
    return result


def make_case(g, sizes, rho, location, beta, stratum):
    group = [i for i, n in enumerate(sizes) for _ in range(n)]
    between = scaled(g)
    x = [
        value
        for i, n in enumerate(sizes)
        for value in (scaled(n) if location == "within" else [between[i]] * n)
    ]
    size_id = str(sizes[0]) if stratum == "balanced" else "2-4-8"
    return {
        "id": f"{stratum}-G{g}-m{size_id}-rho{rho:g}-{location}-beta{beta:g}",
        "stratum": stratum,
        "groups": g,
        "sizes": sizes,
        "rho": rho,
        "location": location,
        "beta": beta,
        "primary": stratum == "balanced" and rho > 0 and beta == 0,
        "x": x,
        "group": group,
    }


def normals(key, count):
    """SHA256 counter uniforms (52 bits, open interval) then Box-Muller pairs."""
    result = []
    for counter in range((count + 1) // 2):
        block = hashlib.sha256(f"{key}|{counter}".encode("ascii")).digest()
        u = [(int.from_bytes(block[i : i + 8], "big") >> 12) + 0.5 for i in [0, 8]]
        u = [v / 2**52 for v in u]
        radius = math.sqrt(-2 * math.log(u[0]))
        angle = 2 * math.pi * u[1]
        result.extend([radius * math.cos(angle), radius * math.sin(angle)])
    return result[:count]


def response(case, replicate):
    key = f"{PROTOCOL}|pilot|{case['id']}|{replicate}"
    draws = normals(key, case["groups"] + len(case["x"]))
    b = [math.sqrt(case["rho"]) * v for v in draws[: case["groups"]]]
    e = draws[case["groups"] :]
    return [
        2 + case["beta"] * x + b[g] + math.sqrt(1 - case["rho"]) * error
        for x, g, error in zip(case["x"], case["group"], e)
    ]


def generate(destination, replicates):
    if not 1 <= replicates <= 100:
        raise ValueError("pilot replicates must be between 1 and 100")
    destination.mkdir(parents=True, exist_ok=False)
    registry = cases()
    write_json(destination / "designs.json", registry)
    with (destination / "inputs.jsonl").open("x", encoding="utf-8") as stream:
        for case in registry:
            for rep in range(replicates):
                emit(stream, {"case": case["id"], "replicate": rep, "y": response(case, rep)})
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "namespace": "pilot",
            "replicates": replicates,
            "attempted_datasets": len(registry) * replicates,
            "bootstrap_repetitions": BOOTSTRAPS,
            "primary_family": FAMILY,
            "candidate_version": "0.2.7",
            "candidate_release_git_sha": subprocess.check_output(
                ["git", "rev-list", "-n", "1", "v0.2.7"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "inputs_sha256": digest(destination / "inputs.jsonl"),
            "designs_sha256": digest(destination / "designs.json"),
            "sources": {
                name: digest(ROOT / name)
                for name in [
                    "scripts/lmm_calibration.py",
                    "scripts/ci/lme_ci.py",
                    "comparisons/scientific/lmm_calibration.R",
                    "docs/LMM_CALIBRATION_PROTOCOL.md",
                    "docs/LMM_CALIBRATION_AI_REVIEW.md",
                    "scientific/pyproject.toml",
                    "scientific/uv.lock",
                    "scientific/cases.json",
                ]
            },
            "candidate_control": {
                "reml": True,
                "tolerance": 1e-6,
                "max_iterations": 1000,
                "require_convergence": False,
            },
            "reference_control": {"optimizer": "bobyqa", "maxfun": 100000},
        },
    )


def verify_inputs(destination):
    manifest = json.loads((destination / "manifest.json").read_text())
    if (
        manifest["namespace"] != "pilot"
        or manifest["protocol"] != PROTOCOL
        or not 1 <= manifest["replicates"] <= 100
    ):
        raise ValueError("unsupported protocol/namespace/replicate budget")
    for name in ["inputs", "designs"]:
        suffix = "jsonl" if name == "inputs" else "json"
        if digest(destination / f"{name}.{suffix}") != manifest[f"{name}_sha256"]:
            raise ValueError(f"{name} hash changed")
    for name, expected in manifest["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return manifest


def known_variance(case, y):
    """GLS with true block covariance: Sherman-Morrison and a 2x2 inverse."""
    sigma2, tau2 = 1 - case["rho"], case["rho"]
    s00 = s01 = s11 = z0 = z1 = 0.0
    start = 0
    for n in case["sizes"]:
        x, v = case["x"][start : start + n], y[start : start + n]
        c = tau2 / (sigma2 * (sigma2 + n * tau2))
        sx, sy = sum(x), sum(v)
        s00 += n / sigma2 - c * n * n
        s01 += sx / sigma2 - c * n * sx
        s11 += sum(t * t for t in x) / sigma2 - c * sx * sx
        z0 += sy / sigma2 - c * n * sy
        z1 += sum(a * b for a, b in zip(x, v)) / sigma2 - c * sx * sy
        start += n
    det = s00 * s11 - s01 * s01
    return (s00 * z1 - s01 * z0) / det, math.sqrt(s00 / det)


def candidate(destination):
    manifest = verify_inputs(destination)
    import lme_python as lme
    import polars as pl
    from lme_python import lme_python as native

    if importlib.metadata.version("lme-python") != manifest["candidate_version"]:
        raise ValueError("candidate version differs from frozen release")
    if not Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("candidate must be installed in the isolated environment")
    registry = {c["id"]: c for c in json.loads((destination / "designs.json").read_text())}
    started = time.perf_counter()
    with (destination / "candidate.jsonl").open("x", encoding="utf-8") as stream:
        for index, input_ in enumerate(lines(destination / "inputs.jsonl")):
            case = registry[input_["case"]]
            record = {"case": case["id"], "replicate": input_["replicate"], "methods": {}}
            before = time.perf_counter()
            data = pl.DataFrame(
                {"y": input_["y"], "x": case["x"], "group": [f"g{g:03d}" for g in case["group"]]}
            )
            try:
                fit = lme.lmer(
                    "y ~ x + (1 | group)",
                    data,
                    reml=True,
                    control=lme.FitControl(
                        tolerance=1e-6, max_iterations=1000, require_convergence=False
                    ),
                )
                beta = fit.coefficients[1]
                record.update(
                    beta=beta,
                    sigma2=fit.sigma2,
                    tau2=fit.sigma2 * fit.theta[0] ** 2,
                    converged=fit.converged is True,
                    singular=abs(fit.theta[0]) < 1e-4,
                    optimizer_diagnostics=fit.diagnostics,
                )
                if not record["converged"]:
                    raise ValueError("nonconvergence")
                se = fit.std_errors[1]
                record["methods"]["normal"] = {
                    "beta": beta,
                    "se": se,
                    "df": None,
                    "f": (beta / se) ** 2,
                    "p": math.erfc(abs(beta / se) / math.sqrt(2)),
                }
                for name in ["satterthwaite", "kenward_roger"]:
                    try:
                        getattr(fit, f"with_{name}")(data)
                        test = fit.test_contrast([[0.0, 1.0]], ddf_method=name)
                        # The public Python API does not expose adjusted covariance.
                        # A unit contrast displacement avoids beta^2/F near zero.
                        unit = fit.test_contrast_vs(
                            [[0.0, 1.0]], [fit.coefficients[0], beta - 1], ddf_method=name
                        )
                        variance = 1 / unit.f_value
                        expected_f = beta * beta / variance
                        if test.num_df != 1 or not math.isclose(
                            test.f_value, expected_f, rel_tol=1e-8, abs_tol=1e-10
                        ):
                            raise ValueError("scalar test/variance inconsistency")
                        if not math.isfinite(test.den_df) or test.den_df <= 0:
                            raise ValueError("invalid df")
                        record["methods"][name] = {
                            "beta": beta,
                            "se": math.sqrt(variance),
                            "df": test.den_df,
                            "f": test.f_value,
                            "p": test.p_value,
                        }
                    except BaseException as error:
                        if isinstance(error, (KeyboardInterrupt, SystemExit)):
                            raise
                        record["methods"][name] = {"error": f"{type(error).__name__}: {error}"}
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                record["fit_error"] = f"{type(error).__name__}: {error}"
            record["seconds"] = time.perf_counter() - before
            # Nonfinite values remain recorded as strings and fail analysis validation.
            emit(stream, finite_json(record))
            if (index + 1) % 100 == 0:
                print(f"candidate: {index + 1}/{manifest['attempted_datasets']}", flush=True)
    write_json(
        destination / "candidate-environment.json",
        {
            "engine": importlib.metadata.version("lme-python"),
            "polars": pl.__version__,
            "native_sha256": digest(Path(native.__file__)),
            "seconds": time.perf_counter() - started,
        },
    )


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def wilson(k, n):
    if n == 0:
        return None
    z = statistics.NormalDist().inv_cdf(0.975)
    p, a = k / n, z * z / n
    center = (p + a / 2) / (1 + a)
    half = z * math.sqrt(p * (1 - p) / n + a / (4 * n)) / (1 + a)
    return [max(0, center - half), min(1, center + half)]


def binary_measure(values, attempted, bounds):
    n, k = len(values), sum(values)
    failed = attempted - n
    return {
        "evaluable": n,
        "count": k,
        "rate": k / n if n else None,
        "mcse": math.sqrt(k / n * (1 - k / n) / n) if n else None,
        "wilson95": wilson(k, n),
        "unconditional_identification_bounds": [k / attempted, (k + failed) / attempted],
        "failure_envelope95": [bounds["pointwise"][k][0], bounds["pointwise"][k + failed][1]],
        "failure_envelope_simultaneous95": [
            bounds["simultaneous"][k][0],
            bounds["simultaneous"][k + failed][1],
        ],
    }


def fit_status(fit):
    if fit.get("converged") is True:
        return "converged"
    error = fit.get("fit_error")
    if error and error not in {"nonconvergence", "ValueError: nonconvergence"}:
        return "exception"
    if fit.get("converged") is False or error:
        return "nonconverged"
    return "unknown_convergence"


def diagnostic_counts(fits, bounds):
    statuses = {
        name: 0 for name in ["converged", "nonconverged", "exception", "unknown_convergence"]
    }
    for fit in fits:
        statuses[fit_status(fit)] += 1
    flags = [fit["singular"] for fit in fits if isinstance(fit.get("singular"), bool)]
    return {
        "statuses": statuses,
        "singular_classifiable": len(flags),
        "singular_unknown": len(fits) - len(flags),
        "singular_rate": binary_measure(flags, len(fits), bounds),
    }


def outcome(method, truth):
    if "error" in method:
        raise ValueError(method["error"])
    beta, se, p, critical = [method[x] for x in ["beta", "se", "p", "critical"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [beta, se, p, critical]):
        raise ValueError("nonfinite output")
    if se <= 0 or not 0 <= p <= 1 or critical <= 0:
        raise ValueError("invalid probability/SE/critical value")
    lower, upper = beta - critical * se, beta + critical * se
    reject = p < 0.05
    if reject != (abs(beta) > critical * se):
        raise ValueError("test/interval disagreement")
    return {
        "beta": beta,
        "se": se,
        "error": beta - truth,
        "reject": reject,
        "cover": lower <= truth <= upper,
        "width": upper - lower,
        "below": truth < lower,
        "above": truth > upper,
        "df": method.get("df"),
    }


def performance(rows):
    if len(rows) < 2:
        return {
            name: None
            for name in ["bias", "rmse", "se_calibration", "empirical_sd", "mean_se", "mean_width"]
        }
    error = [v["error"] for v in rows]
    sd = statistics.stdev(error)
    return {
        "bias": statistics.mean(error),
        "rmse": math.sqrt(statistics.mean(v * v for v in error)),
        "se_calibration": math.sqrt(statistics.mean(v["se"] ** 2 for v in rows)) / sd
        if sd
        else None,
        "empirical_sd": sd,
        "mean_se": statistics.mean(v["se"] for v in rows),
        "mean_width": statistics.mean(v["width"] for v in rows),
    }


def analyse(destination):
    manifest = verify_inputs(destination)
    registry = {c["id"]: c for c in json.loads((destination / "designs.json").read_text())}
    bounds = json.loads((destination / "binomial-bounds.json").read_text())
    expected = {(c, r) for c in registry for r in range(manifest["replicates"])}
    attempts = {}
    for record in lines(destination / "paired.jsonl"):
        key = record["case"], record["replicate"]
        if key not in expected or key in attempts:
            raise ValueError("unknown or duplicate paired attempt")
        attempts[key] = record
    if set(attempts) != expected:
        raise ValueError("missing paired attempts")
    inputs = {(v["case"], v["replicate"]): v["y"] for v in lines(destination / "inputs.jsonl")}
    summaries, paired, fit_summaries = [], [], []
    engines = {
        "candidate": ["normal", "satterthwaite", "kenward_roger"],
        "reference": ["normal", "satterthwaite", "kenward_roger"],
    }
    for case_id, case in registry.items():
        n = manifest["replicates"]
        valid = defaultdict(dict)
        diagnostics = defaultdict(
            lambda: {
                "failures": [],
                "singular": 0,
                "nonconverged": 0,
                "seconds": 0.0,
                "variance_errors": [],
            }
        )
        for rep in range(n):
            record = attempts[case_id, rep]
            for engine, methods in engines.items():
                fit = record[engine]
                diag = diagnostics[engine]
                diag["singular"] += fit.get("singular", False)
                diag["nonconverged"] += fit_status(fit) == "nonconverged"
                diag["seconds"] += fit["seconds"]
                if fit.get("converged") and all(
                    isinstance(fit.get(v), (int, float)) and math.isfinite(fit[v])
                    for v in ["sigma2", "tau2"]
                ):
                    diag["variance_errors"].append(
                        [fit["sigma2"] - (1 - case["rho"]), fit["tau2"] - case["rho"]]
                    )
                for method in methods:
                    label = f"{engine}:{method}"
                    try:
                        if not fit.get("converged"):
                            raise ValueError(fit.get("fit_error", "nonconvergence"))
                        valid[label][rep] = outcome(
                            fit["methods"].get(method, {"error": "missing method"}), case["beta"]
                        )
                    except ValueError as error:
                        diagnostics[label]["failures"].append(
                            {"replicate": rep, "reason": str(error)}
                        )
            beta, se = known_variance(case, inputs[case_id, rep])
            valid["control:known_variance"][rep] = outcome(
                {
                    "beta": beta,
                    "se": se,
                    "critical": statistics.NormalDist().inv_cdf(0.975),
                    "p": math.erfc(abs(beta / se) / math.sqrt(2)),
                },
                case["beta"],
            )
        # One resampling schedule shared by every method, preserving paired datasets.
        seed = int.from_bytes(
            hashlib.sha256(f"{PROTOCOL}|analysis|pilot|{case_id}".encode()).digest(), "big"
        )
        rng = random.Random(seed)
        resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAPS)]
        for label in [f"{e}:{m}" for e, ms in engines.items() for m in ms] + [
            "control:known_variance"
        ]:
            rows = list(valid[label].values())
            summary = {
                "case": case_id,
                "primary": case["primary"],
                "method": label,
                "attempted": n,
                "failed": n - len(rows),
                "failure_rate": binary_measure(
                    [i not in valid[label] for i in range(n)], n, bounds
                ),
                "rejection": binary_measure([r["reject"] for r in rows], n, bounds),
                "coverage": binary_measure([r["cover"] for r in rows], n, bounds),
                "performance": performance(rows),
                "bootstrap_mcse": {},
                "failure_details": diagnostics[label]["failures"],
            }
            bootstrap = [
                performance([valid[label][i] for i in ids if i in valid[label]])
                for ids in resamples
            ]
            for metric in performance(rows):
                sampled = [v[metric] for v in bootstrap if v[metric] is not None]
                summary["bootstrap_mcse"][metric] = (
                    statistics.stdev(sampled) if len(sampled) > 1 else None
                )
            if rows:
                summary.update(
                    empirical_sd=statistics.stdev(r["error"] for r in rows)
                    if len(rows) > 1
                    else None,
                    mean_se=statistics.mean(r["se"] for r in rows),
                    bias_mcse=statistics.stdev(r["error"] for r in rows) / math.sqrt(len(rows))
                    if len(rows) > 1
                    else None,
                    mean_width=statistics.mean(r["width"] for r in rows),
                    lower_noncoverage=binary_measure([r["below"] for r in rows], n, bounds),
                    upper_noncoverage=binary_measure([r["above"] for r in rows], n, bounds),
                )
            # Response summaries expose whether failures select unusual datasets.
            summary["response_selection"] = {
                name: [
                    {
                        "replicate": i,
                        "mean": statistics.mean(inputs[case_id, i]),
                        "sd": statistics.stdev(inputs[case_id, i]),
                    }
                    for i in range(n)
                    if (i in valid[label]) == evaluable
                ]
                for name, evaluable in [("evaluable_means", True), ("failed_means", False)]
            }
            summaries.append(summary)
        for engine in engines:
            diag = diagnostics[engine]
            variance = diag.pop("variance_errors")
            fit_records = [attempts[case_id, i][engine] for i in range(n)]
            counts = diagnostic_counts(fit_records, bounds)
            diag.update(counts["statuses"])
            fit_summaries.append(
                {
                    "case": case_id,
                    "engine": engine,
                    "attempted": n,
                    "diagnostics": diag,
                    "variance_bias": [
                        statistics.mean(row[i] for row in variance) if variance else None
                        for i in [0, 1]
                    ],
                    "variance_bias_mcse": [
                        statistics.stdev(row[i] for row in variance) / math.sqrt(len(variance))
                        if len(variance) > 1
                        else None
                        for i in [0, 1]
                    ],
                    "variance_evaluable": len(variance),
                    "singular_rate": counts["singular_rate"],
                    "singular_classifiable": counts["singular_classifiable"],
                    "singular_unknown": counts["singular_unknown"],
                    "fit_failure_rate": binary_measure(
                        [
                            not attempts[case_id, i][engine].get("converged", False)
                            for i in range(n)
                        ],
                        n,
                        bounds,
                    ),
                }
            )
            for method in ["satterthwaite", "kenward_roger"]:
                interior = [
                    i
                    for i in valid[f"{engine}:{method}"]
                    if not attempts[case_id, i][engine].get("singular", False)
                ]
                if case["stratum"] == "balanced" and interior:
                    expected_df = (
                        case["groups"] * (case["sizes"][0] - 1) - 1
                        if case["location"] == "within"
                        else case["groups"] - 2
                    )
                    diag[f"{method}_interior_df_max_error"] = max(
                        abs(valid[f"{engine}:{method}"][i]["df"] - expected_df) for i in interior
                    )
        for method in ["normal", "satterthwaite", "kenward_roger"]:
            a, b = valid[f"candidate:{method}"], valid[f"reference:{method}"]
            common = sorted(a.keys() & b.keys())
            record = {"case": case_id, "method": method, "common_evaluable": len(common)}
            for metric in ["reject", "cover", "beta", "se"]:
                differences = [float(a[i][metric]) - float(b[i][metric]) for i in common]
                record[metric] = {
                    "mean_difference": statistics.mean(differences) if differences else None,
                    "mcse": statistics.stdev(differences) / math.sqrt(len(differences))
                    if len(differences) > 1
                    else None,
                    "max_absolute_difference": max(map(abs, differences)) if differences else None,
                }
            paired.append(record)
    write_json(
        destination / "summary.json",
        {
            "schema_version": 1,
            "scope": "exploratory pilot; no confirmatory validation claim",
            "manifest": manifest,
            "summaries": summaries,
            "paired_differences": paired,
            "fit_summaries": fit_summaries,
            "artifacts": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
        },
    )
    print(
        f"pilot analysis: {len(attempts)} datasets, {len(summaries)} case-method rows", flush=True
    )
    for engine in engines:
        if not any(
            row["failed"] < row["attempted"]
            for row in summaries
            if row["method"].startswith(f"{engine}:")
        ):
            raise ValueError(f"no evaluable {engine} outcomes; inspect retained summary")


def reanalyse(destination):
    """Correct only diagnostic labels in immutable completed pilot evidence."""
    original_path = destination / "summary.json"
    report = json.loads(original_path.read_text())
    manifest = report["manifest"]
    if manifest["namespace"] != "pilot" or manifest["protocol"] != PROTOCOL:
        raise ValueError("unsupported original pilot")
    for name, expected in report["artifacts"].items():
        if Path(name).name != name or digest(destination / name) != expected:
            raise ValueError(f"original evidence changed: {name}")
    for name, expected in manifest["sources"].items():
        if digest(destination / "frozen-sources" / name) != expected:
            raise ValueError(f"frozen original source changed: {name}")
    bounds = json.loads((destination / "binomial-bounds.json").read_text())
    grouped = defaultdict(list)
    for row in lines(destination / "paired.jsonl"):
        for engine in ["candidate", "reference"]:
            grouped[row["case"], engine].append(row[engine])
    for summary in report["fit_summaries"]:
        fits = grouped[summary["case"], summary["engine"]]
        if len(fits) != summary["attempted"]:
            raise ValueError("attempt count changed")
        counts = diagnostic_counts(fits, bounds)
        summary["diagnostics"].update(counts["statuses"])
        for name in ["singular_rate", "singular_classifiable", "singular_unknown"]:
            summary[name] = counts[name]
    report["analysis_revision"] = {
        "id": "v2-diagnostics-2",
        "original_summary_sha256": digest(original_path),
        "analysis_source_sha256": digest(Path(__file__)),
        "changes": [
            "separate exceptions and unknown convergence from nonconvergence",
            "do not count unknown singularity as nonsingular",
        ],
        "fits_repeated": 0,
        "responses_replaced": 0,
    }
    write_json(destination / "summary-reanalysis.json", report)
    print("diagnostic reanalysis complete; original summary and all fits retained", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["generate", "candidate", "analyse", "reanalyse"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--replicates", type=int, default=100)
    args = parser.parse_args()
    if args.operation == "generate":
        generate(args.directory, args.replicates)
    elif args.operation == "candidate":
        candidate(args.directory)
    elif args.operation == "analyse":
        analyse(args.directory)
    else:
        reanalyse(args.directory)


if __name__ == "__main__":
    main()
