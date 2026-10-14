"""LMM calibration runner: exploratory pilot (v2) and frozen confirmatory study (v3).

Pilot runs (`namespace="pilot"`, `lmm-calibration-v2`) accept 1..100 replicates
and never make a confirmatory validation claim. Confirmatory runs
(`namespace="confirmatory"`, `lmm-calibration-v3`) use a disjoint seed
namespace, require 20,000 candidate replicates and at least 1,000 paired
reference replicates per case, execute with bounded workers and per-case
compressed shards, and evaluate the prespecified 48-claim Bonferroni
Clopper-Pearson decision rules.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import operator
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROTOCOL = "lmm-calibration-v2"
CONFIRMATORY_PROTOCOL = "lmm-calibration-v3"
CONFIRMATORY_REPLICATES = 20000
DEFAULT_REFERENCE_REPLICATES = 1000
MAX_WORKERS = 16
FAMILY = 48
BOOTSTRAPS = 1000
REJECTION_BAND = [0.04, 0.06]
COVERAGE_BAND = [0.94, 0.96]
ROOT = Path(__file__).resolve().parents[1]

PILOT_SOURCES = [
    "scripts/lmm_calibration.py",
    "scripts/ci/lme_ci.py",
    "comparisons/scientific/lmm_calibration.R",
    "docs/LMM_CALIBRATION_PROTOCOL.md",
    "docs/LMM_CALIBRATION_AI_REVIEW.md",
    "scientific/pyproject.toml",
    "scientific/uv.lock",
    "scientific/cases.json",
]

CONFIRMATORY_SOURCES = PILOT_SOURCES + [
    "docs/LMM_CALIBRATION_IMPLEMENTATION_REVIEW.md",
    "docs/LMM_CALIBRATION_CONFIRMATORY_REVIEW.md",
    "comparisons/scientific/lmm_confirmation_planning.R",
    "comparisons/scientific/restore_r_environment.R",
    "scientific/renv.lock",
]

THREAD_LIMITS = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value):
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_gzip_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as gz:
            for record in records:
                line = json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                gz.write(line.encode("utf-8"))


def gzip_lines(path):
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def emit(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def worker_env():
    return {**os.environ, **THREAD_LIMITS}


def run_workers(build_cmd, worker_count):
    if not 1 <= worker_count <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    env = worker_env()
    procs = [
        subprocess.Popen(build_cmd(idx, worker_count), cwd=ROOT, env=env)
        for idx in range(worker_count)
    ]
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"worker failure: exit codes {codes}")


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


def response(case, replicate, *, protocol=PROTOCOL, namespace="pilot"):
    key = f"{protocol}|{namespace}|{case['id']}|{replicate}"
    draws = normals(key, case["groups"] + len(case["x"]))
    b = [math.sqrt(case["rho"]) * v for v in draws[: case["groups"]]]
    e = draws[case["groups"] :]
    return [
        2 + case["beta"] * x + b[g] + math.sqrt(1 - case["rho"]) * error
        for x, g, error in zip(case["x"], case["group"], e)
    ]


def validate_confirmatory_budget(replicates, reference_replicates, workers, allow_test_budget):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if allow_test_budget:
        if replicates < 1 or not 1 <= reference_replicates <= replicates:
            raise ValueError("invalid test confirmatory replicate budget")
    elif replicates != CONFIRMATORY_REPLICATES or not 1000 <= reference_replicates <= replicates:
        raise ValueError(
            f"confirmatory namespace requires {CONFIRMATORY_REPLICATES} replicates "
            "and 1000..20000 reference replicates"
        )


def generate_confirmatory_worker(destination, replicates, worker_index, worker_count):
    registry = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
    for case in registry[worker_index::worker_count]:
        shard = destination / "cases" / case["id"] / "inputs.jsonl.gz"
        write_gzip_jsonl(
            shard,
            (
                {
                    "case": case["id"],
                    "replicate": rep,
                    "y": response(
                        case, rep, protocol=CONFIRMATORY_PROTOCOL, namespace="confirmatory"
                    ),
                }
                for rep in range(replicates)
            ),
        )


def generate(
    destination,
    replicates,
    *,
    namespace="pilot",
    reference_replicates=DEFAULT_REFERENCE_REPLICATES,
    workers=1,
    _allow_test_budget=False,
):
    if namespace == "pilot":
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
                "sources": {name: digest(ROOT / name) for name in PILOT_SOURCES},
                "candidate_control": {
                    "reml": True,
                    "tolerance": 1e-6,
                    "max_iterations": 1000,
                    "require_convergence": False,
                },
                "reference_control": {"optimizer": "bobyqa", "maxfun": 100000},
            },
        )
        return

    if namespace != "confirmatory":
        raise ValueError(f"unsupported namespace: {namespace}")
    validate_confirmatory_budget(replicates, reference_replicates, workers, _allow_test_budget)
    destination.mkdir(parents=True, exist_ok=False)
    registry = cases()
    write_json(destination / "designs.json", registry)
    active_workers = min(workers, len(registry))
    if active_workers == 1:
        generate_confirmatory_worker(destination, replicates, 0, 1)
    else:
        run_workers(
            lambda idx, count: [
                sys.executable,
                str(Path(__file__).resolve()),
                "generate-worker",
                str(destination),
                "--replicates",
                str(replicates),
                "--worker-index",
                str(idx),
                "--worker-count",
                str(count),
            ],
            active_workers,
        )
    case_inputs_sha256 = {}
    for case in registry:
        shard = destination / "cases" / case["id"] / "inputs.jsonl.gz"
        if not shard.is_file():
            raise ValueError(f"missing generated input shard: {case['id']}")
        case_inputs_sha256[case["id"]] = digest(shard)
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 2,
            "protocol": CONFIRMATORY_PROTOCOL,
            "namespace": "confirmatory",
            "replicates": replicates,
            "reference_replicates": reference_replicates,
            "attempted_datasets": len(registry) * replicates,
            "reference_attempted_datasets": len(registry) * reference_replicates,
            "workers": workers,
            "max_workers": MAX_WORKERS,
            "thread_limits": THREAD_LIMITS,
            "bootstrap_repetitions": BOOTSTRAPS,
            "primary_family": FAMILY,
            "rejection_band": REJECTION_BAND,
            "coverage_band": COVERAGE_BAND,
            "candidate_version": "0.2.7",
            "candidate_release_git_sha": subprocess.check_output(
                ["git", "rev-list", "-n", "1", "v0.2.7"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "inputs_sha256": json_digest(case_inputs_sha256),
            "case_inputs_sha256": case_inputs_sha256,
            "designs_sha256": digest(destination / "designs.json"),
            "sources": {name: digest(ROOT / name) for name in CONFIRMATORY_SOURCES},
            "candidate_control": {
                "reml": True,
                "tolerance": 1e-6,
                "max_iterations": 1000,
                "require_convergence": False,
            },
            "reference_control": {"optimizer": "bobyqa", "maxfun": 100000},
        },
    )


def verify_inputs(destination, *, _allow_test_budget=False):
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    namespace = manifest.get("namespace")
    if namespace == "pilot":
        if manifest["protocol"] != PROTOCOL or not 1 <= manifest["replicates"] <= 100:
            raise ValueError("unsupported protocol/namespace/replicate budget")
        for name in ["inputs", "designs"]:
            suffix = "jsonl" if name == "inputs" else "json"
            if digest(destination / f"{name}.{suffix}") != manifest[f"{name}_sha256"]:
                raise ValueError(f"{name} hash changed")
    elif namespace == "confirmatory":
        if manifest["protocol"] != CONFIRMATORY_PROTOCOL:
            raise ValueError("unsupported confirmatory protocol")
        validate_confirmatory_budget(
            manifest["replicates"],
            manifest.get("reference_replicates", 0),
            manifest.get("workers", 1),
            _allow_test_budget,
        )
        if digest(destination / "designs.json") != manifest["designs_sha256"]:
            raise ValueError("designs hash changed")
        registry = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
        expected_cases = manifest.get("case_inputs_sha256", {})
        if set(expected_cases) != {c["id"] for c in registry}:
            raise ValueError("case input hash registry mismatch")
        for case in registry:
            shard = destination / "cases" / case["id"] / "inputs.jsonl.gz"
            if not shard.is_file() or digest(shard) != expected_cases[case["id"]]:
                raise ValueError(f"inputs hash changed: {case['id']}")
        if json_digest(expected_cases) != manifest["inputs_sha256"]:
            raise ValueError("aggregate inputs hash changed")
    else:
        raise ValueError("unsupported protocol/namespace/replicate budget")
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


def evaluate_candidate_methods(fit, data, beta):
    methods = {}
    try:
        se = fit.std_errors[1]
        if not math.isfinite(se) or se <= 0:
            raise ValueError("invalid SE")
        methods["normal"] = {
            "beta": beta,
            "se": se,
            "df": None,
            "f": (beta / se) ** 2,
            "p": math.erfc(abs(beta / se) / math.sqrt(2)),
        }
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        methods["normal"] = {"error": f"{type(error).__name__}: {error}"}
    for name in ["satterthwaite", "kenward_roger"]:
        try:
            getattr(fit, f"with_{name}")(data)
            test = fit.test_contrast([[0.0, 1.0]], ddf_method=name)
            # The public Python API does not expose adjusted covariance.
            # A unit contrast displacement avoids beta^2/F near zero.
            unit = fit.test_contrast_vs(
                [[0.0, 1.0]], [fit.coefficients[0], beta - 1], ddf_method=name
            )
            if (
                test.num_df != 1
                or unit.num_df != 1
                or not math.isfinite(unit.f_value)
                or unit.f_value <= 0
            ):
                raise ValueError("scalar test/variance inconsistency")
            if (
                not math.isfinite(test.den_df)
                or test.den_df <= 0
                or not math.isfinite(unit.den_df)
                or not math.isclose(test.den_df, unit.den_df, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("invalid df")
            variance = 1 / unit.f_value
            expected_f = beta * beta / variance
            if not math.isclose(test.f_value, expected_f, rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("scalar test/variance inconsistency")
            methods[name] = {
                "beta": beta,
                "se": math.sqrt(variance),
                "df": test.den_df,
                "f": test.f_value,
                "p": test.p_value,
            }
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            methods[name] = {"error": f"{type(error).__name__}: {error}"}
    return methods


def fit_candidate_dataset(lme, data, case_id, replicate):
    record = {"case": case_id, "replicate": replicate, "methods": {}}
    before = time.perf_counter()
    try:
        fit = lme.lmer(
            "y ~ x + (1 | group)",
            data,
            reml=True,
            control=lme.FitControl(tolerance=1e-6, max_iterations=1000, require_convergence=False),
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
        record["methods"] = evaluate_candidate_methods(fit, data, beta)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if record.get("converged") is not False:
            record["converged"] = False
        record["fit_error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = time.perf_counter() - before
    return finite_json(record)


def candidate_confirmatory_worker(destination, worker_index, worker_count):
    import lme_python as lme
    import polars as pl

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
    n_expected = manifest["replicates"]
    for case in registry[worker_index::worker_count]:
        case_dir = destination / "cases" / case["id"]
        out_path = case_dir / "candidate.jsonl.gz"
        base = pl.DataFrame({"x": case["x"], "group": [f"g{g:03d}" for g in case["group"]]})

        def generate_records(case=case, case_dir=case_dir, base=base):
            count = 0
            for input_ in gzip_lines(case_dir / "inputs.jsonl.gz"):
                if input_["case"] != case["id"] or input_["replicate"] != count:
                    raise ValueError(f"input sequence mismatch in {case['id']}")
                data = base.with_columns(pl.Series("y", input_["y"], dtype=pl.Float64))
                yield fit_candidate_dataset(lme, data, case["id"], input_["replicate"])
                count += 1
            if count != n_expected:
                raise ValueError(f"incomplete candidate inputs in {case['id']}")

        write_gzip_jsonl(out_path, generate_records())
        print(
            f"candidate worker {worker_index + 1}/{worker_count} completed "
            f"{case['id']} ({n_expected} fits)",
            flush=True,
        )


def candidate(destination, *, workers=None, _allow_test_budget=False):
    manifest = verify_inputs(destination, _allow_test_budget=_allow_test_budget)
    import lme_python as lme
    import polars as pl
    from lme_python import lme_python as native

    if importlib.metadata.version("lme-python") != manifest["candidate_version"]:
        raise ValueError("candidate version differs from frozen release")
    if not Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("candidate must be installed in the isolated environment")
    registry_list = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
    registry = {c["id"]: c for c in registry_list}
    started = time.perf_counter()
    if manifest["namespace"] == "pilot":
        with (destination / "candidate.jsonl").open("x", encoding="utf-8") as stream:
            for index, input_ in enumerate(lines(destination / "inputs.jsonl")):
                case = registry[input_["case"]]
                data = pl.DataFrame(
                    {
                        "y": input_["y"],
                        "x": case["x"],
                        "group": [f"g{g:03d}" for g in case["group"]],
                    }
                )
                record = fit_candidate_dataset(lme, data, case["id"], input_["replicate"])
                # Nonfinite values remain recorded as strings and fail analysis validation.
                emit(stream, record)
                if (index + 1) % 100 == 0:
                    print(f"candidate: {index + 1}/{manifest['attempted_datasets']}", flush=True)
        write_json(
            destination / "candidate-environment.json",
            {
                "engine": importlib.metadata.version("lme-python"),
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "polars": pl.__version__,
                "native_sha256": digest(Path(native.__file__)),
                "seconds": time.perf_counter() - started,
            },
        )
        return

    active_workers = min(workers or manifest.get("workers", 1), len(registry_list))
    run_workers(
        lambda idx, count: [
            sys.executable,
            str(Path(__file__).resolve()),
            "candidate-worker",
            str(destination),
            "--worker-index",
            str(idx),
            "--worker-count",
            str(count),
        ],
        active_workers,
    )
    case_candidate_sha256 = {}
    for case in registry_list:
        shard = destination / "cases" / case["id"] / "candidate.jsonl.gz"
        if not shard.is_file():
            raise ValueError(f"missing candidate shard: {case['id']}")
        case_candidate_sha256[case["id"]] = digest(shard)
    write_json(
        destination / "candidate-environment.json",
        {
            "engine": importlib.metadata.version("lme-python"),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "polars": pl.__version__,
            "native_sha256": digest(Path(native.__file__)),
            "workers": active_workers,
            "thread_limits": THREAD_LIMITS,
            "case_candidate_sha256": case_candidate_sha256,
            "seconds": time.perf_counter() - started,
        },
    )


def resolve_rscript(rscript=""):
    runner = rscript or shutil.which("Rscript")
    if not runner and os.name == "nt":
        candidate_path = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "R/R-4.6.1/bin/Rscript.exe"
        )
        if candidate_path.is_file():
            runner = str(candidate_path)
    if not runner:
        raise ValueError("Rscript required; pass --rscript PATH")
    return runner


def reference(destination, *, rscript="", workers=None, _allow_test_budget=False):
    manifest = verify_inputs(destination, _allow_test_budget=_allow_test_budget)
    runner = resolve_rscript(rscript)
    script = str(ROOT / "comparisons/scientific/lmm_calibration.R")
    if manifest["namespace"] == "pilot":
        subprocess.run([runner, "--vanilla", script, str(destination)], cwd=ROOT, check=True)
        return
    registry_list = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
    active_workers = min(workers or manifest.get("workers", 1), len(registry_list))
    started = time.perf_counter()
    run_workers(
        lambda idx, count: [
            runner,
            "--vanilla",
            script,
            str(destination),
            "--worker",
            str(idx + 1),
            str(count),
        ],
        active_workers,
    )
    subprocess.run(
        [runner, "--vanilla", script, str(destination), "--finalize"],
        cwd=ROOT,
        env=worker_env(),
        check=True,
    )
    ref_env_path = destination / "reference-environment.json"
    ref_env = json.loads(ref_env_path.read_text(encoding="utf-8"))
    ref_env["workers"] = active_workers
    ref_env["thread_limits"] = THREAD_LIMITS
    ref_env["wall_seconds"] = time.perf_counter() - started
    ref_env["case_distributions_sha256"] = {
        c["id"]: digest(destination / "cases" / c["id"] / "candidate-distributions.json")
        for c in registry_list
    }
    ref_env["case_reference_sha256"] = {
        c["id"]: digest(destination / "cases" / c["id"] / "reference.jsonl") for c in registry_list
    }
    ref_env_path.write_text(json.dumps(ref_env, allow_nan=False, indent=2) + "\n", encoding="utf-8")


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


def binary_measure(
    values, attempted, bounds, *, pointwise_key="pointwise", simultaneous_key="simultaneous"
):
    n, k = len(values), sum(values)
    failed = attempted - n
    return {
        "evaluable": n,
        "count": k,
        "rate": k / n if n else None,
        "mcse": math.sqrt(k / n * (1 - k / n) / n) if n else None,
        "wilson95": wilson(k, n),
        "unconditional_identification_bounds": [k / attempted, (k + failed) / attempted],
        "failure_envelope95": [bounds[pointwise_key][k][0], bounds[pointwise_key][k + failed][1]],
        "failure_envelope_simultaneous95": [
            bounds[simultaneous_key][k][0],
            bounds[simultaneous_key][k + failed][1],
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


def diagnostic_counts(fits, bounds, *, pointwise_key="pointwise", simultaneous_key="simultaneous"):
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
        "singular_rate": binary_measure(
            flags,
            len(fits),
            bounds,
            pointwise_key=pointwise_key,
            simultaneous_key=simultaneous_key,
        ),
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


def fast_bootstrap_mcse(valid_by_label, labels, n, resamples):
    """Compute bootstrap MCSE identically to performance() using C itemgetter + fsum."""
    metrics = ["bias", "rmse", "se_calibration", "empirical_sd", "mean_se", "mean_width"]
    result = {label: {m: None for m in metrics} for label in labels}
    full_getters = None
    for label in labels:
        valid_map = valid_by_label[label]
        if len(valid_map) < 2:
            continue
        if len(valid_map) == n:
            if full_getters is None:
                full_getters = [operator.itemgetter(*ids) for ids in resamples]
            err = [valid_map[i]["error"] for i in range(n)]
            err2 = [e * e for e in err]
            se = [valid_map[i]["se"] for i in range(n)]
            se2 = [s * s for s in se]
            width = [valid_map[i]["width"] for i in range(n)]
            samples = {m: [] for m in metrics}
            for get in full_getters:
                s_e = math.fsum(get(err))
                s_e2 = math.fsum(get(err2))
                s_s = math.fsum(get(se))
                s_s2 = math.fsum(get(se2))
                s_w = math.fsum(get(width))
                mean_e = s_e / n
                var = max(0.0, (s_e2 - s_e * mean_e) / (n - 1))
                sd = math.sqrt(var)
                samples["bias"].append(mean_e)
                samples["rmse"].append(math.sqrt(s_e2 / n))
                if sd:
                    samples["se_calibration"].append(math.sqrt(s_s2 / n) / sd)
                samples["empirical_sd"].append(sd)
                samples["mean_se"].append(s_s / n)
                samples["mean_width"].append(s_w / n)
            for m in metrics:
                if len(samples[m]) > 1:
                    result[label][m] = statistics.stdev(samples[m])
        else:
            bootstrap = [
                performance([valid_map[i] for i in ids if i in valid_map]) for ids in resamples
            ]
            for m in metrics:
                sampled = [v[m] for v in bootstrap if v[m] is not None]
                if len(sampled) > 1:
                    result[label][m] = statistics.stdev(sampled)
    return result


def summarize_response_selection(response_stats, valid_map, n):
    eval_stats = [response_stats[i] for i in range(n) if i in valid_map]
    fail_stats = [response_stats[i] for i in range(n) if i not in valid_map]

    def aggregate(rows):
        return {
            "count": len(rows),
            "mean_of_means": statistics.mean(r["mean"] for r in rows) if rows else None,
            "sd_of_means": statistics.stdev(r["mean"] for r in rows) if len(rows) > 1 else None,
            "mean_of_sds": statistics.mean(r["sd"] for r in rows) if rows else None,
            "sd_of_sds": statistics.stdev(r["sd"] for r in rows) if len(rows) > 1 else None,
        }

    return {
        "evaluable_summary": aggregate(eval_stats),
        "failed_summary": aggregate(fail_stats),
        "failed_means": fail_stats,
    }


def classify_band(envelope, band):
    lo, hi = envelope
    b_lo, b_hi = band
    if lo >= b_lo and hi <= b_hi:
        return "supported"
    if hi < b_lo or lo > b_hi:
        return "contradicted"
    return "inconclusive"


def build_confirmatory_decisions(summaries):
    claims = []
    by_method = {
        method: {
            "primary_cases": 0,
            "supported": 0,
            "inconclusive": 0,
            "contradicted": 0,
            "validated_for_primary_scope": False,
            "inconclusive_cases": [],
            "contradicted_cases": [],
        }
        for method in ["candidate:satterthwaite", "candidate:kenward_roger"]
    }
    comparators = []
    for row in summaries:
        if not row["primary"]:
            continue
        rej_env = row["rejection"]["failure_envelope_simultaneous95"]
        cov_env = row["coverage"]["failure_envelope_simultaneous95"]
        rej_decision = classify_band(rej_env, REJECTION_BAND)
        cov_decision = classify_band(cov_env, COVERAGE_BAND)
        complement = (
            rej_decision == cov_decision
            and row["rejection"]["count"] + row["coverage"]["count"]
            == row["rejection"]["evaluable"]
        )
        if not complement:
            raise ValueError(
                f"null rejection/coverage complement failure in {row['case']} {row['method']}"
            )
        entry = {
            "case": row["case"],
            "method": row["method"],
            "attempted": row["attempted"],
            "evaluable": row["rejection"]["evaluable"],
            "failed": row["failed"],
            "rejection_rate": row["rejection"]["rate"],
            "coverage_rate": row["coverage"]["rate"],
            "rejection_envelope_simultaneous95": rej_env,
            "coverage_envelope_simultaneous95": cov_env,
            "decision": rej_decision,
            "complement_verified": complement,
        }
        if row["method"] in by_method:
            claims.append(entry)
            m_summary = by_method[row["method"]]
            m_summary["primary_cases"] += 1
            m_summary[rej_decision] += 1
            if rej_decision == "inconclusive":
                m_summary["inconclusive_cases"].append(row["case"])
            elif rej_decision == "contradicted":
                m_summary["contradicted_cases"].append(row["case"])
        else:
            comparators.append(entry)
    if len(claims) != FAMILY:
        raise ValueError(f"expected {FAMILY} primary claims, found {len(claims)}")
    for method, m_summary in by_method.items():
        if m_summary["primary_cases"] != FAMILY // 2:
            raise ValueError(f"incomplete primary case count for {method}")
        m_summary["validated_for_primary_scope"] = (
            m_summary["supported"] == m_summary["primary_cases"]
        )
    return {
        "family_size": FAMILY,
        "rejection_band": REJECTION_BAND,
        "coverage_band": COVERAGE_BAND,
        "claims": claims,
        "methods": by_method,
        "comparators": comparators,
    }


def analyse_confirmatory_case(destination, case, manifest, bounds):
    case_id = case["id"]
    case_dir = destination / "cases" / case_id
    n_cand = manifest["replicates"]
    n_ref = manifest["reference_replicates"]

    dist = json.loads((case_dir / "candidate-distributions.json").read_text(encoding="utf-8"))
    for mname in ["normal", "satterthwaite", "kenward_roger"]:
        for key in ["critical", "distribution_p", "error"]:
            if len(dist[mname][key]) != n_cand:
                raise ValueError(f"distribution length mismatch in {case_id} {mname}.{key}")

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
    cand_fits = []
    response_stats = []
    norm_crit = statistics.NormalDist().inv_cdf(0.975)

    in_iter = gzip_lines(case_dir / "inputs.jsonl.gz")
    cand_iter = gzip_lines(case_dir / "candidate.jsonl.gz")
    for rep in range(n_cand):
        try:
            in_row = next(in_iter)
            cand_row = next(cand_iter)
        except StopIteration as error:
            raise ValueError(f"truncated shard in {case_id}") from error
        if (
            in_row["case"] != case_id
            or in_row["replicate"] != rep
            or cand_row["case"] != case_id
            or cand_row["replicate"] != rep
        ):
            raise ValueError(f"shard alignment mismatch in {case_id} replicate {rep}")
        y = in_row["y"]
        response_stats.append(
            {"replicate": rep, "mean": statistics.mean(y), "sd": statistics.stdev(y)}
        )
        cand_fits.append(cand_row)
        diag = diagnostics["candidate"]
        diag["singular"] += cand_row.get("singular", False) is True
        diag["nonconverged"] += fit_status(cand_row) == "nonconverged"
        diag["seconds"] += cand_row["seconds"]
        if cand_row.get("converged") and all(
            isinstance(cand_row.get(v), (int, float)) and math.isfinite(cand_row[v])
            for v in ["sigma2", "tau2"]
        ):
            diag["variance_errors"].append(
                [cand_row["sigma2"] - (1 - case["rho"]), cand_row["tau2"] - case["rho"]]
            )
        for method in ["normal", "satterthwaite", "kenward_roger"]:
            label = f"candidate:{method}"
            try:
                if not cand_row.get("converged"):
                    raise ValueError(cand_row.get("fit_error", "nonconvergence"))
                raw_m = cand_row["methods"].get(method, {"error": "missing method"})
                if "error" in raw_m:
                    raise ValueError(raw_m["error"])
                r_err = dist[method]["error"][rep]
                if r_err is not None:
                    raise ValueError(r_err)
                enriched = {
                    **raw_m,
                    "critical": dist[method]["critical"][rep],
                    "distribution_p": dist[method]["distribution_p"][rep],
                }
                valid[label][rep] = outcome(enriched, case["beta"])
            except ValueError as error:
                diagnostics[label]["failures"].append({"replicate": rep, "reason": str(error)})

        beta, se = known_variance(case, y)
        valid["control:known_variance"][rep] = outcome(
            {
                "beta": beta,
                "se": se,
                "critical": norm_crit,
                "p": math.erfc(abs(beta / se) / math.sqrt(2)),
            },
            case["beta"],
        )

    if next(in_iter, None) is not None or next(cand_iter, None) is not None:
        raise ValueError(f"extra shard records in {case_id}")

    ref_fits = []
    for rep, ref_row in enumerate(lines(case_dir / "reference.jsonl")):
        if ref_row["case"] != case_id or ref_row["replicate"] != rep:
            raise ValueError(f"reference replicate mismatch in {case_id}")
        fit = ref_row["reference"]
        ref_fits.append(fit)
        diag = diagnostics["reference"]
        diag["singular"] += fit.get("singular", False) is True
        diag["nonconverged"] += fit_status(fit) == "nonconverged"
        diag["seconds"] += fit["seconds"]
        if fit.get("converged") and all(
            isinstance(fit.get(v), (int, float)) and math.isfinite(fit[v])
            for v in ["sigma2", "tau2"]
        ):
            diag["variance_errors"].append(
                [fit["sigma2"] - (1 - case["rho"]), fit["tau2"] - case["rho"]]
            )
        for method in ["normal", "satterthwaite", "kenward_roger"]:
            label = f"reference:{method}"
            try:
                if not fit.get("converged"):
                    raise ValueError(fit.get("fit_error", "nonconvergence"))
                valid[label][rep] = outcome(
                    fit["methods"].get(method, {"error": "missing method"}), case["beta"]
                )
            except ValueError as error:
                diagnostics[label]["failures"].append({"replicate": rep, "reason": str(error)})
    if len(ref_fits) != n_ref:
        raise ValueError(f"missing reference attempts in {case_id}")

    cand_labels = [
        "candidate:normal",
        "candidate:satterthwaite",
        "candidate:kenward_roger",
        "control:known_variance",
    ]
    ref_labels = ["reference:normal", "reference:satterthwaite", "reference:kenward_roger"]

    seed_cand = int.from_bytes(
        hashlib.sha256(
            f"{CONFIRMATORY_PROTOCOL}|analysis|confirmatory|{case_id}".encode()
        ).digest(),
        "big",
    )
    rng_cand = random.Random(seed_cand)
    resamples_cand = [
        [rng_cand.randrange(n_cand) for _ in range(n_cand)] for _ in range(BOOTSTRAPS)
    ]
    mcse_cand = fast_bootstrap_mcse(valid, cand_labels, n_cand, resamples_cand)

    if n_ref == n_cand:
        resamples_ref = resamples_cand
    else:
        seed_ref = int.from_bytes(
            hashlib.sha256(
                f"{CONFIRMATORY_PROTOCOL}|analysis|confirmatory|reference|{case_id}".encode()
            ).digest(),
            "big",
        )
        rng_ref = random.Random(seed_ref)
        resamples_ref = [
            [rng_ref.randrange(n_ref) for _ in range(n_ref)] for _ in range(BOOTSTRAPS)
        ]
    mcse_ref = fast_bootstrap_mcse(valid, ref_labels, n_ref, resamples_ref)
    mcse_all = {**mcse_cand, **mcse_ref}

    case_summaries = []
    all_labels = [
        "candidate:normal",
        "candidate:satterthwaite",
        "candidate:kenward_roger",
        "reference:normal",
        "reference:satterthwaite",
        "reference:kenward_roger",
        "control:known_variance",
    ]
    for label in all_labels:
        is_ref = label.startswith("reference:")
        n = n_ref if is_ref else n_cand
        pw_key = "reference_pointwise" if is_ref else "pointwise"
        sim_key = "reference_simultaneous" if is_ref else "simultaneous"
        rows = [valid[label][i] for i in sorted(valid[label])]
        summary = {
            "case": case_id,
            "primary": case["primary"],
            "method": label,
            "attempted": n,
            "failed": n - len(rows),
            "failure_rate": binary_measure(
                [i not in valid[label] for i in range(n)],
                n,
                bounds,
                pointwise_key=pw_key,
                simultaneous_key=sim_key,
            ),
            "rejection": binary_measure(
                [r["reject"] for r in rows],
                n,
                bounds,
                pointwise_key=pw_key,
                simultaneous_key=sim_key,
            ),
            "coverage": binary_measure(
                [r["cover"] for r in rows],
                n,
                bounds,
                pointwise_key=pw_key,
                simultaneous_key=sim_key,
            ),
            "performance": performance(rows),
            "bootstrap_mcse": mcse_all[label],
            "failure_details": diagnostics[label]["failures"],
        }
        if rows:
            summary.update(
                empirical_sd=statistics.stdev(r["error"] for r in rows) if len(rows) > 1 else None,
                mean_se=statistics.mean(r["se"] for r in rows),
                bias_mcse=statistics.stdev(r["error"] for r in rows) / math.sqrt(len(rows))
                if len(rows) > 1
                else None,
                mean_width=statistics.mean(r["width"] for r in rows),
                lower_noncoverage=binary_measure(
                    [r["below"] for r in rows],
                    n,
                    bounds,
                    pointwise_key=pw_key,
                    simultaneous_key=sim_key,
                ),
                upper_noncoverage=binary_measure(
                    [r["above"] for r in rows],
                    n,
                    bounds,
                    pointwise_key=pw_key,
                    simultaneous_key=sim_key,
                ),
            )
        summary["response_selection"] = summarize_response_selection(
            response_stats[:n], valid[label], n
        )
        case_summaries.append(summary)

    case_fit_summaries = []
    for engine, fit_records, n, pw_key, sim_key in [
        ("candidate", cand_fits, n_cand, "pointwise", "simultaneous"),
        ("reference", ref_fits, n_ref, "reference_pointwise", "reference_simultaneous"),
    ]:
        diag = diagnostics[engine]
        variance = diag.pop("variance_errors")
        counts = diagnostic_counts(
            fit_records, bounds, pointwise_key=pw_key, simultaneous_key=sim_key
        )
        diag.update(counts["statuses"])
        for method in ["satterthwaite", "kenward_roger"]:
            interior = [
                i for i in valid[f"{engine}:{method}"] if not fit_records[i].get("singular", False)
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
        case_fit_summaries.append(
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
                    [not r.get("converged", False) for r in fit_records],
                    n,
                    bounds,
                    pointwise_key=pw_key,
                    simultaneous_key=sim_key,
                ),
            }
        )

    case_paired = []
    for method in ["normal", "satterthwaite", "kenward_roger"]:
        a = {k: v for k, v in valid[f"candidate:{method}"].items() if k < n_ref}
        b = valid[f"reference:{method}"]
        common = sorted(a.keys() & b.keys())
        record = {
            "case": case_id,
            "method": method,
            "paired_attempted": n_ref,
            "common_evaluable": len(common),
        }
        for metric in ["reject", "cover", "beta", "se"]:
            differences = [float(a[i][metric]) - float(b[i][metric]) for i in common]
            record[metric] = {
                "mean_difference": statistics.mean(differences) if differences else None,
                "mcse": statistics.stdev(differences) / math.sqrt(len(differences))
                if len(differences) > 1
                else None,
                "max_absolute_difference": max(map(abs, differences)) if differences else None,
            }
        case_paired.append(record)

    return {
        "summaries": case_summaries,
        "fit_summaries": case_fit_summaries,
        "paired_differences": case_paired,
    }


def analyse_confirmatory_worker(destination, worker_index, worker_count):
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
    bounds = json.loads((destination / "binomial-bounds.json").read_text(encoding="utf-8"))
    for case in registry[worker_index::worker_count]:
        out_path = destination / "cases" / case["id"] / "case-summary.json"
        write_json(out_path, analyse_confirmatory_case(destination, case, manifest, bounds))


def analyse(destination, *, workers=None, _allow_test_budget=False):
    manifest = verify_inputs(destination, _allow_test_budget=_allow_test_budget)
    registry_list = json.loads((destination / "designs.json").read_text(encoding="utf-8"))
    registry = {c["id"]: c for c in registry_list}
    bounds = json.loads((destination / "binomial-bounds.json").read_text(encoding="utf-8"))
    engines = {
        "candidate": ["normal", "satterthwaite", "kenward_roger"],
        "reference": ["normal", "satterthwaite", "kenward_roger"],
    }

    if manifest["namespace"] == "confirmatory":
        active_workers = min(workers or manifest.get("workers", 1), len(registry_list))
        if active_workers == 1:
            analyse_confirmatory_worker(destination, 0, 1)
        else:
            run_workers(
                lambda idx, count: [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "analyse-worker",
                    str(destination),
                    "--worker-index",
                    str(idx),
                    "--worker-count",
                    str(count),
                ],
                active_workers,
            )
        summaries, paired, fit_summaries = [], [], []
        case_summary_sha256 = {}
        for case in registry_list:
            case_summary_path = destination / "cases" / case["id"] / "case-summary.json"
            case_summary_sha256[case["id"]] = digest(case_summary_path)
            part = json.loads(case_summary_path.read_text(encoding="utf-8"))
            summaries.extend(part["summaries"])
            fit_summaries.extend(part["fit_summaries"])
            paired.extend(part["paired_differences"])
        decisions = build_confirmatory_decisions(summaries)
        artifacts = {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()}
        artifacts["case_summaries_sha256"] = json_digest(case_summary_sha256)
        write_json(
            destination / "summary.json",
            {
                "schema_version": 2,
                "scope": (
                    "frozen confirmatory LMM calibration study; "
                    "decisions apply only to the declared 24 balanced interior null cases"
                ),
                "manifest": manifest,
                "confirmatory_decisions": decisions,
                "summaries": summaries,
                "paired_differences": paired,
                "fit_summaries": fit_summaries,
                "artifacts": artifacts,
            },
        )
        print(
            f"confirmatory analysis: {manifest['attempted_datasets']} candidate datasets, "
            f"{manifest['reference_attempted_datasets']} paired reference datasets, "
            f"{len(summaries)} case-method rows",
            flush=True,
        )
        for engine in engines:
            if not any(
                row["failed"] < row["attempted"]
                for row in summaries
                if row["method"].startswith(f"{engine}:")
            ):
                raise ValueError(f"no evaluable {engine} outcomes; inspect retained summary")
        return

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
    report = json.loads(original_path.read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if manifest["namespace"] != "pilot" or manifest["protocol"] != PROTOCOL:
        raise ValueError("unsupported original pilot")
    for name, expected in report["artifacts"].items():
        if Path(name).name != name or digest(destination / name) != expected:
            raise ValueError(f"original evidence changed: {name}")
    for name, expected in manifest["sources"].items():
        if digest(destination / "frozen-sources" / name) != expected:
            raise ValueError(f"frozen original source changed: {name}")
    bounds = json.loads((destination / "binomial-bounds.json").read_text(encoding="utf-8"))
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
    parser.add_argument(
        "operation",
        choices=[
            "generate",
            "generate-worker",
            "candidate",
            "candidate-worker",
            "reference",
            "analyse",
            "analyse-worker",
            "reanalyse",
        ],
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--namespace", choices=["pilot", "confirmatory"], default="pilot")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--reference-replicates", type=int, default=DEFAULT_REFERENCE_REPLICATES)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--rscript", default="")
    args = parser.parse_args()
    if args.operation == "generate":
        generate(
            args.directory,
            args.replicates,
            namespace=args.namespace,
            reference_replicates=args.reference_replicates,
            workers=args.workers or 1,
        )
    elif args.operation == "generate-worker":
        generate_confirmatory_worker(
            args.directory, args.replicates, args.worker_index, args.worker_count
        )
    elif args.operation == "candidate":
        candidate(args.directory, workers=args.workers)
    elif args.operation == "candidate-worker":
        candidate_confirmatory_worker(args.directory, args.worker_index, args.worker_count)
    elif args.operation == "reference":
        reference(args.directory, rscript=args.rscript, workers=args.workers)
    elif args.operation == "analyse":
        analyse(args.directory, workers=args.workers)
    elif args.operation == "analyse-worker":
        analyse_confirmatory_worker(args.directory, args.worker_index, args.worker_count)
    else:
        reanalyse(args.directory)


if __name__ == "__main__":
    main()
