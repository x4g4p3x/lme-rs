"""Frozen Gaussian random-slope pilot v2 with stricter controls; never confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import lmm_calibration as shared

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "lmm-random-slope-pilot-v2"
ENGINE_COMMIT = "580b270edfcfd8f544818f3ff8171aacbd30d099"
METHODS = ("normal", "satterthwaite", "kenward_roger")
CONTROL = {"tolerance": 1e-12, "max_iterations": 5000, "require_convergence": False}
BOOTSTRAPS = 1000
SOURCES = [
    "scripts/lmm_random_slope.py",
    "scripts/lmm_random_slope_v2.py",
    "scripts/lmm_random_slope_optimum.py",
    "scripts/diagnose_lmm_random_slope_controls.py",
    "scripts/tests/test_lmm_random_slope_v2.py",
    "scripts/tests/test_lmm_random_slope_optimum.py",
    "scripts/report_lmm_random_slope_v2.py",
    "scripts/report_lmm_random_slope.py",
    "scripts/check_lmm_satterthwaite_dense.py",
    "scripts/tests/test_lmm_satterthwaite_dense.py",
    "scripts/lmm_calibration.py",
    "scripts/ci/lme_ci.py",
    "Taskfile.yml",
    "scripts/scientific_validation.py",
    "scripts/tests/test_lmm_random_slope.py",
    "comparisons/scientific/lmm_random_slope_v2.R",
    "comparisons/scientific/restore_r_environment.R",
    "scientific/renv.lock",
    "docs/LMM_RANDOM_SLOPE_PILOT_V2_PROTOCOL.md",
    "docs/LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md",
    "docs/LMM_RANDOM_SLOPE_PILOT_V2_REVIEW.md",
    "scientific/RANDOM_SLOPE_NUMERICAL_CONTROLS.md",
    "scientific/RANDOM_SLOPE_NUMERICAL_CONTROLS.json",
]
GATES = {
    "beta": [1e-5, 1e-4],
    "variance": [1e-6, 1e-3],
    "criterion": [1e-6, 1e-8],
    "se": [1e-6, 1e-3],
    "df": [0.05, 0.005],
    "f": [1e-6, 0.002],
    "p": [1e-4, 0],
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def registry():
    result = []
    factors = [
        ("interior", g, [m] * g, v, r, b)
        for g, m, v, r, b in itertools.product(
            [8, 20, 50], [5, 15], [0.05, 0.25], [-0.5, 0, 0.5], [0, 0.5]
        )
    ]
    factors += [
        ("zero_slope", g, [m] * g, 0, None, b)
        for g, m, b in itertools.product([8, 20, 50], [5, 15], [0, 0.5])
    ]
    for stratum, correlations in [
        ("near_correlation", [-0.95, 0.95]),
        ("exact_correlation", [-1, 1]),
    ]:
        factors += [
            (stratum, g, [5] * g, 0.25, r, 0) for g, r in itertools.product([8, 20], correlations)
        ]
    factors += [
        ("imbalanced", g, [3, 5, 10] * (g // 3) + [3, 5, 10][: g % 3], 0.25, 0, 0) for g in [8, 20]
    ]
    for stratum, g, sizes, v, r, b in factors:
        size_id = str(sizes[0]) if len(set(sizes)) == 1 else "cycle3-5-10"
        r_id = "undefined" if r is None else f"{r:g}"
        covariance = 0 if r is None else r * math.sqrt(0.5 * v)
        result.append(
            {
                "id": f"{stratum}-G{g}-m{size_id}-v{v:g}-r{r_id}-beta{b:g}",
                "stratum": stratum,
                "groups": g,
                "sizes": sizes,
                "v1": v,
                "correlation": r,
                "beta": b,
                "primary_future": stratum == "interior" and b == 0,
                "sigma": [[0.5, covariance], [covariance, v]],
                "x": [x for n in sizes for x in shared.scaled(n)],
                "group": [i for i, n in enumerate(sizes) for _ in range(n)],
            }
        )
    return result


def response(case, replicate, namespace):
    if namespace not in ("smoke", "pilot", "confirmation", "analysis") or replicate < 0:
        raise ValueError("invalid seed namespace/replicate")
    g, n = case["groups"], len(case["x"])
    draws = shared.normals(f"{PROTOCOL}|{namespace}|{case['id']}|{replicate}", 2 * g + n)
    r = case["correlation"]
    b0 = [math.sqrt(0.5) * z for z in draws[:g]]
    b1 = (
        [0.0] * g
        if r is None
        else [
            math.sqrt(case["v1"]) * (r * a + math.sqrt(1 - r * r) * b)
            for a, b in zip(draws[:g], draws[g : 2 * g])
        ]
    )
    return [
        2 + case["beta"] * x + b0[i] + b1[i] * x + e
        for x, i, e in zip(case["x"], case["group"], draws[2 * g :])
    ]


def smoke_cases():
    cases = registry()
    selected = [
        next(c for c in cases if c["stratum"] == s)
        for s in ("interior", "zero_slope", "near_correlation", "exact_correlation", "imbalanced")
    ]
    selected.append(
        next(
            c
            for c in cases
            if c["groups"] == 50
            and c["sizes"][0] == 15
            and c["v1"] == 0.25
            and c["correlation"] == 0.5
            and c["beta"] == 0
        )
    )
    return selected


def inverse2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if det <= 0:
        raise ValueError("nonpositive oracle information determinant")
    return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]]


def block_gls(x, y, sizes, sigma):
    """Independent group-coefficient GLS; accepts Fraction values for exact tests."""
    if len(x) != len(y) or sum(sizes) != len(x) or any(n < 2 for n in sizes):
        raise ValueError("inconsistent oracle design")
    information = [[0, 0], [0, 0]]
    score = [0, 0]
    start = 0
    for n in sizes:
        xx, yy = x[start : start + n], y[start : start + n]
        if abs(sum(xx)) > 1e-10:
            raise ValueError("oracle requires group-centered predictor")
        sxx = sum(t * t for t in xx)
        if sxx <= 0:
            raise ValueError("constant group predictor")
        one = sxx / sxx
        c = [sum(yy) / n, sum(a * b for a, b in zip(xx, yy)) / sxx]
        vi = inverse2(
            [[sigma[0][0] + one / n, sigma[0][1]], [sigma[1][0], sigma[1][1] + one / sxx]]
        )
        for i in range(2):
            score[i] += sum(vi[i][j] * c[j] for j in range(2))
            for j in range(2):
                information[i][j] += vi[i][j]
        start += n
    covariance = inverse2(information)
    beta = [sum(covariance[i][j] * score[j] for j in range(2)) for i in range(2)]
    return beta, covariance


def covariance_diagnostic(matrix, sigma2):
    unknown = {"state": "invalid", "rank": None, "correlation": None}
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        return unknown
    values = [sigma2] + [v for row in matrix for v in row]
    if (
        len(matrix) != 2
        or any(len(row) != 2 for row in matrix)
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        or sigma2 <= 0
    ):
        return unknown
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    scale = max(a + d, sigma2)
    if abs(b - c) > 1e-12 * scale:
        return unknown
    half = (a + d) / 2
    radius = math.hypot((a - d) / 2, (b + c) / 2)
    eigenvalues = [half - radius, half + radius]
    raw_correlation = b / math.sqrt(a * d) if a > 0 and d > 0 else None
    rank = sum(v > 1e-8 * scale for v in eigenvalues)
    invalid = min(eigenvalues) < -1e-12 * scale
    return {
        "state": "invalid" if invalid else ("singular" if rank < 2 else "full_rank"),
        "rank": None if invalid else rank,
        "eigenvalues": eigenvalues,
        "scale": scale,
        "raw_correlation": raw_correlation,
        "correlation": raw_correlation if not invalid and min(a, d) > 1e-8 * scale else None,
    }


def candidate_covariance(theta, sigma2):
    if len(theta) != 3:
        raise ValueError("expected intercept/slope lower-column theta")
    a, b, c = theta
    return [[sigma2 * a * a, sigma2 * a * b], [sigma2 * a * b, sigma2 * (b * b + c * c)]]


def verified_enrichment(original, paired):
    """Keep original candidate numbers; R may add quantiles or a distribution error."""

    def equal(a, b):
        if isinstance(a, bool) or isinstance(b, bool):
            return type(a) is type(b) and a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-300)
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    original_fit = {k: v for k, v in original.items() if k != "methods"}
    paired_fit = {k: v for k, v in paired.items() if k != "methods"}
    if not equal(original_fit, paired_fit):
        raise ValueError("paired candidate fit was modified")
    methods = paired["methods"] or {}
    if methods.keys() != original["methods"].keys():
        raise ValueError("paired candidate method inventory was modified")
    result = {**original, "methods": {}}
    for name, raw in original["methods"].items():
        enriched = methods[name]
        if "error" in enriched and "error" not in raw:
            if not equal(raw, enriched.get("raw")):
                raise ValueError("distribution failure changed original method")
            result["methods"][name] = enriched
        else:
            source = {k: v for k, v in enriched.items() if k not in ("critical", "distribution_p")}
            if not equal(raw, source):
                raise ValueError("paired candidate method was modified")
            result["methods"][name] = {
                **raw,
                **{k: enriched[k] for k in ("critical", "distribution_p") if k in enriched},
            }
    return result


def cp(k, n):
    """Exact binomial-tail inversion; no R/SciPy interval implementation reused."""
    if not 0 <= k <= n:
        raise ValueError("invalid binomial counts")
    if n == 0:
        return [0.0, 1.0]

    def solve(tail, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo + hi) / 2
            value = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in tail)
            if (value < 0.025) == increasing:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return [
        0.0 if k == 0 else solve(range(k, n + 1), True),
        1.0 if k == n else solve(range(k + 1), False),
    ]


def binary(k, valid, attempted):
    if not 0 <= k <= valid <= attempted:
        raise ValueError("invalid failure denominator")
    p = k / valid if valid else None
    return {
        "count": k,
        "evaluable": valid,
        "attempted": attempted,
        "rate": p,
        "pointwise95": cp(k, valid),
        "mcse": math.sqrt(p * (1 - p) / valid) if valid else None,
        "failure_envelope95": [cp(k, attempted)[0], cp(k + attempted - valid, attempted)[1]],
    }


def engine_environment():
    import polars
    from lme_python import lme_python as native

    if not Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("candidate is not installed in an isolated environment")
    return {
        "version": importlib.metadata.version("lme-python"),
        "native_sha256": shared.digest(Path(native.__file__)),
        "python": platform.python_version(),
        "polars": polars.__version__,
        "platform": platform.platform(),
    }


def freeze(destination, candidate_python, wheel):
    # The candidate source must be exactly the already reviewed engine commit.
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        != ENGINE_COMMIT
    ):
        raise ValueError("engine commit changed; review and declare a new candidate")
    engine_paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "python/src",
            "Cargo.toml",
            "Cargo.lock",
            "python/Cargo.toml",
            "python/Cargo.lock",
            "python/pyproject.toml",
            "build.rs",
            "python/build.rs",
            ".cargo",
            "python/.cargo",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if subprocess.check_output(["git", "diff", "HEAD", "--", *engine_paths], cwd=ROOT):
        raise ValueError("candidate engine sources changed after the recorded commit")
    environment = json.loads(
        subprocess.check_output(
            [str(candidate_python), str(Path(__file__)), "environment"],
            cwd=ROOT,
            env=shared.worker_env(),
            text=True,
        )
    )
    if environment["version"] != "0.2.8.dev0":
        raise ValueError("unexpected candidate package version")
    with zipfile.ZipFile(wheel) as archive:
        native_entries = [name for name in archive.namelist() if name.endswith((".pyd", ".so"))]
        if (
            len(native_entries) != 1
            or hashlib.sha256(archive.read(native_entries[0])).hexdigest()
            != environment["native_sha256"]
        ):
            raise ValueError("frozen wheel native does not match installed candidate")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCES + engine_paths:
        target = destination / "frozen-sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        hashes[name] = shared.digest(target)
    shutil.copyfile(wheel, destination / wheel.name)
    packages = json.loads(
        subprocess.check_output(
            ["uv", "pip", "list", "--format", "json", "--python", str(candidate_python)],
            cwd=ROOT,
            text=True,
        )
    )
    lock = "# Install the separately hashed candidate wheel with --no-deps.\n" + "".join(
        f"{p['name']}=={p['version']}\n"
        for p in sorted(packages, key=lambda p: p["name"])
        if p["name"].replace("_", "-") != "lme-python"
    )
    (destination / "requirements.txt").write_text(lock, encoding="utf-8")
    shared.write_json(
        destination / "freeze.json",
        {
            "protocol": PROTOCOL,
            "engine_commit": ENGINE_COMMIT,
            "environment": environment,
            "sources": hashes,
            "wheel": {"name": wheel.name, "sha256": shared.digest(wheel)},
            "requirements_sha256": shared.digest(destination / "requirements.txt"),
            "registry_sha256": shared.json_digest(registry()),
            "control": CONTROL,
            "gates": GATES,
            "bootstrap_repetitions": BOOTSTRAPS,
            "thread_limits": shared.THREAD_LIMITS,
            "maximum_workers": 4,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def verify_freeze(directory):
    frozen = load(directory / "freeze.json")
    if frozen["protocol"] != PROTOCOL or frozen["control"] != CONTROL or frozen["gates"] != GATES:
        raise ValueError("frozen settings differ")
    if frozen["registry_sha256"] != shared.json_digest(registry()):
        raise ValueError("registry changed after freeze")
    for name, expected in frozen["sources"].items():
        if shared.digest(directory / "frozen-sources" / name) != expected:
            raise ValueError(f"archived source changed: {name}")
        if name in SOURCES and shared.digest(ROOT / name) != expected:
            raise ValueError(f"active source changed after freeze: {name}")
    if shared.digest(directory / frozen["wheel"]["name"]) != frozen["wheel"]["sha256"]:
        raise ValueError("wheel changed")
    if shared.digest(directory / "requirements.txt") != frozen["requirements_sha256"]:
        raise ValueError("dependency freeze changed")
    return frozen


def generate(destination, frozen_dir, namespace):
    frozen = verify_freeze(frozen_dir)
    if namespace not in ("smoke", "pilot"):
        raise ValueError("this runner cannot launch confirmation")
    cases = registry()
    if namespace == "smoke":
        cases = smoke_cases()
    replicates = 20 if namespace == "pilot" else 2
    destination.mkdir(parents=True, exist_ok=False)
    shared.write_json(destination / "designs.json", cases)
    hashes = {}
    for case in cases:
        name = f"cases/{case['id']}/inputs.jsonl.gz"
        shared.write_gzip_jsonl(
            destination / name,
            (
                {"case": case["id"], "replicate": i, "y": response(case, i, namespace)}
                for i in range(replicates)
            ),
        )
        hashes[name] = shared.digest(destination / name)
    shared.write_json(
        destination / "manifest.json",
        {
            "protocol": PROTOCOL,
            "namespace": namespace,
            "replicates": replicates,
            "attempted": len(cases) * replicates,
            "designs_sha256": shared.digest(destination / "designs.json"),
            "freeze_sha256": shared.digest(frozen_dir / "freeze.json"),
            "freeze_canonical_sha256": shared.json_digest(frozen),
            "freeze": frozen,
            "inputs": hashes,
        },
    )


def verify_run(destination):
    manifest = load(destination / "manifest.json")
    if manifest["protocol"] != PROTOCOL or manifest["namespace"] not in ("smoke", "pilot"):
        raise ValueError("unsupported run manifest")
    if manifest["freeze"]["control"] != CONTROL or manifest["freeze"]["gates"] != GATES:
        raise ValueError("run controls differ from active harness")
    if shared.json_digest(manifest["freeze"]) != manifest["freeze_canonical_sha256"]:
        raise ValueError("embedded freeze digest differs")
    if shared.digest(destination / "designs.json") != manifest["designs_sha256"]:
        raise ValueError("design hash differs")
    expected = registry()
    if manifest["namespace"] == "smoke":
        expected = smoke_cases()
    if (
        load(destination / "designs.json") != expected
        or manifest["replicates"] != (20 if manifest["namespace"] == "pilot" else 2)
        or manifest["attempted"] != len(expected) * manifest["replicates"]
    ):
        raise ValueError("run registry/budget differs from frozen protocol")
    if set(manifest["inputs"]) != {f"cases/{c['id']}/inputs.jsonl.gz" for c in expected}:
        raise ValueError("input inventory differs")
    for name, expected in manifest["inputs"].items():
        if shared.digest(destination / name) != expected:
            raise ValueError("input hash differs")
    for name in SOURCES:
        if shared.digest(ROOT / name) != manifest["freeze"]["sources"][name]:
            raise ValueError(f"active source differs from freeze: {name}")
    return manifest


def ordered(path, case, n, compressed=True):
    iterator = shared.gzip_lines(path) if compressed else shared.lines(path)
    rows = []
    for i, row in enumerate(iterator):
        if (
            i >= n
            or row.get("case") != case
            or type(row.get("replicate")) is not int
            or row["replicate"] != i
        ):
            raise ValueError("extra/duplicate/misaligned record")
        rows.append(row)
    if len(rows) != n:
        raise ValueError("missing records")
    return rows


def candidate_fit(lme, data):
    record = {"methods": {}, "converged": False}
    before = time.perf_counter()
    try:
        fit = lme.lmer(
            "y ~ x + (1 + x | group)", data, reml=True, control=lme.FitControl(**CONTROL)
        )
        matrix = candidate_covariance(fit.theta, fit.sigma2)
        record.update(
            beta=fit.coefficients[1],
            sigma2=fit.sigma2,
            covariance=matrix,
            criterion=fit.reml_criterion,
            theta=fit.theta,
            converged=fit.converged is True,
            diagnostics=fit.diagnostics,
            covariance_diagnostic=covariance_diagnostic(matrix, fit.sigma2),
        )
        if record["converged"]:
            record["methods"] = shared.evaluate_candidate_methods(fit, data, fit.coefficients[1])
        else:
            record["fit_error"] = "nonconvergence"
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        record["converged"] = False
        record["fit_error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = time.perf_counter() - before
    return shared.finite_json(record)


def candidate_worker(destination, worker, workers):
    import lme_python as lme
    import polars as pl

    manifest = verify_run(destination)
    if engine_environment() != manifest["freeze"]["environment"]:
        raise ValueError("candidate environment differs from frozen build")
    for case in load(destination / "designs.json")[worker::workers]:
        folder = destination / "cases" / case["id"]
        inputs = ordered(folder / "inputs.jsonl.gz", case["id"], manifest["replicates"])
        output = []
        for row in inputs:
            data = pl.DataFrame(
                {"x": case["x"], "y": row["y"], "group": [f"g{i:03}" for i in case["group"]]}
            )
            output.append(
                {
                    "case": case["id"],
                    "replicate": row["replicate"],
                    "candidate": candidate_fit(lme, data),
                }
            )
        shared.write_gzip_jsonl(folder / "candidate.jsonl.gz", output)
        print(f"candidate {worker + 1}/{workers}: {case['id']}", flush=True)


def phase(destination, candidate_python, rscript, workers):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    verify_run(destination)
    for operation in ("candidate", "reference"):
        started = time.perf_counter()
        processes = []
        for i in range(workers):
            cmd = (
                [
                    str(candidate_python),
                    str(Path(__file__)),
                    "worker",
                    str(destination),
                    "--worker",
                    str(i),
                    "--workers",
                    str(workers),
                ]
                if operation == "candidate"
                else [
                    rscript,
                    "--vanilla",
                    "comparisons/scientific/lmm_random_slope_v2.R",
                    str(destination),
                    str(i),
                    str(workers),
                ]
            )
            log = (destination / f"{operation}-worker-{i}.log").open("x", encoding="utf-8")
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=shared.worker_env(), stdout=log, stderr=log
            )
            processes.append((process, log))
        peaks = {str(p.pid): None for p, _ in processes}
        while any(p.poll() is None for p, _ in processes):
            # No peak-memory claim: old Windows counters sampled launcher shims.
            time.sleep(0.25)
        codes = []
        for process, log in processes:
            codes.append(process.wait())
            log.close()
        shared.write_json(
            destination / f"{operation}-execution.json",
            {
                "workers": workers,
                "seconds": time.perf_counter() - started,
                "exit_codes": codes,
                "thread_limits": shared.THREAD_LIMITS,
                "sampled_process_peak_working_set_bytes": peaks,
                "memory_scope": "not measured; Windows launcher counters deliberately excluded",
            },
        )
        if any(codes):
            raise RuntimeError(f"{operation} workers failed; all logs/attempts retained: {codes}")
        print(f"{operation} complete: {time.perf_counter() - started:.1f}s", flush=True)


def process_peak_memory(process):
    """Read-only OS peak counter; unavailable is explicit, never reported as zero."""
    if sys.platform != "win32" or process.poll() is not None:
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    if query(wintypes.HANDLE(int(process._handle)), ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize
    return None


def parity(candidate, reference):
    checks = []
    scale = max(
        1,
        candidate["sigma2"],
        reference["sigma2"],
        sum(candidate["covariance"][i][i] for i in range(2)),
        sum(reference["covariance"][i][i] for i in range(2)),
    )

    def check(label, actual, expected, gate):
        atol, rtol = GATES[gate]
        limit = atol * (scale if gate == "variance" else 1) + rtol * abs(expected)
        checks.append(
            {
                "metric": label,
                "difference": actual - expected,
                "limit": limit,
                "passed": abs(actual - expected) <= limit,
            }
        )

    for metric in ("beta", "criterion", "sigma2"):
        check(
            metric,
            candidate[metric],
            reference[metric],
            "variance" if metric == "sigma2" else metric,
        )
    for i, j in ((0, 0), (0, 1), (1, 1)):
        check(
            f"covariance_{i}{j}",
            candidate["covariance"][i][j],
            reference["covariance"][i][j],
            "variance",
        )
    for method in METHODS:
        a, b = candidate["methods"].get(method, {}), reference["methods"].get(method, {})
        if "error" in a or "error" in b or not a or not b:
            continue
        for metric in ("se", "f", "p") + (() if method == "normal" else ("df",)):
            check(f"{method}_{metric}", a[metric], b[metric], metric)
    return checks


def balanced_checks(case, y, fit):
    """Prospective constrained optimum/precision/rank checks, never substitute fits."""
    from lmm_random_slope_optimum import optimum

    if len(set(case["sizes"])) != 1:
        return None
    m = case["sizes"][0]
    x = case["x"][:m]
    if any(case["x"][i : i + m] != x for i in range(0, len(case["x"]), m)):
        raise ValueError("oracle requires common group predictor")
    expected = optimum(x, [y[i : i + m] for i in range(0, len(y), m)])
    result = {"oracle": expected, "evaluable": False, "checks": [], "rank_match": None}
    if fit.get("converged") is not True or not all(
        k in fit for k in ("covariance", "sigma2", "criterion")
    ):
        return result
    if not isinstance(fit["criterion"], (int, float)) or not math.isfinite(fit["criterion"]):
        return result
    diagnostic = covariance_diagnostic(fit["covariance"], fit["sigma2"])
    if diagnostic["state"] == "invalid":
        return result
    scale = max(
        1,
        fit["sigma2"],
        expected["sigma2"],
        sum(fit["covariance"][i][i] for i in range(2)),
        sum(expected["covariance"][i][i] for i in range(2)),
    )
    values = [
        ("criterion", fit["criterion"], expected["criterion"], "criterion"),
        ("sigma2", fit["sigma2"], expected["sigma2"], "variance"),
    ]
    values += [
        (f"covariance_{i}{j}", fit["covariance"][i][j], expected["covariance"][i][j], "variance")
        for i, j in ((0, 0), (0, 1), (1, 1))
    ]
    for label, actual, wanted, gate in values:
        atol, rtol = GATES[gate]
        limit = atol * (scale if gate == "variance" else 1) + rtol * abs(wanted)
        result["checks"].append(
            {
                "metric": label,
                "difference": actual - wanted,
                "limit": limit,
                "passed": math.isfinite(actual) and abs(actual - wanted) <= limit,
            }
        )
    result["evaluable"] = True
    result["rank_match"] = (
        diagnostic["rank"]
        == covariance_diagnostic(expected["covariance"], expected["sigma2"])["rank"]
    )
    return result


def analyse(destination):
    manifest = verify_run(destination)
    n = manifest["replicates"]
    summaries, paired, fit_rows, artifacts = [], [], [], {}
    balanced = []
    for case in load(destination / "designs.json"):
        folder = destination / "cases" / case["id"]
        inputs = ordered(folder / "inputs.jsonl.gz", case["id"], n)
        candidate = ordered(folder / "candidate.jsonl.gz", case["id"], n)
        enriched = ordered(folder / "paired.jsonl", case["id"], n, False)
        valid = {f"{engine}:{m}": {} for engine in ("candidate", "reference") for m in METHODS}
        valid["control:known_variance"] = {}
        comparisons, binary_disagreements = [], Counter()
        fit_states = {e: Counter() for e in ("candidate", "reference")}
        failures = {label: Counter() for label in valid}
        for i, (raw, cand, pair) in enumerate(zip(inputs, candidate, enriched)):
            candidate_line = json.dumps(cand, allow_nan=False, separators=(",", ":"))
            if pair["candidate_line_sha256"] != hashlib.sha256(candidate_line.encode()).hexdigest():
                raise ValueError("paired candidate does not match retained original")
            pair["candidate"] = verified_enrichment(cand["candidate"], pair["candidate"])
            if len(set(case["sizes"])) == 1:
                balanced.append(
                    {
                        "case": case["id"],
                        "replicate": i,
                        "engines": {
                            engine: balanced_checks(case, raw["y"], pair[engine])
                            for engine in ("candidate", "reference")
                        },
                    }
                )
            control_beta, control_cov = block_gls(case["x"], raw["y"], case["sizes"], case["sigma"])
            se = math.sqrt(control_cov[1][1])
            control = {
                "beta": control_beta[1],
                "se": se,
                "df": None,
                "p": math.erfc(abs(control_beta[1] / se) / math.sqrt(2)),
                "critical": statistics.NormalDist().inv_cdf(0.975),
            }
            valid["control:known_variance"][i] = shared.outcome(control, case["beta"])
            for engine in ("candidate", "reference"):
                fit = pair[engine]
                status = "converged" if fit.get("converged") is True else "failed"
                fit_states[engine][status] += 1
                if status == "failed":
                    fit_states[engine][
                        "nonconvergence"
                        if fit.get("fit_error") == "nonconvergence"
                        else "exception"
                    ] += 1
                diagnostic = (
                    covariance_diagnostic(fit["covariance"], fit.get("sigma2"))
                    if "covariance" in fit
                    else {"state": "unknown"}
                )
                fit_states[engine][diagnostic["state"]] += 1
                for method in METHODS:
                    if status != "converged" or diagnostic["state"] in ("invalid", "unknown"):
                        failures[f"{engine}:{method}"][
                            "fit_failure" if status != "converged" else "invalid_covariance"
                        ] += 1
                        continue
                    try:
                        valid[f"{engine}:{method}"][i] = shared.outcome(
                            fit["methods"][method], case["beta"]
                        )
                    except (ValueError, KeyError, TypeError) as error:
                        failures[f"{engine}:{method}"][str(error)] += 1
            a, b = pair["candidate"], pair["reference"]
            if all(
                f.get("converged") is True
                and covariance_diagnostic(f.get("covariance"), f.get("sigma2"))["state"]
                != "invalid"
                and all(
                    isinstance(f.get(k), (int, float)) and math.isfinite(f[k])
                    for k in ("beta", "criterion", "sigma2")
                )
                for f in (a, b)
            ):
                comparisons.append({"replicate": i, "checks": parity(a, b)})
            for method in METHODS:
                ca, re = valid[f"candidate:{method}"].get(i), valid[f"reference:{method}"].get(i)
                if ca is not None and re is not None:
                    binary_disagreements[f"{method}:evaluable"] += 1
                    for metric in ("reject", "cover"):
                        binary_disagreements[f"{method}:{metric}"] += ca[metric] != re[metric]
        seed = int(hashlib.sha256(f"{PROTOCOL}|analysis|{case['id']}".encode()).hexdigest(), 16)
        rng = random.Random(seed)
        resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAPS)]
        mcse = shared.fast_bootstrap_mcse(valid, list(valid), n, resamples)
        for label, observed in valid.items():
            rows = list(observed.values())
            summaries.append(
                {
                    "case": case["id"],
                    "method": label,
                    "attempted": n,
                    "failed": n - len(rows),
                    "status": "exploratory_pilot_only",
                    "failure_reasons": dict(failures[label]),
                    "rejection": binary(sum(v["reject"] for v in rows), len(rows), n),
                    "coverage": binary(sum(v["cover"] for v in rows), len(rows), n),
                    "failure_rate95": cp(n - len(rows), n),
                    "performance": shared.performance(rows),
                    "performance_mcse": mcse[label],
                }
            )
        fit_rows.append({"case": case["id"], "engines": fit_states})
        reference_environment = load(folder / "reference-environment.json")
        for k, interval in enumerate(reference_environment["cp95"]):
            if any(abs(a - b) > 1e-12 for a, b in zip(cp(k, n), interval)):
                raise ValueError("independent exact-binomial calculation differs from R")
        paired.append(
            {"case": case["id"], "binary": dict(binary_disagreements), "numeric": comparisons}
        )
        for name in (
            "inputs.jsonl.gz",
            "candidate.jsonl.gz",
            "paired.jsonl",
            "reference-environment.json",
        ):
            artifacts[f"cases/{case['id']}/{name}"] = shared.digest(folder / name)
    shared.write_json(
        destination / "summary.json",
        {
            "protocol": PROTOCOL,
            "namespace": manifest["namespace"],
            "attempted": manifest["attempted"],
            "engine_commit": ENGINE_COMMIT,
            "candidate": manifest["freeze"]["environment"],
            "manifest_sha256": shared.digest(destination / "manifest.json"),
            "bootstrap_repetitions": BOOTSTRAPS,
            "summaries": summaries,
            "paired": paired,
            "fit_diagnostics": fit_rows,
            "balanced_checks": balanced,
            "artifacts": artifacts,
            "scope": "exploratory harness/cost pilot; no confirmatory calibration decision",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["environment", "freeze", "generate", "worker", "run", "analyse"]
    )
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--namespace", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    if args.operation == "environment":
        print(json.dumps(engine_environment()))
    elif args.operation == "freeze":
        freeze(args.directory, args.candidate_python, args.wheel)
    elif args.operation == "generate":
        generate(args.directory, args.freeze, args.namespace)
    elif args.operation == "worker":
        candidate_worker(args.directory, args.worker, args.workers)
    elif args.operation == "run":
        phase(args.directory, args.candidate_python, args.rscript, args.workers)
    else:
        analyse(args.directory)


if __name__ == "__main__":
    main()
