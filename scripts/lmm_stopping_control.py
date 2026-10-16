"""Separate fresh-control pilot for the unchanged, exactly identified repaired engine."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import lmm_random_slope_scaling as base
from lmm_resource_caps import CappedGroup, Caps, disk_sample, monitor, violation
from lmm_stream_store import canonical, sha

ROOT, VARIANTS = base.ROOT, base.VARIANTS
case_metadata, peak_working_set, write_new = (
    base.case_metadata,
    base.peak_working_set,
    base.write_new,
)
PROTOCOL = "lmm-random-slope-stopping-control-pilot-v1"
ANCHOR = ROOT / "scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_FREEZE.json"
CONTROL = {"tolerance": 1e-10, "max_iterations": 5000, "require_convergence": False}
ADDITIONS = [
    "scripts/lmm_stopping_control.py",
    "scripts/report_lmm_stopping_control.py",
    "scripts/complete_lmm_stopping_control.py",
    "scripts/tests/test_lmm_stopping_control.py",
    "docs/LMM_RANDOM_SLOPE_STOPPING_CONTROL_PROTOCOL.md",
    "scientific/RANDOM_SLOPE_STOPPING_DIAGNOSTIC.json",
    "scientific/RANDOM_SLOPE_STOPPING_DIAGNOSTIC_PROVENANCE.json",
    "scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_FREEZE.json",
]


def engine_inputs():
    _, names = base.old.engine_inventory()
    return {name: base.old.pilot.shared.digest(ROOT / name) for name in names}


def historical_runs():
    """Snapshot prior namespaces before a study exists; no input generation."""
    result = []
    for path in sorted((ROOT / "benchmark-results/scientific").rglob("manifest.json")):
        raw = base.old.pilot.load(path)
        if isinstance(raw.get("namespace"), str):
            result.append(
                {"namespace": raw["namespace"], "sha256": base.old.pilot.shared.digest(path)}
            )
    return result


def validate(manifest):
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("mode") != "rehearsal"
        or manifest.get("scenario") != "stopping_control_pilot"
    ):
        raise ValueError("confirmation disabled")
    if len(manifest["namespace"]) > 64 or not re.fullmatch(
        r"stopping-control-pilot-v1-[a-z0-9-]{1,40}", manifest["namespace"]
    ):
        raise ValueError("fresh stopping-control namespace required")
    history = manifest["historical_runs"]
    if not history or any(
        not isinstance(row.get("namespace"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
        for row in history
    ):
        raise ValueError("missing historical namespace identities")
    if manifest["namespace"] in {row["namespace"] for row in history}:
        raise ValueError("seed namespace was already used")
    for name, count in (
        ("primary_n", 20),
        ("secondary_n", 20),
        ("reference_n", 20),
        ("workers", 4),
        ("chunk_size", 8),
    ):
        if type(manifest.get(name)) is not int or manifest[name] != count:
            raise ValueError("fixed small pilot inventory differs")
    if (
        manifest["cases"] != base.old.pilot.registry()
        or manifest["variants"] != VARIANTS
        or manifest["caps"] != Caps().record()
        or manifest["control"] != CONTROL
    ):
        raise ValueError("design/reference/cap/control policy differs")
    anchor = base.old.pilot.load(ANCHOR)
    if (
        manifest["candidate"] != anchor["candidate"]
        or manifest["wheel_sha256"] != anchor["wheel_sha256"]
        or manifest["reuse_policy"] != "exact-published-repair-engine-v1"
        or not any(
            row["namespace"] == anchor["namespace"]
            and row["sha256"] == base.old.pilot.shared.digest(ANCHOR)
            for row in history
        )
    ):
        raise ValueError("exact published repaired engine/environment reuse required")
    revision = manifest["engine_revision"]
    if revision != anchor["engine_revision"]:
        raise ValueError("original engine build identity changed")
    study = manifest["study_revision"]
    if type(study["working_tree"]) is not bool or not re.fullmatch(
        r"[0-9a-f]{40}", study["base_commit"]
    ):
        raise ValueError("invalid actual study base/worktree identity")
    if revision["sources"] != {
        name: manifest["sources"][name] for name in manifest["engine_sources"]
    }:
        raise ValueError("wheel-build engine sources differ from study sources")


def verify(folder):
    manifest = base.old.pilot.load(folder / "manifest.json")
    validate(manifest)
    anchor = base.old.pilot.load(ANCHOR)
    expected = set(anchor["sources"]) | set(ADDITIONS)
    if manifest["predecessor_manifest_sha256"] != base.old.pilot.shared.digest(ANCHOR):
        raise ValueError("published predecessor identity changed")
    if (
        set(manifest["sources"]) != expected
        or manifest["engine_sources"] != anchor["engine_sources"]
    ):
        raise ValueError("frozen source inventory differs")
    build_file = folder / "engine-build-inputs.json"
    if (
        base.old.pilot.shared.digest(build_file) != manifest["build_inputs_sha256"]
        or base.old.pilot.load(build_file) != manifest["engine_revision"]
    ):
        raise ValueError("pre-build receipt changed")
    for name, digest in manifest["sources"].items():
        if (
            base.old.pilot.shared.digest(ROOT / name) != digest
            or base.old.pilot.shared.digest(folder / "sources" / name) != digest
        ):
            raise ValueError("frozen source changed: " + name)
    return manifest


def freeze(folder, anchor, candidate_python, wheel, build_inputs, namespace):
    if folder.resolve().is_relative_to(anchor.resolve()) or anchor.resolve().is_relative_to(
        folder.resolve()
    ):
        raise ValueError("study output must not overlap predecessor")
    if folder.exists():
        raise FileExistsError(folder)
    previous = base.old.pilot.load(anchor / "manifest.json")
    if base.old.pilot.shared.digest(anchor / "manifest.json") != base.old.pilot.shared.digest(
        ANCHOR
    ):
        raise ValueError("predecessor manifest differs from published evidence")
    for name, digest in previous["sources"].items():
        if base.old.pilot.shared.digest(anchor / "sources" / name) != digest:
            raise ValueError("historical archived source changed")
        if (
            name not in previous["engine_sources"]
            and base.old.pilot.shared.digest(ROOT / name) != digest
        ):
            raise ValueError("immutable historical scientific source changed")
    revision = base.old.pilot.load(build_inputs)
    if (
        revision != previous["engine_revision"]
        or revision["sources"] != engine_inputs()
        or base.old.pilot.shared.digest(build_inputs) != previous["build_inputs_sha256"]
    ):
        raise ValueError("engine changed or original pre-build receipt differs")
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);import lmm_random_slope_v2 as p;"
        "print(json.dumps(p.engine_environment()))"
    )
    environment = json.loads(
        subprocess.check_output(
            [str(candidate_python), "-c", code, str(ROOT / "scripts")],
            cwd=ROOT,
            text=True,
            env=dict(os.environ, **base.old.pilot.shared.THREAD_LIMITS),
        )
    )
    with zipfile.ZipFile(wheel) as archive:
        names = [p for p in archive.namelist() if p.endswith((".pyd", ".so"))]
        if len(names) != 1 or sha(archive.read(names[0])) != environment["native_sha256"]:
            raise ValueError("wheel/native installed bytes differ")
    if (
        environment != previous["candidate"]
        or base.old.pilot.shared.digest(wheel) != previous["wheel_sha256"]
    ):
        raise ValueError("exact published native/wheel/environment required")
    manifest = copy.deepcopy(previous)
    manifest.update(
        protocol=PROTOCOL,
        scenario="stopping_control_pilot",
        namespace=namespace,
        control=CONTROL,
        reuse_policy="exact-published-repair-engine-v1",
        historical_runs=historical_runs(),
        study_revision={
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
            "scope": "current study sources; original engine build identity retained separately",
        },
        candidate=environment,
        engine_revision=revision,
        wheel_sha256=base.old.pilot.shared.digest(wheel),
        predecessor_manifest_sha256=base.old.pilot.shared.digest(ANCHOR),
        build_inputs_sha256=base.old.pilot.shared.digest(build_inputs),
    )
    manifest["sources"] = {
        name: base.old.pilot.shared.digest(ROOT / name)
        for name in sorted(set(previous["sources"]) | set(ADDITIONS))
    }
    validate(manifest)
    folder.mkdir(parents=True)
    for name in manifest["sources"]:
        target = folder / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    with (folder / "engine-build-inputs.json").open("xb") as target:
        target.write(build_inputs.read_bytes())
    write_new(folder / "manifest.json", manifest)
    verify(folder)


def candidate_fit(lme, data):
    """Explicit prospective control; historical candidate helpers stay immutable."""
    pilot = base.old.pilot
    record, before = {"methods": {}, "converged": False}, time.perf_counter()
    try:
        fit = lme.lmer(
            "y ~ x + (1 + x | group)", data, reml=True, control=lme.FitControl(**CONTROL)
        )
        matrix = pilot.candidate_covariance(fit.theta, fit.sigma2)
        record.update(
            beta=fit.coefficients[1],
            sigma2=fit.sigma2,
            covariance=matrix,
            criterion=fit.reml_criterion,
            theta=fit.theta,
            converged=fit.converged is True,
            diagnostics=fit.diagnostics,
            covariance_diagnostic=pilot.covariance_diagnostic(matrix, fit.sigma2),
        )
        if record["converged"]:
            record["methods"] = pilot.shared.evaluate_candidate_methods(
                fit, data, fit.coefficients[1]
            )
        else:
            record["fit_error"] = "nonconvergence"
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        record["converged"] = False
        record["fit_error"] = f"{type(error).__name__}: {error}"
    record["seconds"] = time.perf_counter() - before
    return pilot.shared.finite_json(record)


def worker(folder, phase, index, receipt, rscript=None):
    manifest = verify(folder)
    if (
        type(index) is not int
        or not 0 <= index < 4
        or not re.fullmatch(r"(?:candidate|reference)-[0-9]+", receipt)
    ):
        raise ValueError("invalid worker identity")
    timings = dict.fromkeys(
        (
            "start_commit_seconds",
            "outcome_commit_seconds",
            "fit_callback_seconds",
            "input_generation_commit_seconds",
            "validation_recovery_seconds",
        ),
        0.0,
    )
    before, session = time.perf_counter(), None
    try:
        if phase == "candidate":
            import lme_python as lme
            import polars as pl

            if base.old.pilot.engine_environment() != manifest["candidate"]:
                raise ValueError("loaded native differs")

            def fit(case, row):
                data = pl.DataFrame(
                    {"x": case["x"], "y": row["y"], "group": [f"g{i:03}" for i in case["group"]]}
                )
                return candidate_fit(lme, data)

            base.run_cases(folder, manifest, phase, index, fit, timings)
            r_env, r_peak = None, None
        else:
            with (folder / "executions" / f"{receipt}-r-{index}.log").open(
                "x", encoding="utf-8"
            ) as log:
                session = base.old.RSession(rscript, log)
                base.run_cases(folder, manifest, phase, index, session.fit, timings)
                r_env = session.request({"command": "environment"})
                r_peak = peak_working_set(r_env["native_pid"])
        write_new(
            folder / "executions" / f"{receipt}-worker-{index}.json",
            {
                "phase": phase,
                "worker": index,
                "manifest_sha": sha(canonical(manifest)),
                "native_python_pid": os.getpid(),
                "python_peak_working_set_bytes": peak_working_set(),
                "r_environment": r_env,
                "r_peak_working_set_bytes": r_peak,
                "timings": timings,
                "worker_seconds": time.perf_counter() - before,
                "memory_scope": "native resident high-water; not a whole-host simultaneous peak",
            },
        )
    finally:
        if session is not None:
            session.close()


def launch(folder, phase, candidate_python, rscript=None):
    manifest = verify(folder)
    if phase not in ("candidate", "reference"):
        raise ValueError("invalid phase")
    caps, receipt = Caps(**manifest["caps"]), f"{phase}-{time.time_ns()}"
    target, logs = folder / "executions", []
    write_new(
        target / f"{receipt}-start.json",
        {
            "manifest_sha": sha(canonical(manifest)),
            "phase": phase,
            "workers": 4,
            "caps": caps.record(),
        },
    )
    group, result = CappedGroup(caps), {"abort_reason": "launch_not_completed"}
    before = time.perf_counter()
    try:
        initial = disk_sample(folder)
        reason = violation(caps, 0, initial)
        if reason:
            result = {"abort_reason": reason, "preflight_disk": initial}
        else:
            for i in range(4):
                log = (target / f"{receipt}-python-{i}.log").open("x", encoding="utf-8")
                logs.append(log)
                args = [
                    str(candidate_python),
                    str(Path(__file__).resolve()),
                    "worker",
                    str(folder),
                    "--phase",
                    phase,
                    "--index",
                    str(i),
                    "--receipt",
                    receipt,
                ]
                if rscript is not None:
                    args.extend(["--rscript", str(rscript)])
                group.start(
                    args,
                    cwd=ROOT,
                    env=dict(os.environ, **base.old.pilot.shared.THREAD_LIMITS),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            result = monitor(group, caps, folder, phase_started=before)
    finally:
        group.close()
        for process in group.processes:
            process.wait(timeout=10)
        for log in logs:
            log.close()
        write_new(
            target / f"{receipt}-end.json",
            {
                "seconds": time.perf_counter() - before,
                "exit_codes": [p.returncode for p in group.processes],
                "native_root_pids": [p.pid for p in group.processes],
                "caps": caps.record(),
                "resource_guard": result,
                "final_disk": disk_sample(folder),
            },
        )
    if result["abort_reason"] or any(p.returncode != 0 for p in group.processes):
        raise RuntimeError("retained interrupted/failed stopping-control execution: " + receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("freeze")
    init.add_argument("folder", type=Path)
    for name in ("anchor", "candidate-python", "wheel", "build-inputs"):
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument("--namespace", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("folder", type=Path)
    execute.add_argument("--phase", choices=("candidate", "reference"), required=True)
    execute.add_argument("--candidate-python", type=Path, required=True)
    execute.add_argument("--rscript", type=Path)
    child = commands.add_parser("worker")
    child.add_argument("folder", type=Path)
    child.add_argument("--phase", choices=("candidate", "reference"), required=True)
    child.add_argument("--index", type=int, required=True)
    child.add_argument("--receipt", required=True)
    child.add_argument("--rscript", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"freeze": freeze, "run": launch, "worker": worker}[command](**args)
