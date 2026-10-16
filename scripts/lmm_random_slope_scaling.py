"""Separate capped scaling/native-fault rehearsals; no confirmatory campaign."""

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

import lmm_random_slope_stream as old
from lmm_resource_caps import CappedGroup, Caps, disk_sample, monitor, violation
from lmm_stream_store import Journal, canonical, case_lock, sha

ROOT, VARIANTS = old.ROOT, old.VARIANTS
PROTOCOL = "lmm-random-slope-scaling-v1"
write_new, peak_working_set, case_metadata = old.write_new, old.peak_working_set, old.case_metadata
ADDITIONS = [
    "scripts/lmm_resource_caps.py",
    "scripts/lmm_random_slope_scaling.py",
    "scripts/report_lmm_random_slope_scaling.py",
    "scripts/tests/test_lmm_resource_caps.py",
    "scripts/tests/test_lmm_random_slope_scaling.py",
    "scripts/verify_lmm_scaling_controls.py",
    "scripts/complete_lmm_scaling.py",
    "docs/LMM_RANDOM_SLOPE_SCALING_PROTOCOL.md",
]


def validate(manifest):
    if manifest["protocol"] != PROTOCOL or manifest["mode"] != "rehearsal":
        raise ValueError("confirmation disabled")
    fault = manifest["scenario"] == "native_fault"
    role = "native" if fault else "n" + str(manifest["primary_n"])
    if not re.fullmatch(
        r"scaling-v1-[a-z0-9-]{1,30}-" + role + r"-[a-z0-9]+", manifest["namespace"]
    ):
        raise ValueError("scaling seed namespace must identify its distinct budget/scenario")
    for name in ("secondary_n", "reference_n", "chunk_size", "workers"):
        if type(manifest[name]) is not int:
            raise ValueError("rehearsal inventories must use integer counts")
    if manifest["scenario"] not in ("scaling", "native_fault"):
        raise ValueError("invalid rehearsal scenario")
    if manifest["cases"] != (old.pilot.registry()[:1] if fault else old.pilot.registry()):
        raise ValueError("case inventory differs")
    n = manifest["primary_n"]
    if type(n) is not int or n not in ((3,) if fault else (10, 20)) or manifest["secondary_n"] != n:
        raise ValueError("only prespecified small scaling budgets are allowed")
    if manifest["reference_n"] != (0 if fault else n) or manifest["chunk_size"] != 8:
        raise ValueError("reference/chunk budget differs")
    if manifest["variants"] != VARIANTS or manifest["workers"] != (1 if fault else 4):
        raise ValueError("reference bank or worker inventory differs")
    if manifest["caps"] != Caps().record():
        raise ValueError("prospective scientific caps differ")


def verify(folder):
    manifest = old.pilot.load(folder / "manifest.json")
    validate(manifest)
    anchor, engine_names = old.engine_inventory()
    if (
        manifest["engine_sources"] != engine_names
        or manifest["candidate"] != anchor["environment"]
        or manifest["engine_commit"] != old.pilot.ENGINE_COMMIT
        or manifest["control"] != old.pilot.CONTROL
    ):
        raise ValueError("reviewed candidate identity differs")
    expected = set(old.pilot.SOURCES + old.NEW_SOURCES + engine_names + ADDITIONS)
    if set(manifest["sources"]) != expected:
        raise ValueError("source inventory differs")
    for name, h in manifest["sources"].items():
        if (
            old.pilot.shared.digest(ROOT / name) != h
            or old.pilot.shared.digest(folder / "sources" / name) != h
        ):
            raise ValueError("frozen source changed: " + name)
    return manifest


def freeze(folder, anchor, candidate_python, wheel, namespace, n=10, scenario="scaling"):
    if folder.exists():
        raise FileExistsError(folder)
    base = old.verify(anchor)
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);import lmm_random_slope_v2 as p;"
        "print(json.dumps(p.engine_environment()))"
    )
    installed = json.loads(
        subprocess.check_output(
            [str(candidate_python), "-c", code, str(ROOT / "scripts")],
            cwd=ROOT,
            env=dict(os.environ, **old.pilot.shared.THREAD_LIMITS),
            text=True,
        )
    )
    if installed != base["candidate"] or old.pilot.shared.digest(wheel) != base["wheel_sha256"]:
        raise ValueError("installed candidate/wheel differs")
    with zipfile.ZipFile(wheel) as archive:
        native = [p for p in archive.namelist() if p.endswith((".pyd", ".so"))]
        if len(native) != 1 or sha(archive.read(native[0])) != installed["native_sha256"]:
            raise ValueError("wheel/native mismatch")
    manifest = copy.deepcopy(base)
    manifest.update(
        protocol=PROTOCOL,
        namespace=namespace,
        scenario=scenario,
        primary_n=n,
        secondary_n=n,
        reference_n=0 if scenario == "native_fault" else n,
        cases=base["cases"][:1] if scenario == "native_fault" else base["cases"],
        workers=1 if scenario == "native_fault" else 4,
        caps=Caps().record(),
        chunk_size=8,
    )
    validate(manifest)
    for name in ADDITIONS:
        manifest["sources"][name] = old.pilot.shared.digest(ROOT / name)
    folder.mkdir(parents=True)
    for name in manifest["sources"]:
        target = folder / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_new(folder / "manifest.json", manifest)
    verify(folder)


def verify_bank(folder):
    bank = old.pilot.load(folder / "bank.json")
    manifests = []
    for name, n, scenario in (
        ("n10", 10, "scaling"),
        ("n20", 20, "scaling"),
        ("controls/native-fault", 3, "native_fault"),
    ):
        path = folder / name
        manifest = verify(path)
        if manifest["primary_n"] != n or manifest["scenario"] != scenario:
            raise ValueError("bank role differs")
        if bank["manifests"].get(name) != old.pilot.shared.digest(path / "manifest.json"):
            raise ValueError("bank manifest identity differs")
        manifests.append(manifest)
    if len({m["namespace"] for m in manifests}) != 3:
        raise ValueError("bank namespaces overlap")
    if any(m["sources"] != manifests[0]["sources"] for m in manifests):
        raise ValueError("bank source freezes differ")
    return bank


def freeze_bank(folder, anchor, candidate_python, wheel, namespace):
    if folder.exists():
        raise FileExistsError(folder)
    for name, n, scenario, role in (
        ("n10", 10, "scaling", "n10"),
        ("n20", 20, "scaling", "n20"),
        ("controls/native-fault", 3, "native_fault", "native"),
    ):
        freeze(
            folder / name,
            anchor,
            candidate_python,
            wheel,
            f"scaling-v1-{namespace}-{role}-a",
            n,
            scenario,
        )
    write_new(
        folder / "bank.json",
        {
            "scope": "three source/seed freezes before any input generation",
            "manifests": {
                name: old.pilot.shared.digest(folder / name / "manifest.json")
                for name in ("n10", "n20", "controls/native-fault")
            },
        },
    )
    verify_bank(folder)


def publish_marker(path, value):
    """Publish only a fully written/fsynced marker, without replacing evidence."""
    pending = path.with_suffix(".pending")
    write_new(pending, value)
    os.link(pending, path)
    pending.unlink()


def execute(journal, engine, row, callback, timings):
    if journal.attempted(engine, row["replicate"]):
        return
    before = time.perf_counter()
    journal.start(engine, row["replicate"])
    timings["start_commit_seconds"] += time.perf_counter() - before
    before = time.perf_counter()
    try:
        fit = callback()
    except (TimeoutError, BrokenPipeError, old.ExecutionIntegrityError):
        raise
    except Exception as error:
        fit = {"converged": False, "methods": {}, "fit_error": f"{type(error).__name__}: {error}"}
    timings["fit_callback_seconds"] += time.perf_counter() - before
    before = time.perf_counter()
    journal.finish(engine, row["replicate"], fit)
    timings["outcome_commit_seconds"] += time.perf_counter() - before


def run_cases(folder, manifest, phase, index, fit, timings):
    for case in manifest["cases"][index :: manifest["workers"]]:
        path = folder / "cases" / case["id"]
        with case_lock(path / "case.lock"):
            journal = Journal(path / "journal.sqlite", case_metadata(manifest, case))
            try:
                before = time.perf_counter()
                journal.validate()
                journal.recover()
                timings["validation_recovery_seconds"] += time.perf_counter() - before
                if phase == "candidate":
                    for first in range(
                        journal.input_count(), journal.metadata["n"], manifest["chunk_size"]
                    ):
                        before = time.perf_counter()
                        journal.add_chunk(
                            [
                                {
                                    "case": case["id"],
                                    "replicate": i,
                                    "y": old.response(case, i, manifest["namespace"]),
                                }
                                for i in range(
                                    first,
                                    min(first + manifest["chunk_size"], journal.metadata["n"]),
                                )
                            ]
                        )
                        timings["input_generation_commit_seconds"] += time.perf_counter() - before
                elif (
                    journal.input_count() != journal.metadata["n"]
                    or journal.db.execute(
                        "SELECT count(*) FROM outcomes WHERE engine='candidate'"
                    ).fetchone()[0]
                    != journal.metadata["n"]
                ):
                    raise ValueError("reference requires complete candidate inventory")
                for row in journal.iter_inputs():
                    if phase == "candidate":
                        execute(journal, "candidate", row, lambda c=case, r=row: fit(c, r), timings)
                    elif row["replicate"] < manifest["reference_n"]:
                        for variant in VARIANTS:
                            execute(
                                journal,
                                "reference:" + variant["id"],
                                row,
                                lambda c=case, r=row, v=variant: fit(c, r, v),
                                timings,
                            )
                before = time.perf_counter()
                journal.validate()
                timings["validation_recovery_seconds"] += time.perf_counter() - before
            finally:
                journal.close()


def worker(folder, phase, index, receipt, rscript=None):
    manifest = verify(folder)
    if (
        type(index) is not int
        or not 0 <= index < manifest["workers"]
        or not re.fullmatch(r"(?:candidate|reference)-[0-9]+", receipt)
    ):
        raise ValueError("invalid worker/receipt identity")
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

            environment = old.pilot.engine_environment()
            if environment != manifest["candidate"]:
                raise ValueError("loaded native differs")

            def fit(case, row):
                if manifest["scenario"] == "native_fault" and row["replicate"] == 1:
                    publish_marker(
                        folder / "fault/native-started.json",
                        {
                            "native_pid": os.getpid(),
                            "environment": environment,
                            "case": case["id"],
                            "replicate": 1,
                            "input_sha": sha(canonical(row)),
                        },
                    )
                    while True:
                        time.sleep(0.1)
                data = pl.DataFrame(
                    {"x": case["x"], "y": row["y"], "group": [f"g{i:03}" for i in case["group"]]}
                )
                return old.pilot.candidate_fit(lme, data)

            run_cases(folder, manifest, phase, index, fit, timings)
            r_env, r_peak = None, None
        else:
            with (folder / "executions" / f"{receipt}-r-{index}.log").open(
                "x", encoding="utf-8"
            ) as log:
                session = old.RSession(rscript, log)
                run_cases(folder, manifest, phase, index, session.fit, timings)
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
                "memory_scope": (
                    "per-native-process resident high-water; "
                    "not simultaneous peak or committed bytes"
                ),
            },
        )
    finally:
        if session is not None:
            session.close()


def launch(folder, phase, candidate_python, rscript=None, inject_native=False):
    manifest = verify(folder)
    verify_bank(folder.parent.parent if manifest["scenario"] == "native_fault" else folder.parent)
    if phase not in ("candidate", "reference") or (
        manifest["scenario"] == "native_fault" and phase != "candidate"
    ):
        raise ValueError("invalid phase for rehearsal")
    marker = folder / "fault/native-started.json"
    if inject_native and (manifest["scenario"] != "native_fault" or marker.exists()):
        raise ValueError("native interruption can occur only once in its dedicated scenario")
    caps, receipt = Caps(**manifest["caps"]), f"{phase}-{time.time_ns()}"
    target, logs = folder / "executions", []
    write_new(
        target / f"{receipt}-start.json",
        {
            "manifest_sha": sha(canonical(manifest)),
            "phase": phase,
            "workers": manifest["workers"],
            "caps": caps.record(),
            "injected_native_interrupt": inject_native,
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
            for i in range(manifest["workers"]):
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
                    env=dict(os.environ, **old.pilot.shared.THREAD_LIMITS),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            result = monitor(
                group,
                caps,
                folder,
                phase_started=before,
                interrupt_when=marker.exists if inject_native else None,
            )
    finally:
        group.close()
        for p in group.processes:
            p.wait(timeout=10)
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
        raise RuntimeError("retained interrupted/failed capped execution: " + receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    init = subs.add_parser("freeze")
    init.add_argument("folder", type=Path)
    for name in ("anchor", "candidate-python", "wheel"):
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument("--namespace", required=True)
    init.add_argument("--n", type=int, default=10)
    init.add_argument("--scenario", choices=("scaling", "native_fault"), default="scaling")
    launch_parser = subs.add_parser("run")
    launch_parser.add_argument("folder", type=Path)
    launch_parser.add_argument("--phase", choices=("candidate", "reference"), required=True)
    launch_parser.add_argument("--candidate-python", type=Path, required=True)
    launch_parser.add_argument("--rscript", type=Path)
    launch_parser.add_argument("--inject-native", action="store_true")
    child = subs.add_parser("worker")
    child.add_argument("folder", type=Path)
    child.add_argument("--phase", choices=("candidate", "reference"), required=True)
    child.add_argument("--index", type=int, required=True)
    child.add_argument("--receipt", required=True)
    child.add_argument("--rscript", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"freeze": freeze, "run": launch, "worker": worker}[command](**args)
