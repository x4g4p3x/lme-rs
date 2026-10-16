"""Frozen, bounded random-slope rehearsal runner. Full confirmation is disabled."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import lmm_random_slope_v2 as pilot
from diagnose_lmm_random_slope_reference import VARIANTS
from lmm_stream_store import Journal, canonical, case_lock, sha

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "lmm-random-slope-stream-rehearsal-v1"
NEW_SOURCES = [
    "scripts/lmm_stream_store.py",
    "scripts/lmm_random_slope_stream.py",
    "scripts/report_lmm_random_slope_stream.py",
    "scripts/plan_lmm_random_slope_confirmation.py",
    "scripts/diagnose_lmm_random_slope_reference.py",
    "scripts/tests/test_lmm_stream_store.py",
    "scripts/tests/test_lmm_random_slope_stream.py",
    "comparisons/scientific/lmm_random_slope_stream.R",
    "comparisons/scientific/lmm_random_slope_stream_check.R",
    "docs/LMM_RANDOM_SLOPE_STREAM_REHEARSAL_PROTOCOL.md",
    "scientific/RANDOM_SLOPE_PILOT_V2_FREEZE.json",
]


class ExecutionIntegrityError(RuntimeError):
    pass


def engine_inventory():
    anchor = pilot.load(ROOT / "scientific/RANDOM_SLOPE_PILOT_V2_FREEZE.json")
    names = sorted(set(anchor["sources"]) - set(pilot.SOURCES))
    return anchor, names


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def response(case, replicate, namespace):
    if (
        type(replicate) is not int
        or replicate < 0
        or not re.fullmatch(r"[a-z0-9-]{1,64}", namespace)
    ):
        raise ValueError("invalid fresh rehearsal namespace/key")
    g, n = case["groups"], len(case["x"])
    draws = pilot.shared.normals(f"{PROTOCOL}|{namespace}|{case['id']}|{replicate}", 2 * g + n)
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


def validate_design(manifest):
    if manifest["mode"] != "rehearsal" or manifest["protocol"] != PROTOCOL:
        raise ValueError("full confirmation is disabled")
    if not re.fullmatch(r"[a-z0-9-]{1,64}", manifest["namespace"]):
        raise ValueError("invalid namespace")
    if manifest["cases"] != pilot.registry() or manifest["variants"] != VARIANTS:
        raise ValueError("design/reference-bank inventory differs")
    for name in ("primary_n", "secondary_n", "reference_n", "chunk_size"):
        value = manifest[name]
        maximum = 64 if name == "chunk_size" else 20
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("rehearsal budget/size exceeds fixed safety limit")
    if manifest["reference_n"] > min(manifest["primary_n"], manifest["secondary_n"]):
        raise ValueError("reference subset exceeds candidate inventory")


def case_metadata(manifest, case):
    return {
        "case": case["id"],
        "manifest_sha": sha(canonical(manifest)),
        "n": manifest["primary_n"] if case["primary_future"] else manifest["secondary_n"],
        "reference_n": manifest["reference_n"],
        "chunk_size": manifest["chunk_size"],
        "engines": ["candidate"] + ["reference:" + v["id"] for v in VARIANTS],
    }


def verify(folder):
    manifest = pilot.load(folder / "manifest.json")
    validate_design(manifest)
    anchor, engines = engine_inventory()
    if (
        manifest["engine_sources"] != engines
        or manifest["candidate"] != anchor["environment"]
        or manifest["control"] != pilot.CONTROL
        or manifest["engine_commit"] != pilot.ENGINE_COMMIT
    ):
        raise ValueError("candidate/source/control identity differs from reviewed engine")
    expected_sources = set(pilot.SOURCES + NEW_SOURCES + manifest["engine_sources"])
    if set(manifest["sources"]) != expected_sources:
        raise ValueError("source inventory differs")
    for name, digest in manifest["sources"].items():
        if (
            pilot.shared.digest(ROOT / name) != digest
            or pilot.shared.digest(folder / "sources" / name) != digest
        ):
            raise ValueError(f"frozen source changed: {name}")
    return manifest


def freeze(
    folder,
    anchor,
    candidate_python,
    wheel,
    namespace,
    primary_n=2,
    secondary_n=2,
    reference_n=2,
    chunk_size=8,
):
    if folder.exists():
        raise FileExistsError(folder)
    original = pilot.verify_run(anchor)
    environment = dict(os.environ, **pilot.shared.THREAD_LIMITS)
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);"
        "import lmm_random_slope_v2 as p;print(json.dumps(p.engine_environment()))"
    )
    installed = json.loads(
        subprocess.check_output(
            [str(candidate_python), "-c", code, str(ROOT / "scripts")],
            cwd=ROOT,
            env=environment,
            text=True,
        )
    )
    if installed != original["freeze"]["environment"]:
        raise ValueError("candidate environment differs from the v2 candidate")
    if pilot.shared.digest(wheel) != original["freeze"]["wheel"]["sha256"]:
        raise ValueError("candidate wheel differs")
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith((".pyd", ".so"))]
        if len(names) != 1 or sha(archive.read(names[0])) != installed["native_sha256"]:
            raise ValueError("installed native differs from wheel")
    _, engine_sources = engine_inventory()
    for name in engine_sources:
        if pilot.shared.digest(ROOT / name) != original["freeze"]["sources"][name]:
            raise ValueError("engine source no longer matches the v2 binary")
    manifest = {
        "schema_version": 1,
        "mode": "rehearsal",
        "protocol": PROTOCOL,
        "namespace": namespace,
        "primary_n": primary_n,
        "secondary_n": secondary_n,
        "reference_n": reference_n,
        "chunk_size": chunk_size,
        "cases": pilot.registry(),
        "variants": VARIANTS,
        "candidate": installed,
        "control": pilot.CONTROL,
        "engine_commit": pilot.ENGINE_COMMIT,
        "engine_sources": engine_sources,
        "wheel_sha256": pilot.shared.digest(wheel),
        "sources": {},
    }
    validate_design(manifest)
    for name in sorted(set(pilot.SOURCES + NEW_SOURCES + engine_sources)):
        manifest["sources"][name] = pilot.shared.digest(ROOT / name)
    folder.mkdir(parents=True)
    for name in manifest["sources"]:
        target = folder / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_new(folder / "manifest.json", manifest)
    verify(folder)


def peak_working_set(pid=None):
    """OS-maintained native-process high-water mark, never a venv launcher sample."""
    pid = os.getpid() if pid is None else pid
    if os.name == "nt":
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

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x410, False, pid)
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            get = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            return (
                int(counters.PeakWorkingSetSize)
                if get(handle, ctypes.byref(counters), counters.cb)
                else None
            )
        finally:
            kernel.CloseHandle(handle)
    path = Path(f"/proc/{pid}/status")
    if path.exists():
        match = re.search(r"^VmHWM:\s+(\d+) kB", path.read_text(), re.MULTILINE)
        return int(match[1]) * 1024 if match else None
    return None


class ProcessGroup:
    """Own descendants even after a root exits: Windows jobs or POSIX sessions."""

    def __init__(self, inherit=False):
        self.processes = []
        self.inherit = inherit
        self.closed = False
        self.job = None
        if os.name == "nt":
            from ctypes import wintypes

            class Basic(ctypes.Structure):
                _fields_ = [
                    ("process_time", ctypes.c_int64),
                    ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD),
                    ("min_ws", ctypes.c_size_t),
                    ("max_ws", ctypes.c_size_t),
                    ("active", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD),
                ]

            class Extended(ctypes.Structure):
                _fields_ = [
                    ("basic", Basic),
                    ("io", ctypes.c_uint64 * 6),
                    ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t),
                    ("peak_job", ctypes.c_size_t),
                ]

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.job = self.kernel.CreateJobObjectW(None, None)
            limits = Extended()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.job or not self.kernel.SetInformationJobObject(
                self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                self.close()
                raise ctypes.WinError(ctypes.get_last_error())

    def start(self, args, **kwargs):
        if self.closed:
            raise RuntimeError("process group is already closed")
        try:
            return self._start(args, **kwargs)
        except BaseException:
            self.close()
            raise

    def _start(self, args, **kwargs):
        if os.name == "nt":
            # Assign the suspended root before it can spawn any native children.
            process = subprocess.Popen(args, creationflags=0x4 | 0x08000000, **kwargs)
            if not self.kernel.AssignProcessToJobObject(self.job, int(process._handle)):
                process.kill()
                process.wait(timeout=10)
                raise ctypes.WinError(ctypes.get_last_error())
            self.processes.append(process)
            resume = ctypes.WinDLL("ntdll").NtResumeProcess
            resume.argtypes = [ctypes.c_void_p]
            resume.restype = ctypes.c_long
            if resume(int(process._handle)) < 0:
                self.close()
                raise RuntimeError("could not resume assigned native worker")
        else:
            process = subprocess.Popen(args, start_new_session=not self.inherit, **kwargs)
            self.processes.append(process)
        return process

    def close(self):
        if self.closed:
            return
        self.closed = True
        if os.name == "nt":
            if self.job:
                self.kernel.CloseHandle(self.job)
                self.job = None
        else:
            for process in self.processes:
                if self.inherit:
                    if process.poll() is None:
                        process.kill()
                    continue
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class RSession:
    def __init__(self, rscript, log):
        self.group = ProcessGroup(inherit=os.name != "nt")
        self.process = self.group.start(
            [
                str(rscript),
                "--vanilla",
                str(ROOT / "comparisons/scientific/lmm_random_slope_stream.R"),
            ],
            cwd=ROOT,
            env=dict(os.environ, **pilot.shared.THREAD_LIMITS),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.messages = queue.Queue(maxsize=2)

        def read():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.environment = self.receive()
            if type(self.environment.get("native_pid")) is not int:
                raise ValueError("R did not identify its native process")
        except BaseException:
            self.close()
            raise

    def receive(self):
        try:
            line = self.messages.get(timeout=120)
        except queue.Empty as error:
            raise TimeoutError("R reference attempt exceeded 120 seconds") from error
        if line is None:
            raise RuntimeError("R reference process ended without an outcome")
        return json.loads(line)

    def request(self, value):
        self.process.stdin.write(canonical(value).decode() + "\n")
        self.process.stdin.flush()
        return self.receive()

    def fit(self, case, row, variant):
        result = self.request(
            {"case": case, "y": row["y"], "replicate": row["replicate"], "variant": variant}
        )
        if (result.get("case"), result.get("replicate"), result.get("variant")) != (
            case["id"],
            row["replicate"],
            variant["id"],
        ):
            raise ExecutionIntegrityError("R reference outcome key differs")
        return result["fit"]

    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.group.close()
            self.process.wait(timeout=5)
        finally:
            self.group.close()
        self.reader.join(timeout=1)
        self.process.stdout.close()


def execute(journal, engine, row, callback):
    if journal.attempted(engine, row["replicate"]):
        return
    journal.start(engine, row["replicate"])
    try:
        fit = callback()
    except (TimeoutError, BrokenPipeError, ExecutionIntegrityError):
        # Abort worker; recovery retains the start as an interrupted failure.
        raise
    except Exception as error:
        fit = {"converged": False, "methods": {}, "fit_error": f"{type(error).__name__}: {error}"}
    journal.finish(engine, row["replicate"], fit)


def worker(folder, phase, index, workers, receipt, rscript=None):
    if (
        type(workers) is not int
        or not 1 <= workers <= 4
        or type(index) is not int
        or not 0 <= index < workers
        or not re.fullmatch(r"(?:candidate|reference)-[0-9]+", receipt)
    ):
        raise ValueError("invalid native worker/receipt identity")
    manifest = verify(folder)
    session = None
    if phase == "candidate":
        import lme_python as lme
        import polars as pl

        if pilot.engine_environment() != manifest["candidate"]:
            raise ValueError("loaded candidate environment differs")

        def fit_candidate(case, row):
            data = pl.DataFrame(
                {"x": case["x"], "y": row["y"], "group": [f"g{i:03}" for i in case["group"]]}
            )
            return pilot.candidate_fit(lme, data)

    try:
        if phase == "reference":
            with (folder / "executions" / f"{receipt}-r-{index}.log").open(
                "x", encoding="utf-8"
            ) as log:
                session = RSession(rscript, log)
                run_cases(
                    folder,
                    manifest,
                    phase,
                    index,
                    workers,
                    lambda c, row, v: session.fit(c, row, v),
                )
                r_environment = session.request({"command": "environment"})
                r_peak = peak_working_set(r_environment["native_pid"])
        else:
            run_cases(folder, manifest, phase, index, workers, fit_candidate)
            r_environment, r_peak = None, None
        write_new(
            folder / "executions" / f"{receipt}-worker-{index}.json",
            {
                "phase": phase,
                "manifest_sha": sha(canonical(manifest)),
                "worker": index,
                "native_python_pid": os.getpid(),
                "python_peak_working_set_bytes": peak_working_set(),
                "r_environment": r_environment,
                "r_peak_working_set_bytes": r_peak,
                "memory_scope": (
                    "OS-maintained per-native-process high-water working set; "
                    "sum is not simultaneous peak"
                ),
            },
        )
    finally:
        if session is not None:
            session.close()


def run_cases(folder, manifest, phase, index, workers, fit):
    for case in manifest["cases"][index::workers]:
        path = folder / "cases" / case["id"]
        with case_lock(path / "case.lock"):
            journal = Journal(path / "journal.sqlite", case_metadata(manifest, case))
            try:
                journal.validate()
                journal.recover()
                if phase == "candidate":
                    start, n = journal.input_count(), journal.metadata["n"]
                    for begin in range(start, n, manifest["chunk_size"]):
                        journal.add_chunk(
                            [
                                {
                                    "case": case["id"],
                                    "replicate": i,
                                    "y": response(case, i, manifest["namespace"]),
                                }
                                for i in range(begin, min(begin + manifest["chunk_size"], n))
                            ]
                        )
                elif journal.input_count() != journal.metadata["n"]:
                    raise ValueError("reference phase requires complete candidate input inventory")
                elif (
                    journal.db.execute(
                        "SELECT count(*) FROM outcomes WHERE engine='candidate'"
                    ).fetchone()[0]
                    != journal.metadata["n"]
                ):
                    raise ValueError("reference phase requires all candidate outcomes")
                for row in journal.iter_inputs():
                    if phase == "candidate":
                        execute(journal, "candidate", row, lambda c=case, r=row: fit(c, r))
                    elif row["replicate"] < manifest["reference_n"]:
                        for variant in VARIANTS:
                            execute(
                                journal,
                                "reference:" + variant["id"],
                                row,
                                lambda c=case, r=row, v=variant: fit(c, r, v),
                            )
                journal.validate()
            finally:
                journal.close()


def launch(folder, phase, candidate_python, rscript=None, workers=4, timeout=600):
    manifest = verify(folder)
    if phase not in ("candidate", "reference") or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("invalid worker phase/count")
    if not 0 < timeout <= 3600:
        raise ValueError("invalid phase timeout")
    receipt = f"{phase}-{time.time_ns()}"
    target = folder / "executions"
    write_new(
        target / f"{receipt}-start.json",
        {
            "manifest_sha": sha(canonical(manifest)),
            "phase": phase,
            "workers": workers,
            "timeout_seconds": timeout,
        },
    )
    processes, logs = [], []
    group = ProcessGroup()
    before = time.perf_counter()
    try:
        for index in range(workers):
            log = (target / f"{receipt}-python-{index}.log").open("x", encoding="utf-8")
            logs.append(log)
            args = [
                str(candidate_python),
                str(Path(__file__).resolve()),
                "worker",
                str(folder),
                "--phase",
                phase,
                "--index",
                str(index),
                "--workers",
                str(workers),
                "--receipt",
                receipt,
            ]
            if rscript is not None:
                args.extend(["--rscript", str(rscript)])
            processes.append(
                group.start(
                    args,
                    cwd=ROOT,
                    env=dict(os.environ, **pilot.shared.THREAD_LIMITS),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(p.poll() is None for p in processes):
            if time.perf_counter() - before > timeout:
                group.close()
                break
            time.sleep(0.1)
    finally:
        group.close()
        for process in processes:
            process.wait(timeout=10)
        for log in logs:
            log.close()
        write_new(
            target / f"{receipt}-end.json",
            {
                "seconds": time.perf_counter() - before,
                "exit_codes": [p.returncode for p in processes],
                "case_storage_bytes": sum(
                    p.stat().st_size for p in (folder / "cases").rglob("*") if p.is_file()
                ),
            },
        )
    if any(p.returncode != 0 for p in processes):
        raise RuntimeError(f"retained failed execution: {receipt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    init = subs.add_parser("freeze")
    init.add_argument("folder", type=Path)
    init.add_argument("--anchor", type=Path, required=True)
    init.add_argument("--candidate-python", type=Path, required=True)
    init.add_argument("--wheel", type=Path, required=True)
    init.add_argument("--namespace", required=True)
    for name, default in (
        ("primary-n", 2),
        ("secondary-n", 2),
        ("reference-n", 2),
        ("chunk-size", 8),
    ):
        init.add_argument("--" + name, type=int, default=default)
    run = subs.add_parser("run")
    run.add_argument("folder", type=Path)
    run.add_argument("--phase", choices=("candidate", "reference"), required=True)
    run.add_argument("--candidate-python", type=Path, required=True)
    run.add_argument("--rscript", type=Path)
    run.add_argument("--workers", type=int, default=4)
    child = subs.add_parser("worker")
    child.add_argument("folder", type=Path)
    child.add_argument("--phase", choices=("candidate", "reference"), required=True)
    child.add_argument("--index", type=int, required=True)
    child.add_argument("--workers", type=int, required=True)
    child.add_argument("--receipt", required=True)
    child.add_argument("--rscript", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "freeze":
        freeze(**args)
    elif command == "run":
        launch(**args)
    else:
        worker(**args)
