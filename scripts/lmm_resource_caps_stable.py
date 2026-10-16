"""Separate disk-race repair for post-study diagnostics; historical controller unchanged."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lmm_random_slope_stream import ProcessGroup


@dataclass(frozen=True)
class Caps:
    seconds: float = 900
    job_commit_bytes: int = 4 * 1024**3
    disk_bytes: int = 512 * 1024**2
    min_free_bytes: int = 1024**3
    poll_seconds: float = 0.1

    def __post_init__(self):
        for name in ("job_commit_bytes", "disk_bytes", "min_free_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("resource bytes must be positive integers")
        for name, maximum in (("seconds", 3600), ("poll_seconds", 1)):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError("invalid time/poll cap")

    def record(self):
        return asdict(self)


def disk_sample(folder):
    folder = Path(folder)
    if not stat.S_ISDIR(folder.stat().st_mode):
        raise NotADirectoryError(folder)
    total = 0
    pending = [folder]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            if directory == folder:
                raise
            continue
        # pathlib.rglob suppresses permission errors while enumerating subtrees.
        # Explicit scandir keeps all permission/I/O failures visible to monitor.
        with entries as scanned:
            for entry in scanned:
                path = Path(entry.path)
                try:
                    info = path.stat()
                except FileNotFoundError:
                    # SQLite can delete WAL/SHM after enumeration. Missing
                    # children occupy no bytes now; other errors remain fatal.
                    continue
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                elif stat.S_ISDIR(info.st_mode) and not entry.is_symlink():
                    pending.append(path)
    if not stat.S_ISDIR(folder.stat().st_mode):
        raise NotADirectoryError(folder)
    return {
        "bytes": total,
        "free_bytes": shutil.disk_usage(folder).free,
    }


def violation(caps, elapsed, disk, memory_event=False):
    if memory_event:
        return "job_memory_limit_event"
    if elapsed > caps.seconds:
        return "phase_time_cap"
    if disk["bytes"] > caps.disk_bytes:
        return "sampled_disk_cap"
    if disk["free_bytes"] < caps.min_free_bytes:
        return "free_space_reserve"
    return None


class CappedGroup(ProcessGroup):
    """Hard aggregate committed-byte ceiling, including nested native R children.

    A job limit denies allocation; it is not a resident-memory/whole-host cap.
    Completion messages are supplementary event evidence, not proof that no
    denied allocation occurred when no message was observed.
    """

    def __init__(self, caps):
        if os.name != "nt":
            raise RuntimeError("reviewed hard job-memory backend requires Windows")
        super().__init__()
        self.port = None
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

        class Association(ctypes.Structure):
            _fields_ = [("key", ctypes.c_void_p), ("port", wintypes.HANDLE)]

        self.extended = Extended

        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_int64 * 4), ("counts", wintypes.DWORD * 4)]

        self.accounting = Accounting

        class ProcessIds(ctypes.Structure):
            _fields_ = [
                ("assigned", wintypes.DWORD),
                ("returned", wintypes.DWORD),
                ("ids", ctypes.c_size_t * 8192),
            ]

        self.process_ids = ProcessIds
        self.kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.kernel.CreateIoCompletionPort.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        self.kernel.CreateIoCompletionPort.restype = wintypes.HANDLE
        self.kernel.GetQueuedCompletionStatus.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.DWORD,
        ]
        try:
            limits = Extended()
            limits.basic.flags = 0x2000 | 0x200  # kill on close + hard job committed-memory limit
            limits.job_memory = caps.job_commit_bytes
            if not self.kernel.SetInformationJobObject(
                self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            self.port = self.kernel.CreateIoCompletionPort(wintypes.HANDLE(-1), None, 0, 1)
            association = Association(1, self.port)
            if not self.port or not self.kernel.SetInformationJobObject(
                self.job, 7, ctypes.byref(association), ctypes.sizeof(association)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        from ctypes import wintypes

        limits = self.extended()
        if not self.kernel.QueryInformationJobObject(
            self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        usage = (ctypes.c_uint64 * 2)()
        if not self.kernel.QueryInformationJobObject(
            self.job, 28, ctypes.byref(usage), ctypes.sizeof(usage), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        events = []
        pids = self.process_ids()
        if not self.kernel.QueryInformationJobObject(
            self.job, 3, ctypes.byref(pids), ctypes.sizeof(pids), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if pids.returned != pids.assigned:
            raise ValueError("incomplete owned-job process inventory")
        accounting = self.accounting()
        if not self.kernel.QueryInformationJobObject(
            self.job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        message, key, pid = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
        while self.kernel.GetQueuedCompletionStatus(
            self.port, ctypes.byref(message), ctypes.byref(key), ctypes.byref(pid), 0
        ):
            events.append({"message": message.value, "native_pid": pid.value})
        if ctypes.get_last_error() != 258:  # WAIT_TIMEOUT means the event queue is empty
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "os_peak_job_memory_bytes": int(limits.peak_job),
            "current_job_commit_bytes": int(usage[0]),
            "configured_job_commit_bytes": int(limits.job_memory),
            "memory_limit_event": any(e["message"] in (9, 10) for e in events),
            "events": events,
            "active_job_processes": int(accounting.counts[2]),
            "job_process_ids": list(pids.ids[: pids.returned]),
        }

    def close(self):
        if getattr(self, "job", None) and not self.closed:
            from ctypes import wintypes

            self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            self.kernel.TerminateJobObject(self.job, 0xE0000001)
        super().close()
        if getattr(self, "port", None):
            self.kernel.CloseHandle(self.port)
            self.port = None


def monitor(group, caps, folder, phase_started=None, interrupt_when=None):
    """Poll owned workers, retaining maxima and the first abort reason."""
    before = time.perf_counter()
    phase_started = before if phase_started is None else phase_started
    peak_disk = peak_commit = os_peak = samples = 0
    minimum_free = None
    all_events, reason = [], None
    previous_sample, largest_gap, roots_done_at = phase_started, 0.0, None
    active = None
    final_pids = []
    while True:
        try:
            roots_done = all(p.poll() is not None for p in group.processes)
            disk, memory = disk_sample(folder), group.snapshot()
            now = time.perf_counter()
            largest_gap = max(largest_gap, now - previous_sample)
            previous_sample = now
            active = memory["active_job_processes"]
            final_pids = memory["job_process_ids"]
            if memory["configured_job_commit_bytes"] != caps.job_commit_bytes:
                raise ValueError("hard memory limit differs from requested cap")
            samples += 1
            peak_disk = max(peak_disk, disk["bytes"])
            minimum_free = (
                disk["free_bytes"]
                if minimum_free is None
                else min(minimum_free, disk["free_bytes"])
            )
            peak_commit = max(peak_commit, memory["current_job_commit_bytes"])
            os_peak = max(os_peak, memory["os_peak_job_memory_bytes"])
            if memory["current_job_commit_bytes"] > caps.job_commit_bytes:
                raise ValueError("observed current commit exceeds hard cap")
            all_events.extend(memory["events"])
            reason = violation(
                caps, time.perf_counter() - phase_started, disk, memory["memory_limit_event"]
            )
            if reason is None and os_peak > caps.job_commit_bytes:
                reason = "os_job_charge_above_limit"
            if reason is None and interrupt_when is not None and interrupt_when():
                current = group.snapshot()
                final_pids = current["job_process_ids"]
                active = current["active_job_processes"]
                peak_commit = max(peak_commit, current["current_job_commit_bytes"])
                os_peak = max(os_peak, current["os_peak_job_memory_bytes"])
                all_events.extend(current["events"])
                reason = violation(
                    caps, time.perf_counter() - phase_started, disk, current["memory_limit_event"]
                )
                if reason is None:
                    reason = (
                        "os_job_charge_above_limit"
                        if os_peak > caps.job_commit_bytes
                        else "injected_native_interrupt"
                    )
            if roots_done and active:
                roots_done_at = now if roots_done_at is None else roots_done_at
                if reason is None and now - roots_done_at >= 0.5:
                    reason = "root_exit_with_live_descendants"
        except (OSError, ValueError) as error:
            reason = "resource_measurement_unavailable:" + type(error).__name__
        if reason or (roots_done and active == 0):
            break
        time.sleep(caps.poll_seconds)
    return {
        "abort_reason": reason,
        "samples": samples,
        "maximum_sampled_disk_bytes": peak_disk,
        "minimum_sampled_free_bytes": minimum_free,
        "maximum_sampled_job_commit_bytes": peak_commit,
        "os_peak_job_memory_bytes": os_peak,
        "job_events": all_events,
        "monitor_seconds": time.perf_counter() - before,
        "active_phase_seconds": time.perf_counter() - phase_started,
        "largest_sample_gap_seconds": largest_gap,
        "final_active_job_processes": active,
        "final_job_process_ids": final_pids,
        "scope": (
            "hard job committed-memory limit; time/disk polled, may overshoot; "
            "parent and separately capped analysis outside this job; OS peak charge "
            "can include denied allocations"
        ),
    }


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_identity(args):
    """Exact arguments are bound by hash without exposing local absolute paths."""
    argv = [str(a) for a in args]
    executable = Path(shutil.which(argv[0]) or argv[0]).resolve()
    sources = {
        str(i): {"name": Path(a).name, "sha256": file_hash(a)}
        for i, a in enumerate(argv[1:], 1)
        if Path(a).suffix.lower() in (".py", ".r") and Path(a).is_file()
    }
    return {
        "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
        "executable": {"name": executable.name, "sha256": file_hash(executable)},
        "script_sources": sources,
    }


def bindings(files, missing=False):
    return (
        {name: file_hash(path) if Path(path).is_file() else None for name, path in files.items()}
        if missing
        else {name: file_hash(path) for name, path in files.items()}
    )


def acceptance_failures(end, caps):
    """Reject missing or contradictory completed-job measurements."""
    failures, guard = [], end.get("resource_guard", {})
    if end.get("caps") != caps:
        return ["cap configuration differs"]
    if guard.get("abort_reason", "missing") is not None:
        failures.append("aborted/unmeasured execution")
    if type(guard.get("samples")) is not int or guard["samples"] <= 0:
        failures.append("missing samples")
    for measured, bound in (
        ("maximum_sampled_job_commit_bytes", "job_commit_bytes"),
        ("os_peak_job_memory_bytes", "job_commit_bytes"),
        ("maximum_sampled_disk_bytes", "disk_bytes"),
        ("active_phase_seconds", "seconds"),
    ):
        value = guard.get(measured)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= caps[bound]
        ):
            failures.append("missing/out-of-limit " + measured)
    for value in (
        guard.get("minimum_sampled_free_bytes"),
        end.get("final_disk", {}).get("free_bytes"),
    ):
        if type(value) is not int or value < caps["min_free_bytes"]:
            failures.append("missing/breached free reserve")
    final_bytes = end.get("final_disk", {}).get("bytes")
    if type(final_bytes) is not int or not 0 < final_bytes <= caps["disk_bytes"]:
        failures.append("missing/breached final disk")
    gap = guard.get("largest_sample_gap_seconds")
    if type(gap) not in (int, float) or not math.isfinite(gap) or gap <= 0:
        failures.append("missing sampling gap")
    if guard.get("final_active_job_processes") != 0:
        failures.append("remaining/unmeasured job processes")
    events = guard.get("job_events")
    if not isinstance(events, list) or any(e.get("message") in (9, 10) for e in events):
        failures.append("missing events or retained memory event")
    return failures


def run_capped(args, caps, folder, receipt_folder, label, input_files=None, output_files=None):
    from lmm_random_slope_stream import ROOT, pilot, write_new

    receipt_folder.mkdir(parents=True, exist_ok=True)
    prefix = label + "-" + str(time.time_ns())
    start_path = receipt_folder / (prefix + "-start.json")
    identity = {
        "caps": caps.record(),
        "label": label,
        "command": command_identity(args),
        "inputs": bindings(input_files or {}),
        "thread_limits": pilot.shared.THREAD_LIMITS,
    }
    write_new(start_path, identity)
    group, result, before = (
        CappedGroup(caps),
        {"abort_reason": "launch_not_completed"},
        time.perf_counter(),
    )
    try:
        disk = disk_sample(folder)
        preflight = violation(caps, 0, disk)
        if preflight:
            result = {"abort_reason": preflight, "preflight_disk": disk}
        else:
            with (receipt_folder / (prefix + ".log")).open("x", encoding="utf-8") as log:
                group.start(
                    args,
                    stdout=log,
                    stderr=log,
                    cwd=ROOT,
                    env=dict(os.environ, **pilot.shared.THREAD_LIMITS),
                )
                result = monitor(group, caps, folder, phase_started=before)
    finally:
        group.close()
        for process in group.processes:
            process.wait(timeout=10)
        end = {
            "caps": caps.record(),
            "resource_guard": result,
            "seconds": time.perf_counter() - before,
            "exit_codes": [p.returncode for p in group.processes],
            "final_disk": disk_sample(folder),
            "start_sha256": file_hash(start_path),
            "identity": identity,
            "outputs": bindings(output_files or {}, missing=True),
        }
        write_new(receipt_folder / (prefix + "-end.json"), end)
    end["receipt"] = prefix + "-end.json"
    return end
