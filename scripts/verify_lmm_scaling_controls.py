"""Receipted negative cap controls and actual native-worker interruption/resume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lmm_random_slope_scaling as run
from lmm_resource_caps import Caps, run_capped
from lmm_stream_store import Journal, canonical, sha


def audit_marker(marker, manifest, before, interrupted_receipt):
    if (
        marker["environment"] != manifest["candidate"]
        or marker["replicate"] != 1
        or marker["case"] != manifest["cases"][0]["id"]
        or marker["input_sha"] != before["inputs"][1]
        or marker["input_sha"] != before["starts"][1][1]
        or marker["native_pid"]
        not in interrupted_receipt["resource_guard"]["final_job_process_ids"]
    ):
        raise ValueError("native marker does not match owned worker/durable input")


def audit_resume(before, after):
    if before["inputs"] != after["inputs"]:
        raise ValueError("resume changed inputs")
    if (
        before["completed_zero"] != after["completed_zero"]
        or before["starts"] != after["starts"][:2]
    ):
        raise ValueError("resume altered completed outcome or previous starts")
    if before["outcome_ids"] != [0] or [r[0] for r in after["starts"]] != [0, 1, 2]:
        raise ValueError("native interruption inventory differs")
    if (
        after["outcome_ids"] != [0, 1, 2]
        or after["interrupted"]["fit_error"] != "interrupted_after_durable_start"
    ):
        raise ValueError("native interruption was hidden or retried")
    if not after["last_fit"].get("converged"):
        raise ValueError("never-started native fit did not converge")


def snapshot(folder):
    manifest = run.verify(folder)
    case = manifest["cases"][0]
    journal = Journal(
        folder / "cases" / case["id"] / "journal.sqlite", run.case_metadata(manifest, case)
    )
    try:
        journal.validate()
        inputs = [sha(canonical(row)) for row in journal.iter_inputs()]
        starts = list(
            journal.db.execute(
                "SELECT replicate,input_sha,started FROM starts "
                "WHERE engine='candidate' ORDER BY replicate"
            )
        )
        outcomes = dict(journal.iter_outcomes("candidate"))
        saved = journal.db.execute(
            "SELECT sha FROM outcomes WHERE engine='candidate' AND replicate=0"
        ).fetchone()
        return {
            "inputs": inputs,
            "starts": starts,
            "completed_zero": saved[0] if saved else None,
            "outcome_ids": list(outcomes),
            "interrupted": outcomes.get(1),
            "last_fit": outcomes.get(2),
        }
    finally:
        journal.close()


def controls(folder, candidate_python):
    run.verify_bank(folder.parent)
    native = folder / "native-fault"
    run.verify(native)
    child = (
        "try:\n bytearray(512*1024**2)\nexcept MemoryError:\n "
        "print('allocation_denied',flush=True)\n raise SystemExit(91)\nraise SystemExit(92)"
    )
    nested = """import sys
sys.path.insert(0,sys.argv[1])
from lmm_random_slope_stream import ProcessGroup
group=ProcessGroup()
try:
 p=group.start([sys.executable,'-c',sys.argv[2]])
 raise SystemExit(p.wait())
finally: group.close()
"""
    cases = [
        (
            "time",
            Caps(seconds=0.15),
            [sys.executable, "-c", "import time;time.sleep(30)"],
            "phase_time_cap",
        ),
        (
            "disk",
            Caps(disk_bytes=65536),
            [
                sys.executable,
                "-c",
                "import sys,time;from pathlib import Path;"
                "Path(sys.argv[1]).write_bytes(b'x'*1048576);time.sleep(30)",
                str((folder / "disk/retained.bin").resolve()),
            ],
            "sampled_disk_cap",
        ),
        (
            "memory",
            Caps(job_commit_bytes=128 * 1024**2),
            [sys.executable, "-c", nested, str(run.ROOT / "scripts"), child],
            "memory_denial",
        ),
    ]
    run.write_new(
        folder / "controls-plan.json",
        {
            "scope": "prospective synthetic negative controls; not statistical data",
            "source_sha256": run.old.pilot.shared.digest(Path(__file__)),
            "cases": [
                {
                    "label": label,
                    "caps": caps.record(),
                    "program_sha256": sha(code[2].encode()),
                    "expected": expected,
                }
                for label, caps, code, expected in cases
            ],
            "native_fault": (
                "replicate0 completed;replicate1 native-loaded durable start interrupted;"
                "only2 fitted on resume"
            ),
        },
    )
    results = {}
    for label, caps, args, expected in cases:
        directory = folder / label
        directory.mkdir()
        result = run_capped(args, caps, directory, directory / "receipts", label)
        if expected == "memory_denial":
            logs = "".join(p.read_text() for p in (directory / "receipts").glob("*.log"))
            if not (
                "allocation_denied" in logs
                or result["resource_guard"]["abort_reason"]
                in ("job_memory_limit_event", "os_job_charge_above_limit")
            ):
                raise ValueError("hard memory cap was not independently demonstrated")
        elif result["resource_guard"]["abort_reason"] != expected:
            raise ValueError("wrong negative-cap abort reason")
        if not result["exit_codes"] or result["exit_codes"] == [0]:
            raise ValueError("negative control did not terminate its owned worker")
        results[label] = result
    try:
        run.launch(native, "candidate", candidate_python, inject_native=True)
    except RuntimeError:
        pass
    else:
        raise ValueError("native interruption did not occur")
    first = sorted((native / "executions").glob("*-end.json"))[-1]
    if (
        json.loads(first.read_text())["resource_guard"]["abort_reason"]
        != "injected_native_interrupt"
    ):
        raise ValueError("native fault failed for an unrelated resource/execution reason")
    marker = run.old.pilot.load(native / "fault/native-started.json")
    if marker["environment"] != run.verify(native)["candidate"] or marker["replicate"] != 1:
        raise ValueError("interrupted worker had not verified the intended native engine")
    before = snapshot(native)
    audit_marker(marker, run.verify(native), before, run.old.pilot.load(first))
    run.write_new(folder / "native-before-resume.json", before)
    run.launch(native, "candidate", candidate_python)
    after = snapshot(native)
    audit_resume(before, after)
    run.write_new(folder / "native-after-resume.json", after)
    run.write_new(
        folder / "result.json",
        {
            "scope": "cap/native fault controls only; excluded from scaling/calibration data",
            "accepted": True,
            "negative_controls": results,
            "native_marker": marker,
            "before_sha256": run.old.pilot.shared.digest(folder / "native-before-resume.json"),
            "after_sha256": run.old.pilot.shared.digest(folder / "native-after-resume.json"),
            "native_inventory": {
                "inputs": 3,
                "starts": 3,
                "outcomes": 3,
                "completed_fits": 2,
                "interrupted": 1,
                "r_attempts": 0,
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--candidate-python", type=Path, required=True)
    controls(**vars(parser.parse_args()))
