"""Retained synthetic negative controls for the successor's hardened controller."""

import argparse
import json
from pathlib import Path

from diagnose_lmm_stopping import digest, write_new
from lmm_calibration import THREAD_LIMITS
from lmm_resource_caps_stable import Caps, command_identity, run_capped

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = [
    "scripts/verify_lmm_covariance_caps.py",
    "scripts/lmm_resource_caps_stable.py",
    "scripts/diagnose_lmm_stopping.py",
    "scripts/lmm_calibration.py",
    "scripts/lmm_random_slope_v2.py",
    "scripts/diagnose_lmm_random_slope_reference.py",
    "scripts/lmm_stream_store.py",
    "scripts/lmm_random_slope_stream.py",
]


def cases():
    child = (
        "try:\n bytearray(512*1024**2)\nexcept MemoryError:\n "
        "print('allocation_denied',flush=True)\n raise SystemExit(91)\nraise SystemExit(92)"
    )
    return [
        ("time", Caps(seconds=0.15), "import time;time.sleep(30)", "phase_time_cap"),
        (
            "disk",
            Caps(disk_bytes=65536),
            "import sys,time;from pathlib import Path;"
            "Path(sys.argv[1]).write_bytes(b'x'*1048576);time.sleep(30)",
            "sampled_disk_cap",
        ),
        ("memory", Caps(job_commit_bytes=128 * 1024**2), child, "memory_denial"),
    ]


def audit(folder, sources=None, result=None, expected_python=None, copied=False):
    plan = json.loads((folder / "plan.json").read_bytes())
    if result is None:
        result = json.loads((folder / "result.json").read_bytes())
    expected_sources = {name: digest(ROOT / name) for name in SOURCE_NAMES}
    if (
        plan["sources"] != expected_sources
        or sources is not None
        and any(sources.get(name) != value for name, value in expected_sources.items())
        or result.get("accepted") is not True
        or result["plan_sha256"] != digest(folder / "plan.json")
        or set(result["controls"]) != {"time", "disk", "memory"}
        or len(plan["cases"]) != 3
    ):
        raise ValueError("cap-control plan/source identity differs")
    invocation = plan["invocation"]
    original = Path(invocation["folder"])
    interpreter = Path(invocation["python"])
    if (
        not original.is_absolute()
        or not interpreter.is_absolute()
        or not copied
        and original != folder.resolve()
        or expected_python is not None
        and interpreter != expected_python.resolve()
        or digest(interpreter) != invocation["python_sha256"]
    ):
        raise ValueError("cap-control original invocation differs")
    for declaration, (label, cap, program, expected) in zip(plan["cases"], cases()):
        if declaration != {
            "label": label,
            "caps": cap.record(),
            "program": program,
            "expected": expected,
            "command": declaration.get("command"),
        }:
            raise ValueError("cap-control declaration differs")
        argv = [str(interpreter), "-c", program]
        if label == "disk":
            argv.append(str(original / label / "retained.bin"))
        if declaration["command"] != command_identity(argv):
            raise ValueError("cap-control command invocation differs")
        row = result["controls"][label]
        name = row["receipt"]
        if (
            Path(name).name != name
            or not name.startswith(label + "-")
            or not name.endswith("-end.json")
        ):
            raise ValueError("invalid cap receipt name")
        directory = folder / label / "receipts"
        end = json.loads((directory / name).read_bytes())
        start_path = directory / name.replace("-end.json", "-start.json")
        start = json.loads(start_path.read_bytes())
        identity = {
            "caps": cap.record(),
            "label": label,
            "command": declaration["command"],
            "inputs": {},
            "thread_limits": THREAD_LIMITS,
        }
        if (
            start != identity
            or end.get("identity") != identity
            or end.get("caps") != cap.record()
            or end.get("start_sha256") != digest(start_path)
            or {k: v for k, v in row.items() if k not in ("receipt", "log_sha256")} != end
            or end.get("outputs") != {}
            or len(end.get("exit_codes", [])) != 1
            or end["exit_codes"] == [0]
            or end["resource_guard"].get("samples", 0) <= 0
        ):
            raise ValueError("cap-control raw receipt rejected")
        log_path = directory / name.replace("-end.json", ".log")
        if row.get("log_sha256") != digest(log_path):
            raise ValueError("cap-control log binding differs")
        guard = end["resource_guard"]
        if label == "memory":
            log = log_path.read_text()
            denied = "allocation_denied" in log and end["exit_codes"] == [91]
            event = guard.get("abort_reason") == "job_memory_limit_event" and any(
                event.get("message") in (9, 10) for event in guard.get("job_events", [])
            )
            if not (denied or event):
                raise ValueError("hard memory enforcement was not demonstrated")
        elif guard.get("abort_reason") != expected:
            raise ValueError("wrong negative cap result")
    return result


def controls(folder, candidate_python):
    folder.mkdir(parents=True, exist_ok=False)
    declarations = []
    candidate_python = candidate_python.resolve()
    for label, cap, program, expected in cases():
        args = [str(candidate_python), "-c", program]
        if label == "disk":
            args.append(str((folder / label / "retained.bin").resolve()))
        declarations.append(
            {
                "label": label,
                "caps": cap.record(),
                "program": program,
                "expected": expected,
                "command": command_identity(args),
            }
        )
    write_new(
        folder / "plan.json",
        {
            "scope": "prospective synthetic cap controls only; no statistical inputs",
            "sources": {name: digest(ROOT / name) for name in SOURCE_NAMES},
            "invocation": {
                "folder": str(folder.resolve()),
                "python": str(candidate_python),
                "python_sha256": digest(candidate_python),
            },
            "cases": declarations,
        },
    )
    results = {}
    for label, cap, program, expected in cases():
        target = folder / label
        target.mkdir()
        args = [str(candidate_python), "-c", program]
        if label == "disk":
            args.append(str((target / "retained.bin").resolve()))
        receipt = run_capped(args, cap, target, target / "receipts", label)
        log_path = target / "receipts" / receipt["receipt"].replace("-end.json", ".log")
        results[label] = {**receipt, "log_sha256": digest(log_path)}
    result = {
        "accepted": True,
        "confirmation_claim": False,
        "plan_sha256": digest(folder / "plan.json"),
        "controls": results,
    }
    try:
        audit(folder, result=result)
    except (ValueError, KeyError, OSError) as error:
        write_new(
            folder / "failed.json", {"accepted": False, "error": str(error), "controls": results}
        )
        raise
    write_new(folder / "result.json", result)
    print(json.dumps({"accepted": True, "controls": list(results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--candidate-python", type=Path, required=True)
    controls(**vars(parser.parse_args()))
