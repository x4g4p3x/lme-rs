"""Post-study reanalysis of retained fits; never repairs a failed study's gate."""

import argparse
import json
import sys
from pathlib import Path

import lmm_stopping_control as run
from complete_lmm_scaling import audit_command
from lmm_resource_caps_stable import Caps, acceptance_failures, bindings, file_hash, run_capped

SOURCES = [
    "scripts/diagnose_lmm_stopping_pilot.py",
    "scripts/lmm_resource_caps_stable.py",
    "scripts/tests/test_lmm_stable_disk_sample.py",
    "scripts/tests/test_diagnose_lmm_stopping_pilot.py",
    "docs/LMM_RANDOM_SLOPE_STOPPING_DIAGNOSTIC_REANALYSIS.md",
]


def original_inputs(folder, manifest):
    inputs = {"manifest": folder / "manifest.json"}
    inputs.update(
        {
            "journal:" + c["id"]: folder / "cases" / c["id"] / "journal.sqlite"
            for c in manifest["cases"]
        }
    )
    for path in (folder / "cases").glob("*/journal.sqlite-wal"):
        if path.stat().st_size:
            raise ValueError("nonempty WAL prevents immutable input binding")
    inputs.update({"partial:" + p.name: p for p in (folder / "analysis").iterdir() if p.is_file()})
    inputs.update(
        {
            "original-receipt:" + p.name: p
            for p in (folder / "analysis-executions").iterdir()
            if p.is_file()
        }
    )
    return inputs


def freeze(folder):
    manifest = run.verify(folder)
    ends = list((folder / "analysis-executions").glob("analysis-*-end.json"))
    if (
        len(ends) != 1
        or json.loads(ends[0].read_bytes())["resource_guard"]["abort_reason"]
        != "resource_measurement_unavailable:FileNotFoundError"
    ):
        raise ValueError("expected the retained original disk-measurement abort")
    inputs = original_inputs(folder, manifest)
    snapshot = {**manifest["sources"], **{name: file_hash(run.ROOT / name) for name in SOURCES}}
    for name, expected in snapshot.items():
        target = folder / "diagnostic-sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((run.ROOT / name).read_bytes())
        if file_hash(target) != expected:
            raise ValueError("diagnostic source changed during snapshot")
    run.write_new(
        folder / "diagnostic-freeze.json",
        {
            "scope": "post-study reanalysis only; no replacement or retrospective acceptance",
            "original_engineering_accepted": False,
            "numerical_ready": False,
            "confirmation_claim": False,
            "sources": snapshot,
            "original_inputs": bindings(inputs),
            "caps": manifest["caps"],
            "control": manifest["control"],
        },
    )


def verify(folder):
    manifest = run.verify(folder)
    frozen = json.loads((folder / "diagnostic-freeze.json").read_bytes())
    if frozen["caps"] != manifest["caps"] or frozen["control"] != manifest["control"]:
        raise ValueError("diagnostic cap/control changed")
    for name, expected in frozen["sources"].items():
        if (
            file_hash(run.ROOT / name) != expected
            or file_hash(folder / "diagnostic-sources" / name) != expected
        ):
            raise ValueError("diagnostic source changed")
    inputs = original_inputs(folder, manifest)
    if bindings(inputs) != frozen["original_inputs"]:
        raise ValueError("original fits/partial outputs/abort receipts changed")
    return manifest, inputs


def complete(folder, rscript):
    manifest, inputs = verify(folder)
    caps = Caps(**manifest["caps"])
    receipts, destination = folder / "diagnostic-executions", folder / "diagnostic-analysis"
    declared = dict(inputs, **{"diagnostic-freeze": folder / "diagnostic-freeze.json"})
    analysis_args = [
        sys.executable,
        str(run.ROOT / "scripts/report_lmm_stopping_control.py"),
        str(folder),
        str(destination),
    ]
    outputs = {
        name: destination / name for name in ("summary.json", "audit.jsonl.gz", "flags.jsonl")
    }
    result = run_capped(
        analysis_args, caps, folder, receipts, "diagnostic-analysis", declared, outputs
    )
    if result["exit_codes"] != [0] or acceptance_failures(result, caps.record()):
        raise ValueError("retained failed diagnostic analysis")
    verify(folder)
    summary = json.loads(outputs["summary.json"].read_bytes())
    check = destination / "r-check.json"
    r_args = [
        str(rscript),
        "--vanilla",
        str(run.ROOT / "comparisons/scientific/lmm_random_slope_stream_check.R"),
        str(folder / "manifest.json"),
        str(destination),
        str(check),
    ]
    r_inputs, r_outputs = dict(declared, **outputs), {"r-check.json": check}
    result = run_capped(r_args, caps, folder, receipts, "diagnostic-r-check", r_inputs, r_outputs)
    if result["exit_codes"] != [0] or acceptance_failures(result, caps.record()):
        raise ValueError("retained failed diagnostic R check")
    verify(folder)
    analysis_audit = audit_command(
        folder, receipts, "diagnostic-analysis", analysis_args, declared, outputs, caps
    )
    r_audit = audit_command(
        folder, receipts, "diagnostic-r-check", r_args, r_inputs, r_outputs, caps
    )
    run.write_new(
        folder / "diagnostic-assessment.json",
        {
            "scope": "completed diagnostic analysis; original failed study not accepted",
            "original_engineering_accepted": False,
            "numerical_ready": False,
            "confirmation_claim": False,
            "diagnostic_execution_accepted": True,
            "guards": summary["guards"],
            "manifest_sha256": file_hash(folder / "manifest.json"),
            "diagnostic_freeze_sha256": file_hash(folder / "diagnostic-freeze.json"),
            "analysis_command": analysis_audit,
            "r_command": r_audit,
            "analysis_outputs": bindings(outputs),
            "r_outputs": bindings(r_outputs),
            "original_inputs_unchanged": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "complete"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rscript", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.folder)
    else:
        if args.rscript is None:
            parser.error("--rscript required for complete")
        complete(args.folder, args.rscript)
