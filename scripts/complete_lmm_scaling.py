"""Freeze a three-run bank or execute and verify separately capped analyses."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import lmm_random_slope_scaling as run
from lmm_resource_caps import (
    Caps,
    acceptance_failures,
    bindings,
    command_identity,
    file_hash,
    run_capped,
)


def audit_command(folder, receipt_folder, label, args, inputs, outputs, caps):
    starts = sorted(receipt_folder.glob(label + "-*-start.json"))
    if len(starts) != 1:
        raise ValueError("analysis commands must have exactly one retained attempt")
    start_path = starts[0]
    start = json.loads(start_path.read_text())
    end_path = start_path.with_name(start_path.name.removesuffix("-start.json") + "-end.json")
    end = json.loads(end_path.read_text())
    identity = {
        "caps": caps.record(),
        "label": label,
        "command": command_identity(args),
        "inputs": bindings(inputs),
        "thread_limits": run.old.pilot.shared.THREAD_LIMITS,
    }
    if (
        start != identity
        or end.get("identity") != identity
        or end.get("start_sha256") != file_hash(start_path)
        or end.get("outputs") != bindings(outputs)
        or end.get("exit_codes") != [0]
    ):
        raise ValueError("command/input/output/thread provenance differs")
    failures = acceptance_failures(end, caps.record())
    if failures:
        raise ValueError("capped command rejected: " + "; ".join(failures))
    return {
        "start": {
            "path": start_path.relative_to(folder).as_posix(),
            "sha256": file_hash(start_path),
        },
        "end": {"path": end_path.relative_to(folder).as_posix(), "sha256": file_hash(end_path)},
    }


def storage(folder):
    totals = {
        "journal_physical_bytes": 0,
        "compressed_input_payload_bytes": 0,
        "compressed_outcome_payload_bytes": 0,
    }
    for path in sorted((folder / "cases").glob("*/journal.sqlite")):
        db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            totals["journal_physical_bytes"] += path.stat().st_size
            totals["compressed_input_payload_bytes"] += db.execute(
                "SELECT coalesce(sum(length(payload)),0) FROM chunks"
            ).fetchone()[0]
            totals["compressed_outcome_payload_bytes"] += db.execute(
                "SELECT coalesce(sum(length(payload)),0) FROM outcomes"
            ).fetchone()[0]
        finally:
            db.close()
    totals["physical_minus_compressed_payload_bytes"] = (
        totals["journal_physical_bytes"]
        - totals["compressed_input_payload_bytes"]
        - totals["compressed_outcome_payload_bytes"]
    )
    totals["overhead_scope"] = (
        "includes pages, unused space, indexes and metadata; not pure fixed cost"
    )
    return totals


def complete(folder, rscript):
    manifest = run.verify(folder)
    run.verify_bank(folder.parent)
    if manifest["scenario"] != "scaling":
        raise ValueError("analysis requires scaling scenario")
    caps, receipts = Caps(**manifest["caps"]), folder / "analysis-executions"
    destination = folder / "analysis"
    analysis_args = [
        sys.executable,
        str(run.ROOT / "scripts/report_lmm_random_slope_scaling.py"),
        str(folder),
        str(destination),
    ]
    analysis_inputs = {"manifest": folder / "manifest.json"}
    analysis_inputs.update(
        {
            "journal:" + case["id"]: folder / "cases" / case["id"] / "journal.sqlite"
            for case in manifest["cases"]
        }
    )
    analysis_outputs = {
        name: destination / name for name in ("summary.json", "audit.jsonl.gz", "flags.jsonl")
    }
    first = run_capped(
        analysis_args, caps, folder, receipts, "analysis", analysis_inputs, analysis_outputs
    )
    if acceptance_failures(first, caps.record()) or first["exit_codes"] != [0]:
        raise ValueError("retained failed capped analysis")
    summary = json.loads((destination / "summary.json").read_text())
    if not summary["execution_acceptance"]["accepted"]:
        raise ValueError("fit execution rejected")
    check = destination / "r-check.json"
    r_args = [
        str(rscript),
        "--vanilla",
        str(run.ROOT / "comparisons/scientific/lmm_random_slope_stream_check.R"),
        str(folder / "manifest.json"),
        str(destination),
        str(check),
    ]
    r_inputs = dict(analysis_inputs, **analysis_outputs)
    r_outputs = {"r-check.json": check}
    second = run_capped(r_args, caps, folder, receipts, "r-check", r_inputs, r_outputs)
    if acceptance_failures(second, caps.record()) or second["exit_codes"] != [0]:
        raise ValueError("retained failed capped R verification")
    analysis_audit = audit_command(
        folder, receipts, "analysis", analysis_args, analysis_inputs, analysis_outputs, caps
    )
    r_audit = audit_command(folder, receipts, "r-check", r_args, r_inputs, r_outputs, caps)
    run.write_new(
        folder / "acceptance.json",
        {
            "accepted": True,
            "confirmation_claim": False,
            "manifest_sha256": file_hash(folder / "manifest.json"),
            "analysis_command": analysis_audit,
            "r_command": r_audit,
            "analysis_outputs": bindings(analysis_outputs),
            "r_outputs": bindings(r_outputs),
            "storage": storage(folder),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("folder", type=Path)
    for name in ("anchor", "candidate-python", "wheel"):
        freeze.add_argument("--" + name, type=Path, required=True)
    freeze.add_argument("--namespace", required=True)
    finish = commands.add_parser("complete")
    finish.add_argument("folder", type=Path)
    finish.add_argument("--rscript", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"freeze": run.freeze_bank, "complete": complete}[command](**args)
