"""Command-bound capped analysis and independent R check for the repair pilot."""

import argparse
import json
import sys
from pathlib import Path

import lmm_random_slope_repair as run
from complete_lmm_scaling import audit_command, storage
from lmm_resource_caps import Caps, acceptance_failures, bindings, file_hash, run_capped


def complete(folder, rscript):
    manifest = run.verify(folder)
    if manifest["scenario"] != "repair_pilot":
        raise ValueError("analysis requires repair-pilot scenario")
    caps, receipts = Caps(**manifest["caps"]), folder / "analysis-executions"
    destination = folder / "analysis"
    analysis_args = [
        sys.executable,
        str(run.ROOT / "scripts/report_lmm_random_slope_repair.py"),
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
            "engineering_accepted": True,
            "numerical_ready": (
                not any(summary["guards"].values())
                and not any(row["failed"] for row in summary["counts"])
                and not any(
                    value
                    for key, value in summary["paired_comparisons"].items()
                    if key.endswith(("disagreement", "flagged_pairs"))
                )
            ),
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
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rscript", type=Path, required=True)
    complete(**vars(parser.parse_args()))
