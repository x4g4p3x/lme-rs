"""Post-study refits of retained pilot inputs; never replaces a frozen outcome."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import lmm_random_slope as pilot
from lmm_random_slope_optimum import optimum


def prepare(source, destination, wheel, environment):
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("diagnostic output must be separate from the original study")
    if destination.exists():
        raise FileExistsError(destination)
    manifest = pilot.verify_run(source)
    summary = pilot.load(source / "summary.json")
    for name, expected in summary["artifacts"].items():
        if pilot.shared.digest(source / name) != expected:
            raise ValueError("original outcome differs from retained summary")
    if environment["native_sha256"] == manifest["freeze"]["environment"]["native_sha256"]:
        raise ValueError("diagnostic requires a separately identified candidate")
    with zipfile.ZipFile(wheel) as archive:
        entries = [n for n in archive.namelist() if n.endswith((".pyd", ".so"))]
        if (
            len(entries) != 1
            or hashlib.sha256(archive.read(entries[0])).hexdigest() != environment["native_sha256"]
        ):
            raise ValueError("wheel and installed native candidate differ")
    destination.mkdir(parents=True)
    hashes = {}
    for name in sorted(
        set(manifest["freeze"]["sources"])
        | {"scripts/diagnose_lmm_random_slope_fix.py", "scripts/lmm_random_slope_optimum.py"}
    ):
        target = destination / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pilot.ROOT / name, target)
        hashes[name] = pilot.shared.digest(target)
    shutil.copyfile(wheel, destination / wheel.name)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=pilot.ROOT)
    (destination / "working-tree.patch").write_bytes(patch)
    metadata = {
        "scope": "post-study diagnostic refits; selected old inputs, not new calibration evidence",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pilot.ROOT, text=True
        ).strip(),
        "sources": hashes,
        "patch_sha256": pilot.shared.digest(destination / "working-tree.patch"),
        "environment": environment,
        "wheel": {"name": wheel.name, "sha256": pilot.shared.digest(wheel)},
        "source_manifest_sha256": pilot.shared.digest(source / "manifest.json"),
        "source_summary_sha256": pilot.shared.digest(source / "summary.json"),
        "control": pilot.CONTROL,
        "gates": pilot.GATES,
        "thread_limits": pilot.shared.THREAD_LIMITS,
    }
    pilot.shared.write_json(destination / "candidate.json", metadata)
    return manifest, metadata


def run(source, destination, wheel):
    os.environ.update(pilot.shared.THREAD_LIMITS)
    import lme_python as lme
    import polars as pl

    manifest, metadata = prepare(source, destination, wheel, pilot.engine_environment())
    counts = Counter()
    records = []
    worst = None
    for case in pilot.load(source / "designs.json"):
        rows = pilot.ordered(
            source / "cases" / case["id"] / "inputs.jsonl.gz", case["id"], manifest["replicates"]
        )
        for row in rows:
            data = pl.DataFrame(
                {"x": case["x"], "y": row["y"], "group": [f"g{i:03}" for i in case["group"]]}
            )
            fit = pilot.candidate_fit(lme, data)
            record = {"case": case["id"], "replicate": row["replicate"], "candidate": fit}
            counts["attempted"] += 1
            counts["converged"] += fit["converged"]
            counts["singular"] += fit.get("covariance_diagnostic", {}).get("state") == "singular"
            for method in pilot.METHODS:
                counts[f"{method}_errors"] += "error" in fit["methods"].get(
                    method, {"error": "missing"}
                )
            if len(set(case["sizes"])) == 1:
                m = case["sizes"][0]
                x = case["x"][:m]
                if any(case["x"][i : i + m] != x for i in range(0, len(case["x"]), m)):
                    raise ValueError("balanced oracle requires common within-group predictor")
                expected = optimum(x, [row["y"][i : i + m] for i in range(0, len(row["y"]), m)])
                counts["balanced"] += 1
                if fit.get("criterion") is not None:
                    gap = fit["criterion"] - expected["criterion"]
                    atol, rtol = pilot.GATES["criterion"]
                    limit = atol + rtol * abs(expected["criterion"])
                    counts["above_optimum"] += gap > limit
                    counts["below_oracle"] += gap < -limit
                    record["oracle"] = expected
                    record["criterion_gap"] = gap
                    record["criterion_limit"] = limit
                    item = {"case": case["id"], "replicate": row["replicate"], "gap": gap}
                    if worst is None or gap > worst["gap"]:
                        worst = item
                else:
                    counts["oracle_unevaluable"] += 1
            records.append(record)
        print(case["id"], flush=True)
    pilot.shared.write_gzip_jsonl(destination / "records.jsonl.gz", records)
    # Recheck every original hash after refitting; the diagnostic has no write
    # path into the source tree. Keep this separate from the frozen v1 report.
    pilot.verify_run(source)
    if pilot.shared.digest(source / "summary.json") != metadata["source_summary_sha256"]:
        raise ValueError("original summary changed during diagnostic")
    for name, expected in pilot.load(source / "summary.json")["artifacts"].items():
        if pilot.shared.digest(source / name) != expected:
            raise ValueError("original outcomes changed during diagnostic")
    if counts["attempted"] != manifest["attempted"]:
        raise ValueError("incomplete diagnostic")
    pilot.shared.write_json(
        destination / "summary.json",
        {
            "scope": metadata["scope"],
            "counts": dict(counts),
            "worst_criterion_gap": worst,
            "candidate_sha256": pilot.shared.digest(destination / "candidate.json"),
            "records_sha256": pilot.shared.digest(destination / "records.jsonl.gz"),
            "original_study_unchanged": True,
        },
    )


def audit(source, destination):
    """Read-only comparison with retained R fits and the balanced covariance oracle."""
    output = destination / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = pilot.verify_run(source)
    summary = pilot.load(destination / "summary.json")
    for name, key in [
        ("candidate.json", "candidate_sha256"),
        ("records.jsonl.gz", "records_sha256"),
    ]:
        if pilot.shared.digest(destination / name) != summary[key]:
            raise ValueError("diagnostic artifact changed")
    metadata = pilot.load(destination / "candidate.json")
    for name, key in [
        ("manifest.json", "source_manifest_sha256"),
        ("summary.json", "source_summary_sha256"),
    ]:
        if pilot.shared.digest(source / name) != metadata[key]:
            raise ValueError("original study identity differs")
    for name, expected in pilot.load(source / "summary.json")["artifacts"].items():
        if pilot.shared.digest(source / name) != expected:
            raise ValueError("original outcome changed")
    references = {}
    for case in pilot.load(source / "designs.json"):
        for pair in pilot.ordered(
            source / "cases" / case["id"] / "paired.jsonl",
            case["id"],
            manifest["replicates"],
            False,
        ):
            references[case["id"], pair["replicate"]] = pair["reference"]
    parity_flags, variance_flags = [], []
    rejection_disagreements = Counter({m: 0 for m in pilot.METHODS})
    reference_above_optimum = 0
    seen = set()
    for row in pilot.shared.gzip_lines(destination / "records.jsonl.gz"):
        key = row["case"], row["replicate"]
        if key in seen or key not in references:
            raise ValueError("duplicate or unknown diagnostic record")
        seen.add(key)
        actual, reference = row["candidate"], references[key]
        failed = [v for v in pilot.parity(actual, reference) if not v["passed"]]
        if failed:
            parity_flags.append({"case": key[0], "replicate": key[1], "failed": failed})
        for method in pilot.METHODS:
            a, b = actual["methods"][method], reference["methods"][method]
            if "error" in a or "error" in b:
                raise ValueError("audit needs evaluable method records")
            rejection_disagreements[method] += (a["p"] < 0.05) != (b["p"] < 0.05)
        if "oracle" in row:
            oracle = row["oracle"]
            scale = max(
                1,
                actual["sigma2"],
                oracle["sigma2"],
                sum(actual["covariance"][i][i] for i in range(2)),
                sum(oracle["covariance"][i][i] for i in range(2)),
            )
            atol, rtol = pilot.GATES["variance"]
            values = [("sigma2", actual["sigma2"], oracle["sigma2"])] + [
                (f"covariance_{i}{j}", actual["covariance"][i][j], oracle["covariance"][i][j])
                for i, j in [(0, 0), (0, 1), (1, 1)]
            ]
            failed = [
                {"metric": name, "difference": a - b, "limit": atol * scale + rtol * abs(b)}
                for name, a, b in values
                if abs(a - b) > atol * scale + rtol * abs(b)
            ]
            if failed:
                variance_flags.append({"case": key[0], "replicate": key[1], "failed": failed})
            reference_above_optimum += (
                reference["criterion"] - oracle["criterion"] > row["criterion_limit"]
            )
    if seen != set(references):
        raise ValueError("missing diagnostic records")
    result = {
        "scope": "post-study audit; retained R outcomes, no R refits, no substitution into v1",
        "summary_sha256": pilot.shared.digest(destination / "summary.json"),
        "audit_source_sha256": pilot.shared.digest(Path(__file__)),
        "paired": len(seen),
        "reference_above_balanced_optimum": reference_above_optimum,
        "rejection_disagreements": dict(rejection_disagreements),
        "R_parity_flagged_pairs": len(parity_flags),
        "R_parity_flags": parity_flags,
        "balanced_oracle_variance_flagged_fits": len(variance_flags),
        "balanced_oracle_variance_flags": variance_flags,
    }
    pilot.shared.write_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit(args.source, args.output)
    else:
        if args.wheel is None:
            parser.error("refits require --wheel")
        run(args.source, args.output, args.wheel)
