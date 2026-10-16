"""Selected-case numerical-controls investigation, never replacement pilot results."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import lmm_random_slope as pilot
from lmm_random_slope_optimum import optimum

SELECTED = (
    ("zero_slope-G8-m5-v0-rundefined-beta0.5", 17),
    ("imbalanced-G8-mcycle3-5-10-v0.25-r0-beta0", 18),
)
CONTROLS = tuple(
    {"tolerance": t, "max_iterations": 5000, "require_convergence": False}
    for t in (1e-8, 1e-12, 1e-14)
)


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("selected diagnostic must be separate from the original study")
    if destination.exists():
        raise FileExistsError(destination)
    manifest = pilot.verify_run(source)
    for name, expected in pilot.load(source / "summary.json")["artifacts"].items():
        if pilot.shared.digest(source / name) != expected:
            raise ValueError("original outcome changed")
    cases = {c["id"]: c for c in pilot.load(source / "designs.json")}
    inputs = []
    for name, replicate in SELECTED:
        case = cases[name]
        folder = source / "cases" / name
        rows = pilot.ordered(folder / "inputs.jsonl.gz", name, manifest["replicates"])
        paired = pilot.ordered(folder / "paired.jsonl", name, manifest["replicates"], False)
        inputs.append(
            {
                "case": case,
                "replicate": replicate,
                "y": rows[replicate]["y"],
                "retained_reference": paired[replicate]["reference"],
            }
        )
    destination.mkdir(parents=True)
    pilot.shared.write_json(destination / "inputs.json", inputs)
    return manifest, inputs


def run(source, destination):
    os.environ.update(pilot.shared.THREAD_LIMITS)
    import lme_python as lme
    import polars as pl

    environment = pilot.engine_environment()
    baseline = pilot.load(pilot.ROOT / "scientific/RANDOM_SLOPE_BOUNDARY_FIX_DIAGNOSTIC.json")
    if environment != baseline["candidate"]["environment"]:
        raise ValueError("candidate differs from the recorded boundary-fix binary")
    for name, expected in baseline["candidate"]["sources"].items():
        if (
            name.startswith(("src/", "python/src/"))
            and pilot.shared.digest(pilot.ROOT / name) != expected
        ):
            raise ValueError(f"engine source no longer matches diagnostic candidate: {name}")
    manifest, inputs = prepare(source, destination)
    pilot.shared.write_json(
        destination / "controls.json",
        {
            "scope": (
                "selected post-study controls and derivative investigation; no calibration claim"
            ),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=pilot.ROOT, text=True
            ).strip(),
            "environment": environment,
            "controls": CONTROLS,
            "selected": SELECTED,
            "inputs_sha256": pilot.shared.digest(destination / "inputs.json"),
            "original_manifest_sha256": pilot.shared.digest(source / "manifest.json"),
            "original_summary_sha256": pilot.shared.digest(source / "summary.json"),
            "script_sha256": pilot.shared.digest(Path(__file__)),
            "thread_limits": pilot.shared.THREAD_LIMITS,
        },
    )
    results = []
    for raw in inputs:
        case, y = raw["case"], raw["y"]
        data = pl.DataFrame({"x": case["x"], "y": y, "group": [f"g{i:03}" for i in case["group"]]})
        oracle = None
        if len(set(case["sizes"])) == 1:
            m = case["sizes"][0]
            oracle = optimum(case["x"][:m], [y[i : i + m] for i in range(0, len(y), m)])
        for control in CONTROLS:
            record = {
                "case": case["id"],
                "replicate": raw["replicate"],
                "control": control,
                "oracle": oracle,
            }
            try:
                fit = lme.lmer(
                    "y ~ x + (1 + x | group)", data, reml=True, control=lme.FitControl(**control)
                )
                covariance = pilot.candidate_covariance(fit.theta, fit.sigma2)
                record["candidate"] = {
                    "coefficients": fit.coefficients,
                    "sigma2": fit.sigma2,
                    "theta": fit.theta,
                    "covariance": covariance,
                    "criterion": fit.reml_criterion,
                    "converged": fit.converged,
                    "diagnostics": fit.diagnostics,
                    "covariance_diagnostic": pilot.covariance_diagnostic(covariance, fit.sigma2),
                    "methods": pilot.shared.evaluate_candidate_methods(
                        fit, data, fit.coefficients[1]
                    ),
                }
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                record["candidate"] = {"error": f"{type(error).__name__}: {error}"}
            results.append(pilot.shared.finite_json(record))
    pilot.verify_run(source)
    for name, expected in pilot.load(source / "summary.json")["artifacts"].items():
        if pilot.shared.digest(source / name) != expected:
            raise ValueError("original outcome changed during refits")
    if len(results) != len(SELECTED) * len(CONTROLS) or manifest["replicates"] <= max(
        r for _, r in SELECTED
    ):
        raise ValueError("incomplete selected diagnostic")
    pilot.shared.write_json(destination / "candidate.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
