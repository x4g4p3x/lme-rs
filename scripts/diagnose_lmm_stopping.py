"""Independent high-precision balanced likelihood for retained stopping diagnostics.

Post-study evidence only. Never reads or writes a study outcome database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import struct
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_BUDGET_DIAGNOSTIC.json"
SENSITIVITY_CONTROL = {"tolerance": 1e-10, "max_iterations": 5000, "require_convergence": False}
PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(path):
    fixture = json.loads(FIXTURE.read_bytes())["fixture"]

    def bits(values):
        return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]

    write_new(
        path,
        {
            "fixture_sha256": digest(FIXTURE),
            "x_bits": bits(fixture["case"]["x"]),
            "y_bits": bits(fixture["y"]),
            "theta_bits": bits(fixture["candidate"]["theta"]),
            "group": fixture["case"]["group"],
        },
    )


class BalancedLikelihood:
    """Group projections and 2x2 covariance algebra; independent of Rust solves."""

    def __init__(self, case, y):
        sizes = case["sizes"]
        if len(set(sizes)) != 1 or sizes[0] <= 2 or len(sizes) <= 1:
            raise ValueError("requires balanced groups with G>1 and m>2")
        self.g, self.m = len(sizes), sizes[0]
        if len(y) != self.g * self.m:
            raise ValueError("response size mismatch")
        x = case["x"][: self.m]
        if any(case["x"][i : i + self.m] != x for i in range(0, len(y), self.m)):
            raise ValueError("requires common within-group x")
        expected_groups = [i for i in range(self.g) for _ in range(self.m)]
        if case["group"] != expected_groups:
            raise ValueError("requires ordered contiguous groups")
        x = [Decimal.from_float(v) for v in x]
        # Recenter in high precision: the generated double predictor has a tiny
        # nonzero mean. Transform its covariance too; do not assume exact centering.
        self.x_mean = sum(x) / self.m
        x = [v - self.x_mean for v in x]
        self.sxx = sum(v * v for v in x)
        if self.sxx <= 0:
            raise ValueError("requires full-rank predictor")
        rows = [list(map(Decimal.from_float, y[i : i + self.m])) for i in range(0, len(y), self.m)]
        coefficients = [
            (sum(row) / self.m, sum(a * b for a, b in zip(x, row)) / self.sxx) for row in rows
        ]
        means = [sum(c[j] for c in coefficients) / self.g for j in (0, 1)]
        self.ssw = sum(
            (v - c[0] - c[1] * t) ** 2 for row, c in zip(rows, coefficients) for t, v in zip(x, row)
        )
        self.scatter = [
            sum((c[i] - means[i]) * (c[j] - means[j]) for c in coefficients)
            for i, j in ((0, 0), (0, 1), (1, 1))
        ]

    def criterion(self, theta):
        a, b, c = [v if isinstance(v, Decimal) else Decimal.from_float(v) for v in theta]
        # Relative V = I + Z Lambda Lambda' Z'; work in orthonormal group projections.
        shifted_a = a + self.x_mean * b
        shifted_c = self.x_mean * c
        aa = 1 + self.m * (shifted_a * shifted_a + shifted_c * shifted_c)
        bb = (self.m * self.sxx).sqrt() * (shifted_a * b + shifted_c * c)
        dd = 1 + self.sxx * (b * b + c * c)
        det = aa * dd - bb * bb
        s00, s01, s11 = self.scatter
        rss = (
            self.ssw
            + (dd * self.m * s00 - 2 * bb * (self.m * self.sxx).sqrt() * s01 + aa * self.sxx * s11)
            / det
        )
        df = self.g * self.m - 2
        return (
            df * ((2 * PI * rss / df).ln() + 1)
            + (self.g - 1) * det.ln()
            + (self.g * self.g * self.m * self.sxx).ln()
        )

    def interior_optimum(self):
        sigma = self.ssw / (self.g * (self.m - 2))
        s00, s01, s11 = self.scatter
        v00 = s00 / (self.g - 1) - sigma / self.m
        v01 = s01 / (self.g - 1)
        v11 = s11 / (self.g - 1) - sigma / self.sxx
        v00, v01 = v00 - 2 * self.x_mean * v01 + self.x_mean**2 * v11, v01 - self.x_mean * v11
        if v00 <= 0 or v00 * v11 <= v01 * v01:
            raise ValueError("this diagnostic's analytic optimum must be interior")
        a = (v00 / sigma).sqrt()
        b = v01 / sigma / a
        c = (v11 / sigma - b * b).sqrt()
        return [a, b, c]


def analyze(raw_path, output):
    source = json.loads(FIXTURE.read_bytes())
    fixture = source["fixture"]
    raw = json.loads(Path(raw_path).read_bytes())
    retained = fixture["candidate"]["theta"]
    references = []
    for precision in (60, 80):
        with localcontext() as context:
            context.prec = precision
            likelihood = BalancedLikelihood(fixture["case"], fixture["y"])
            optimum = likelihood.interior_optimum()
            best = likelihood.criterion(optimum)
            retained_cost = likelihood.criterion(retained)
            neighbors = [
                {
                    **row,
                    "independent_criterion": str(likelihood.criterion(row["theta"])),
                    "independent_gap": str(likelihood.criterion(row["theta"]) - best),
                }
                for row in raw["neighbors"]
            ]
            references.append(
                {
                    "decimal_digits": precision,
                    "criterion": str(retained_cost),
                    "optimum_criterion": str(best),
                    "gap": str(retained_cost - best),
                    "optimum_theta": list(map(str, optimum)),
                    "neighbors": neighbors,
                }
            )
    evaluations = raw["argmin_evaluations"]
    tail = evaluations[-200:]
    cost = fixture["candidate"]["criterion"]
    summary = {
        "scope": "post-study stopping diagnostic; no pilot replacement or calibration claim",
        "fixture_sha256": digest(FIXTURE),
        "raw_sha256": digest(raw_path),
        "analyzer_sha256": digest(__file__),
        "retained_candidate": source["candidate"],
        "public_fits": raw["public_fits"],
        "argmin_checkpoints": raw["argmin_checkpoints"],
        "argmin_final": raw["argmin_final"],
        "evaluations": len(evaluations),
        "last_200": {
            "distinct_theta": len({tuple(r["theta"]) for r in tail}),
            "distinct_costs": sorted({r["cost"] for r in tail}),
            "theta_min": [min(r["theta"][i] for r in tail) for i in range(3)],
            "theta_max": [max(r["theta"][i] for r in tail) for i in range(3)],
        },
        "repeats": {
            "count": len(raw["repeats"]),
            "distinct_cached": sorted({r["cached"] for r in raw["repeats"]}),
            "distinct_full": sorted({r["full"] for r in raw["repeats"]}),
            "cached_after_neighbors": raw["repeat_after_neighbors"],
        },
        "objective_ulp": math.ulp(cost),
        "independent": references,
    }
    write_new(output, summary)


def sensitivity_worker(source, output, index):
    declaration = json.loads((output / "sensitivity-freeze.json").read_bytes())
    verify_runtime(declaration)
    import lme_python as lme
    import lmm_random_slope_v2 as pilot
    import polars as pl
    from lmm_stream_store import canonical, sha, unpack

    if declaration["control"] != SENSITIVITY_CONTROL:
        raise ValueError("diagnostic control changed")
    if declaration["script_sha256"] != digest(__file__):
        raise ValueError("diagnostic script changed")
    if digest(source / "manifest.json") != declaration["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    manifest = json.loads((source / "manifest.json").read_bytes())
    if pilot.engine_environment() != manifest["candidate"]:
        raise ValueError("installed engine differs from retained source")
    if any(digest(ROOT / p) != h for p, h in manifest["engine_revision"]["sources"].items()):
        raise ValueError("engine source changed")
    counts = {"attempts": 0, "outcomes": 0, "converged": 0, "balanced": 0, "balanced_pass": 0}
    # Exclusive creation, flushed start before every fit, no resume/retry semantics.
    with (output / f"sensitivity-{index}.jsonl").open("x", encoding="utf-8") as stream:

        def persist(record):
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        for case in manifest["cases"][index::4]:
            journal = source / "cases" / case["id"] / "journal.sqlite"
            wal = journal.with_name("journal.sqlite-wal")
            if (
                digest(journal) != declaration["journals"][case["id"]]
                or wal.exists()
                and wal.stat().st_size
            ):
                raise ValueError("source journal changed")
            with sqlite3.connect(
                journal.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
            ) as db:
                seen = 0
                for begin, n, payload, checksum in db.execute(
                    "SELECT * FROM chunks ORDER BY first"
                ):
                    rows = unpack(payload, checksum)
                    if begin != seen or len(rows) != n:
                        raise ValueError("source chunk inventory differs")
                    for row in rows:
                        key = {"case": case["id"], "replicate": seen}
                        indexed = db.execute(
                            "SELECT chunk,sha FROM inputs WHERE replicate=?", (seen,)
                        ).fetchone()
                        if (
                            row["case"] != case["id"]
                            or row["replicate"] != seen
                            or indexed != (begin, sha(canonical(row)))
                        ):
                            raise ValueError("source input identity differs")
                        persist({**key, "phase": "start", "input_sha256": indexed[1]})
                        counts["attempts"] += 1
                        record = {**key, "phase": "outcome"}
                        try:
                            data = pl.DataFrame(
                                {
                                    "x": case["x"],
                                    "y": row["y"],
                                    "group": [f"g{i:03}" for i in case["group"]],
                                }
                            )
                            fit = lme.lmer(
                                "y ~ x + (1 + x | group)",
                                data,
                                reml=True,
                                control=lme.FitControl(**declaration["control"]),
                            )
                            candidate = {
                                "converged": fit.converged is True,
                                "beta": fit.coefficients[1],
                                "sigma2": fit.sigma2,
                                "criterion": fit.reml_criterion,
                                "theta": fit.theta,
                                "covariance": pilot.candidate_covariance(fit.theta, fit.sigma2),
                                "diagnostics": fit.diagnostics,
                            }
                            check = pilot.balanced_checks(case, row["y"], candidate)
                            record.update(candidate=candidate, numerical=check)
                            counts["converged"] += candidate["converged"]
                            if check is not None:
                                counts["balanced"] += 1
                                counts["balanced_pass"] += bool(
                                    check["evaluable"]
                                    and check["rank_match"]
                                    and all(c["passed"] for c in check["checks"])
                                )
                        except Exception as error:
                            record["error"] = f"{type(error).__name__}: {error}"
                        persist(record)
                        counts["outcomes"] += 1
                        seen += 1
                if seen != 20 or db.execute("SELECT count(*) FROM inputs").fetchone()[0] != seen:
                    raise ValueError("source replicate inventory differs")
            if digest(journal) != declaration["journals"][case["id"]]:
                raise ValueError("source journal changed during diagnostic")
    write_new(output / f"sensitivity-{index}-counts.json", counts)


def sensitivity(source, output):
    from concurrent.futures import ProcessPoolExecutor

    if source.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("diagnostic output must be separate from original study")
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(sensitivity_worker, source, output, index) for index in range(4)]
        for future in futures:
            future.result()
    totals = {}
    for index in range(4):
        for name, count in json.loads(
            (output / f"sensitivity-{index}-counts.json").read_bytes()
        ).items():
            totals[name] = totals.get(name, 0) + count
    if totals["attempts"] != 1880 or totals["outcomes"] != 1880:
        raise ValueError("incomplete diagnostic attempts/outcomes")
    write_new(output / "sensitivity-counts.json", totals)


def freeze_sensitivity(source, output):
    if source.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError("diagnostic output must be separate from original study")
    manifest = json.loads((source / "manifest.json").read_bytes())
    public = json.loads((ROOT / "scientific/RANDOM_SLOPE_REPAIR_PILOT_V1_FREEZE.json").read_bytes())
    if manifest != public or len(manifest["cases"]) != 94:
        raise ValueError("source does not match published pilot")
    summary_path = ROOT / "scientific/RANDOM_SLOPE_REPAIR_PILOT_V1.json"
    summary = json.loads(summary_path.read_bytes())
    if digest(source / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("source manifest bytes differ from published summary")
    journals = {r["case"]: r["journal_sha256"] for r in summary["journal_inventory"]}
    if set(journals) != {c["id"] for c in manifest["cases"]}:
        raise ValueError("published journal inventory differs")
    for case, expected in journals.items():
        journal = source / "cases" / case / "journal.sqlite"
        wal = journal.with_name("journal.sqlite-wal")
        if digest(journal) != expected or wal.exists() and wal.stat().st_size:
            raise ValueError("source journal differs from published evidence or has active WAL")
    runtime = {p: h for p, h in manifest["sources"].items() if p.startswith("scripts/")}
    verify_runtime({"runtime_sources": runtime})
    write_new(
        output / "sensitivity-freeze.json",
        {
            "scope": "post-study sensitivity on reused inputs; no inference or calibration claim",
            "control": SENSITIVITY_CONTROL,
            "script_sha256": digest(__file__),
            "source_manifest_sha256": digest(source / "manifest.json"),
            "public_summary_sha256": digest(summary_path),
            "runtime_sources": runtime,
            "journals": journals,
        },
    )
    shutil.copyfile(__file__, output / "diagnose_lmm_stopping.py")
    for name in runtime:
        target = output / "runtime-sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)


def verify_runtime(declaration):
    for name, expected in declaration["runtime_sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"runtime helper changed: {name}")


def launch_sensitivity(source, output, candidate_python):
    verify_runtime(json.loads((output / "sensitivity-freeze.json").read_bytes()))
    from lmm_resource_caps import Caps, acceptance_failures, run_capped

    result = run_capped(
        [
            str(candidate_python.resolve()),
            str(Path(__file__).resolve()),
            "sensitivity",
            str(source.resolve()),
            str(output.resolve()),
        ],
        Caps(),
        output.resolve(),
        output.resolve() / "executions",
        "sensitivity",
        input_files={
            "declaration": output / "sensitivity-freeze.json",
            "script": Path(__file__),
            "source_manifest": source / "manifest.json",
        },
        output_files={"counts": output / "sensitivity-counts.json"},
    )
    failures = acceptance_failures(result, Caps().record())
    if failures or result["exit_codes"] != [0]:
        raise ValueError(f"diagnostic execution rejected: {failures}; exits={result['exit_codes']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("output", type=Path)
    p = subs.add_parser("freeze-sensitivity")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p = subs.add_parser("launch-sensitivity")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("candidate_python", type=Path)
    p = subs.add_parser("sensitivity")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p = subs.add_parser("analyze")
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "analyze":
        analyze(args.raw, args.output)
    elif args.command == "freeze-sensitivity":
        freeze_sensitivity(args.source, args.output)
    elif args.command == "launch-sensitivity":
        launch_sensitivity(args.source, args.output, args.candidate_python)
    else:
        sensitivity(args.source, args.output)


if __name__ == "__main__":
    main()
