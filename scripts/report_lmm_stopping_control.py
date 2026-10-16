"""Frozen fresh stopping-control pilot analysis; no confirmation claim."""

from __future__ import annotations

import argparse
import gzip
import math
import platform
import time
from collections import Counter
from pathlib import Path

import lmm_random_slope_v2 as pilot
import lmm_stopping_control as run
from lmm_resource_caps import acceptance_failures
from lmm_stream_store import Journal, canonical, case_lock, sha
from plan_lmm_random_slope_confirmation import cp_interval, require_scipy


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def fit_eligible(fit):
    if not isinstance(fit, dict):
        return False
    covariance = fit.get("covariance")
    if (
        not isinstance(covariance, list)
        or len(covariance) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in covariance)
        or not all(numeric(v) for row in covariance for v in row)
        or not numeric(fit.get("sigma2"))
    ):
        return False
    return (
        fit.get("converged") is True
        and all(numeric(fit.get(k)) for k in ("beta", "criterion"))
        and pilot.covariance_diagnostic(fit.get("covariance"), fit.get("sigma2"))["state"]
        != "invalid"
    )


def choose_reference(fits):
    eligible = [(i, f) for i, f in enumerate(fits) if fit_eligible(f)]
    return min(eligible, key=lambda item: (item[1]["criterion"], item[0]))[0] if eligible else None


def numerical(case, y, fit):
    check = pilot.balanced_checks(case, y, fit)
    passed = fit_eligible(fit) and (
        check is None
        or check["evaluable"]
        and check["rank_match"]
        and all(c["passed"] for c in check["checks"])
    )
    return check, passed


def decision(fit, method, truth, numerical_pass):
    """Independent distribution, scalar-contrast and p/CI consistency checks."""
    from scipy.stats import f, norm, t

    if not numerical_pass:
        return None
    methods = fit.get("methods")
    value = methods.get(method, {}) if isinstance(methods, dict) else {}
    if not isinstance(value, dict):
        return None
    if "error" in value or any(not numeric(value.get(k)) for k in ("beta", "se", "f", "p")):
        return None
    beta, se, statistic, p = (value[k] for k in ("beta", "se", "f", "p"))
    if (
        se <= 0
        or statistic < 0
        or not 0 <= p <= 1
        or not math.isclose(beta, fit["beta"], rel_tol=1e-12, abs_tol=1e-12)
    ):
        return None
    ratio = beta / se
    if not math.isclose(statistic, ratio * ratio, rel_tol=1e-8, abs_tol=1e-10):
        return None
    if method == "normal":
        critical, expected_p = float(norm.ppf(0.975)), math.erfc(abs(beta / se) / math.sqrt(2))
    else:
        df = value.get("df")
        if not numeric(df) or df <= 0:
            return None
        critical, expected_p = float(t.ppf(0.975, df)), float(f.sf(statistic, 1, df))
    if not math.isfinite(critical) or critical <= 0 or abs(p - expected_p) > 1e-8:
        return None
    lower, upper = beta - critical * se, beta + critical * se
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    reject, cover = p < 0.05, lower <= truth <= upper
    if reject == (lower <= 0 <= upper):
        return None
    return {
        "reject": reject,
        "cover": cover,
        "error": beta - truth,
        "se": se,
        "width": upper - lower,
    }


def add(count, value):
    count["attempted"] += 1
    if value is None:
        count["failed"] += 1
    else:
        count["reject"] += value["reject"]
        count["cover"] += value["cover"]
        count["sum_error"] += value["error"]
        count["sum_squared_error"] += value["error"] ** 2
        count["sum_se"] += value["se"]
        count["sum_width"] += value["width"]


def summarize_count(count):
    result = {k: count[k] for k in ("attempted", "failed", "reject", "cover")}
    valid = result["attempted"] - result["failed"]
    result.update(
        {
            "bias": count["sum_error"] / valid if valid else None,
            "rmse": math.sqrt(count["sum_squared_error"] / valid) if valid else None,
            "mean_se": count["sum_se"] / valid if valid else None,
            "mean_width": count["sum_width"] / valid if valid else None,
        }
    )
    return result


def matched(iterator, index):
    try:
        key, value = next(iterator)
    except StopIteration as error:
        raise ValueError("missing fit outcome: incomplete execution") from error
    if key != index:
        raise ValueError("fit outcome inventory differs")
    return value


def exhausted(iterator):
    if next(iterator, None) is not None:
        raise ValueError("extra fit outcome")


def execution_audit(folder, manifest):
    failures, receipts = [], []
    phases = Counter()
    lock = pilot.load(run.ROOT / "scientific/renv.lock")
    expected_versions = {n: p["Version"] for n, p in lock["Packages"].items()}
    expected_versions.update(lock["LoadedNamespaces"])
    starts = sorted((folder / "executions").glob("*-start.json"))
    if not starts:
        failures.append("no execution receipts")
    for path in starts:
        start = pilot.load(path)
        prefix = path.name.removesuffix("-start.json")
        phase, workers = start.get("phase"), start.get("workers")
        if (
            phase not in ("candidate", "reference")
            or type(workers) is not int
            or not 1 <= workers <= 4
        ):
            raise ValueError("invalid execution receipt phase/count")
        phases[phase] += 1
        files = [path]
        if start.get("manifest_sha") != sha(canonical(manifest)):
            raise ValueError("execution manifest identity differs")
        end_path = path.parent / f"{prefix}-end.json"
        if not end_path.exists():
            failures.append(prefix + ": missing end receipt")
        else:
            files.append(end_path)
            if pilot.load(end_path)["exit_codes"] != [0] * workers:
                failures.append(prefix + ": failed/incomplete worker exits")
        for i in range(workers):
            worker_path = path.parent / f"{prefix}-worker-{i}.json"
            if not worker_path.exists():
                failures.append(prefix + f": missing worker {i} receipt")
                continue
            files.append(worker_path)
            worker = pilot.load(worker_path)
            if (
                worker.get("manifest_sha") != sha(canonical(manifest))
                or worker.get("phase") != phase
                or worker.get("worker") != i
                or type(worker.get("native_python_pid")) is not int
            ):
                raise ValueError("worker receipt identity differs")
            if (
                not numeric(worker.get("python_peak_working_set_bytes"))
                or worker["python_peak_working_set_bytes"] <= 0
            ):
                failures.append(prefix + f": missing native Python memory {i}")
            if phase == "reference":
                env = worker.get("r_environment")
                if not isinstance(env, dict) or type(env.get("native_pid")) is not int:
                    failures.append(prefix + f": missing R environment {i}")
                else:
                    if not env["r_version"].startswith("R version " + lock["R"]["Version"] + " "):
                        raise ValueError("R runtime version differs")
                    for name, version in env["namespaces"].items():
                        if name not in expected_versions or version != expected_versions[name]:
                            raise ValueError("R namespace version differs: " + name)
                    if any(
                        env["namespaces"].get(n) != v for n, v in lock["LoadedNamespaces"].items()
                    ):
                        raise ValueError("R locked namespace inventory missing")
                if (
                    not numeric(worker.get("r_peak_working_set_bytes"))
                    or worker["r_peak_working_set_bytes"] <= 0
                ):
                    failures.append(prefix + f": missing native R memory {i}")
        receipts.extend(
            {"path": p.relative_to(folder).as_posix(), "sha256": pilot.shared.digest(p)}
            for p in files
        )
    for phase in ("candidate", "reference"):
        if not phases[phase]:
            failures.append("no " + phase + " execution")
    return {"accepted": not failures, "failures": failures, "receipts": receipts}


def capped_execution_audit(folder, manifest):
    failures, receipts = [], []
    expected = manifest["caps"]
    for start_path in sorted((folder / "executions").glob("*-start.json")):
        start = pilot.load(start_path)
        end_path = start_path.with_name(start_path.name.removesuffix("-start.json") + "-end.json")
        if start.get("caps") != expected:
            raise ValueError("execution cap configuration differs")
        if not end_path.exists():
            failures.append("missing capped end receipt")
            continue
        end = pilot.load(end_path)
        failures.extend(acceptance_failures(end, expected))
        receipts.append(
            {
                "path": end_path.relative_to(folder).as_posix(),
                "sha256": pilot.shared.digest(end_path),
            }
        )
    if not receipts:
        failures.append("no capped execution receipts")
    return {
        "accepted": not failures,
        "failures": failures,
        "receipts": receipts,
        "scope": (
            "hard Windows job committed bytes and sampled phase/disk caps; "
            "not a whole-host RAM/disk quota"
        ),
    }


def report(folder, destination):
    require_scipy()
    if destination.exists():
        raise FileExistsError(destination)
    manifest = run.verify(folder)
    executions = execution_audit(folder, manifest)
    cap_receipts = capped_execution_audit(folder, manifest)
    if not cap_receipts["accepted"]:
        executions["accepted"] = False
        executions["failures"].extend(cap_receipts["failures"])
    destination.mkdir(parents=True)
    before = time.perf_counter()
    counts, primary, inventory = [], [], []
    guards, comparisons, known_counts = Counter(), Counter(), Counter()
    variants = {v["id"]: Counter() for v in run.VARIANTS}
    with (
        gzip.open(destination / "audit.jsonl.gz", "xb") as audit,
        (destination / "flags.jsonl").open("xb") as flags,
    ):
        for case in manifest["cases"]:
            path = folder / "cases" / case["id"]
            if not (path / "journal.sqlite").is_file():
                raise ValueError("missing case journal")
            with case_lock(path / "case.lock"):
                journal = Journal(path / "journal.sqlite", run.case_metadata(manifest, case))
                try:
                    journal.validate()
                    if journal.input_count() != journal.metadata["n"]:
                        raise ValueError("incomplete input inventory")
                    candidates = journal.iter_outcomes("candidate")
                    references = [
                        journal.iter_outcomes("reference:" + v["id"]) for v in run.VARIANTS
                    ]
                    local = {
                        (e, m): Counter() for e in ("candidate", "reference") for m in pilot.METHODS
                    }
                    for row in journal.iter_inputs():
                        i = row["replicate"]
                        candidate = matched(candidates, i)
                        check, valid = numerical(case, row["y"], candidate)
                        guards["candidate_unevaluable_or_numerically_flagged"] += not valid
                        guards["candidate_balanced_numerical_miss"] += bool(
                            check and fit_eligible(candidate) and not valid
                        )
                        decisions = {
                            m: decision(candidate, m, case["beta"], valid) for m in pilot.METHODS
                        }
                        for m, value in decisions.items():
                            add(local["candidate", m], value)
                        bank, selected, ref_check = [], None, None
                        if i < manifest["reference_n"]:
                            bank = [matched(stream, i) for stream in references]
                            for v, raw in zip(run.VARIANTS, bank):
                                state = variants[v["id"]]
                                state["attempted"] += 1
                                state["convergence_not_established"] += (
                                    raw.get("converged") is not True
                                )
                                error = raw.get("fit_error")
                                state["optimizer_nonconvergence"] += error == "nonconvergence"
                                state["interrupted"] += error == "interrupted_after_durable_start"
                                state["exception"] += bool(error) and error not in (
                                    "nonconvergence",
                                    "interrupted_after_durable_start",
                                )
                                state["fit_error"] += bool(raw.get("fit_error"))
                                state["invalid_fit"] += not fit_eligible(raw)
                                raw_check, raw_valid = numerical(case, row["y"], raw)
                                state["balanced_numerical_miss"] += bool(
                                    raw_check and fit_eligible(raw) and not raw_valid
                                )
                                failed_methods = [
                                    m
                                    for m in pilot.METHODS
                                    if decision(raw, m, case["beta"], raw_valid) is None
                                ]
                                for m in pilot.METHODS:
                                    state["method_failed:" + m] += m in failed_methods
                                if not raw_valid or failed_methods:
                                    flags.write(
                                        canonical(
                                            {
                                                "kind": "reference_variant",
                                                "case": case["id"],
                                                "replicate": i,
                                                "variant": v["id"],
                                                "numerical": raw_check,
                                                "failed_methods": failed_methods,
                                            }
                                        )
                                        + b"\n"
                                    )
                            selected = choose_reference(bank)
                            reference = bank[selected] if selected is not None else {}
                            ref_check, ref_valid = numerical(case, row["y"], reference)
                            guards["reference_unevaluable_or_numerically_flagged"] += not ref_valid
                            for m in pilot.METHODS:
                                d = decision(reference, m, case["beta"], ref_valid)
                                add(local["reference", m], d)
                                if d is not None and decisions[m] is not None:
                                    comparisons[m + ":paired"] += 1
                                    comparisons[m + ":reject_disagreement"] += (
                                        d["reject"] != decisions[m]["reject"]
                                    )
                                    comparisons[m + ":cover_disagreement"] += (
                                        d["cover"] != decisions[m]["cover"]
                                    )
                            if valid and ref_valid:
                                a = dict(
                                    candidate,
                                    methods={
                                        m: candidate["methods"][m]
                                        for m in pilot.METHODS
                                        if decisions[m] is not None
                                    },
                                )
                                b = dict(
                                    reference,
                                    methods={
                                        m: reference["methods"][m]
                                        for m in pilot.METHODS
                                        if decision(reference, m, case["beta"], ref_valid)
                                        is not None
                                    },
                                )
                                parity = pilot.parity(a, b)
                                comparisons["numerical_pairs"] += 1
                                comparisons["numerical_flagged_pairs"] += any(
                                    not c["passed"] for c in parity
                                )
                                if any(not c["passed"] for c in parity):
                                    flags.write(
                                        canonical(
                                            {
                                                "kind": "paired_numeric",
                                                "case": case["id"],
                                                "replicate": i,
                                                "selected": selected,
                                                "checks": parity,
                                            }
                                        )
                                        + b"\n"
                                    )
                        beta, covariance = pilot.block_gls(
                            case["x"], row["y"], case["sizes"], case["sigma"]
                        )
                        se = math.sqrt(covariance[1][1])
                        known_counts["attempted"] += 1
                        known_counts["reject"] += math.erfc(abs(beta[1] / se) / math.sqrt(2)) < 0.05
                        known_counts["cover"] += (
                            abs(beta[1] - case["beta"]) <= 1.959963984540054 * se
                        )
                        audit.write(
                            canonical(
                                {
                                    "case": case["id"],
                                    "replicate": i,
                                    "y": row["y"],
                                    "candidate": candidate,
                                    "candidate_numerical": check,
                                    "reference_bank": bank,
                                    "selected_reference": selected,
                                    "reference_numerical": ref_check,
                                    "known_beta": beta[1],
                                    "known_se": se,
                                }
                            )
                            + b"\n"
                        )
                        if not valid or any(d is None for d in decisions.values()):
                            flags.write(
                                canonical(
                                    {
                                        "kind": "candidate",
                                        "case": case["id"],
                                        "replicate": i,
                                        "candidate_numerical": check,
                                        "decisions": decisions,
                                    }
                                )
                                + b"\n"
                            )
                    exhausted(candidates)
                    for stream in references:
                        exhausted(stream)
                    for (engine, method), count in local.items():
                        summary = summarize_count(count)
                        counts.append(
                            {"case": case["id"], "engine": engine, "method": method, **summary}
                        )
                        if case["primary_future"] and engine == "candidate" and method != "normal":
                            n, k, m = summary["attempted"], summary["reject"], summary["failed"]
                            bounds = [
                                cp_interval(k, n, 0.05, 72)[0],
                                cp_interval(k + m, n, 0.05, 72)[1],
                            ]
                            primary.append(
                                {
                                    "case": case["id"],
                                    "method": method,
                                    "n": n,
                                    "k": k,
                                    "m": m,
                                    "bounds": bounds,
                                    "rule_pass": bounds[0] >= 0.04 and bounds[1] <= 0.06,
                                }
                            )
                finally:
                    journal.close()
            inventory.append(
                {"case": case["id"], "journal_sha256": pilot.shared.digest(path / "journal.sqlite")}
            )
    if len(primary) != 72:
        raise ValueError("primary family differs")
    run.write_new(
        destination / "summary.json",
        {
            "scope": "small execution/resource rehearsal; no confirmatory calibration claim",
            "manifest_sha256": pilot.shared.digest(folder / "manifest.json"),
            "mode": "rehearsal",
            "confirmation_claim": False,
            "execution_acceptance": executions,
            "resource_cap_acceptance": cap_receipts,
            "reference_variant_attempts": {k: dict(v) for k, v in variants.items()},
            "analysis_runtime": {
                "python": platform.python_version(),
                "scipy": __import__("scipy").__version__,
            },
            "counts": counts,
            "primary": primary,
            "all_primary_rule_pass": all(c["rule_pass"] for c in primary),
            "guards": dict(guards),
            "paired_comparisons": dict(comparisons),
            "known_variance_gls": dict(known_counts),
            "journal_inventory": inventory,
            "analysis_seconds": time.perf_counter() - before,
            "analysis_python_peak_working_set_bytes": run.peak_working_set(),
            "audit_sha256": pilot.shared.digest(destination / "audit.jsonl.gz"),
            "flags_sha256": pilot.shared.digest(destination / "flags.jsonl"),
            "bootstrap": "not run; secondary estimates descriptive without bootstrap intervals",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report(args.folder, args.destination)
