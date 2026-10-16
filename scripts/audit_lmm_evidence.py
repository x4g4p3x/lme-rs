"""Read-only, independent record/count audit of a frozen confirmatory campaign.

Does not import the campaign runner, refit models, recompute bootstrap MCSE, or
modify the evidence. The expected summary digest is the external trust anchor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from itertools import zip_longest
from pathlib import Path

METHODS = ("normal", "satterthwaite", "kenward_roger")
SUMMARY_SHA = "b6f9a8dae787591da4c157ffcc7a9714014e6c8c67f5e3b3edb78bd4c28c399c"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_check(path, expected):
    require(path.is_file(), f"missing artifact: {path.name}")
    actual = sha(path)
    require(actual == expected, f"artifact checksum mismatch: {path.name}")
    return actual


def safe_relative(name):
    path = Path(name)
    require(
        not path.anchor and ".." not in path.parts and ":" not in name and "\\" not in name,
        "unsafe artifact name",
    )
    return path


def records(path, case_id, count):
    opener = gzip.open if path.suffix == ".gz" else open
    seen = 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            require(index < count, f"extra record in {path.name}")
            require(
                row.get("case") == case_id
                and type(row.get("replicate")) is int
                and row["replicate"] == index,
                f"missing, duplicate or misaligned record in {case_id}/{path.name} at {index}",
            )
            seen += 1
            yield row
    require(seen == count, f"truncated records in {case_id}/{path.name}: {seen}/{count}")


def evaluate(fit, method, truth, distribution=None, index=None):
    if fit.get("converged") is not True:
        return None
    row = fit["methods"].get(method, {})
    if not row or "error" in row:
        return None
    if distribution is not None:
        d = distribution[method]
        if d["error"][index] is not None:
            return None
        critical, independent_p = d["critical"][index], d["distribution_p"][index]
    else:
        critical, independent_p = row.get("critical"), row.get("distribution_p")
    beta, se, p = (row.get(k) for k in ["beta", "se", "p"])
    values = [beta, se, p, critical, independent_p]
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
        return None
    if se <= 0 or critical <= 0 or not 0 <= p <= 1 or abs(p - independent_p) > 1e-8:
        return None
    if method != "normal" and not (
        isinstance(row.get("df"), (float, int)) and math.isfinite(row["df"]) and row["df"] > 0
    ):
        return None
    reject = p < 0.05
    require(reject == (abs(beta) > critical * se), "test/interval mismatch")
    return reject, beta - critical * se <= truth <= beta + critical * se


def control_weights(case):
    x, sizes = case["x"], case["sizes"]
    sigma2, tau2 = 1 - case["rho"], case["rho"]
    if case["location"] == "within":
        start = 0
        for n in sizes:
            require(abs(sum(x[start : start + n])) < 1e-10, "within predictor is not centered")
            start += n
        ssx = math.fsum(v * v for v in x)
        return [v / ssx for v in x], math.sqrt(sigma2 / ssx)
    group_x, start = [], 0
    for n in sizes:
        require(
            all(v == x[start] for v in x[start : start + n]),
            "between predictor varies within group",
        )
        group_x.append(x[start])
        start += n
    weights = [1 / (tau2 + sigma2 / n) for n in sizes]
    center = math.fsum(w * v for w, v in zip(weights, group_x)) / sum(weights)
    information = math.fsum(w * (v - center) ** 2 for w, v in zip(weights, group_x))
    coefficients = [
        w * (v - center) / (information * n)
        for w, v, n in zip(weights, group_x, sizes)
        for _ in range(n)
    ]
    return coefficients, math.sqrt(1 / information)


def classification(interval):
    lower, upper = interval
    if lower >= 0.04 and upper <= 0.06:
        return "supported"
    if upper < 0.04 or lower > 0.06:
        return "contradicted"
    return "inconclusive"


def verify_counts(row, counts, bounds, reference=False):
    n, valid, reject, cover = counts
    require(
        row["attempted"] == n and row["failed"] == n - valid,
        "attempt/failure counts differ from raw records",
    )
    prefix = "reference_" if reference else ""
    for metric, k in [("rejection", reject), ("coverage", cover)]:
        original = row[metric]
        require(
            original["evaluable"] == valid and original["count"] == k,
            f"raw {metric} count differs from summary",
        )
        require(original["rate"] == (k / valid if valid else None), f"{metric} rate differs")
        for suffix, table in [("95", "pointwise"), ("_simultaneous95", "simultaneous")]:
            expected = [bounds[prefix + table][k][0], bounds[prefix + table][k + n - valid][1]]
            require(
                original[f"failure_envelope{suffix}"] == expected,
                f"{metric} failure envelope differs",
            )


def audit_case(directory, case, manifest, summary_rows, bounds):
    case_id = case["id"]
    n, nr = manifest["replicates"], manifest["reference_replicates"]
    dist = load(directory / "candidate-distributions.json")
    for method in METHODS:
        require(
            all(len(dist[method][key]) == n for key in ["critical", "distribution_p", "error"]),
            "distribution array length differs",
        )
    counters = {
        f"{e}:{m}": [n if e == "candidate" else nr, 0, 0, 0]
        for e in ["candidate", "reference"]
        for m in METHODS
    }
    counters["control:known_variance"] = [n, 0, 0, 0]
    fits = {e: Counter() for e in ["candidate", "reference"]}
    paired = {m: Counter() for m in METHODS}
    reference = iter(records(directory / "reference.jsonl", case_id, nr))
    coefficients, se = control_weights(case)
    z = statistics.NormalDist().inv_cdf(0.975)
    inputs = records(directory / "inputs.jsonl.gz", case_id, n)
    candidates = records(directory / "candidate.jsonl.gz", case_id, n)
    for rep, (data, fit) in enumerate(zip_longest(inputs, candidates)):
        require(data is not None and fit is not None, "candidate/input length differs")
        y = data["y"]
        require(
            len(y) == len(coefficients)
            and all(isinstance(v, (float, int)) and math.isfinite(v) for v in y),
            "invalid response",
        )
        candidate_outcomes = {}
        engines = [("candidate", fit)]
        if rep < nr:
            engines.append(("reference", next(reference)["reference"]))
        for engine, fitted in engines:
            fits[engine]["converged"] += fitted.get("converged") is True
            fits[engine]["singular"] += fitted.get("singular") is True
            fits[engine]["singular_unknown"] += not isinstance(fitted.get("singular"), bool)
            for method in METHODS:
                outcome = evaluate(
                    fitted, method, case["beta"], dist if engine == "candidate" else None, rep
                )
                if outcome is not None:
                    counter = counters[f"{engine}:{method}"]
                    counter[1] += 1
                    counter[2] += outcome[0]
                    counter[3] += outcome[1]
                if engine == "candidate":
                    candidate_outcomes[method] = outcome
                elif outcome is not None and candidate_outcomes[method] is not None:
                    paired[method]["common"] += 1
                    paired[method]["reject_disagreements"] += (
                        outcome[0] != candidate_outcomes[method][0]
                    )
                    paired[method]["cover_disagreements"] += (
                        outcome[1] != candidate_outcomes[method][1]
                    )
        beta = math.fsum(a * b for a, b in zip(coefficients, y))
        control = counters["control:known_variance"]
        control[1] += 1
        control[2] += abs(beta) > z * se
        control[3] += abs(beta - case["beta"]) <= z * se
    require(next(reference, None) is None, "extra paired audit record")
    for label, counts in counters.items():
        verify_counts(summary_rows[label], counts, bounds, reference=label.startswith("reference:"))
    return {
        "case": case_id,
        "primary": case["primary"],
        "counts": counters,
        "fits": fits,
        "paired": paired,
    }


def audit_run(root, expected_summary):
    hash_check(root / "summary.json", expected_summary)
    summary = load(root / "summary.json")
    for name, expected in summary["artifacts"].items():
        if name != "case_summaries_sha256":
            hash_check(root / safe_relative(name), expected)
    manifest = load(root / "manifest.json")
    require(
        manifest["protocol"] == "lmm-calibration-v3" and manifest["namespace"] == "confirmatory",
        "wrong study",
    )
    require(
        manifest["replicates"] == 20000
        and manifest["reference_replicates"] == 1000
        and manifest["primary_family"] == 48,
        "wrong budget/family",
    )
    for name, expected in manifest["sources"].items():
        hash_check(root / "frozen-sources" / safe_relative(name), expected)
    designs = load(root / "designs.json")
    hash_check(root / "designs.json", manifest["designs_sha256"])
    ids = [c["id"] for c in designs]
    require(
        len(ids) == len(set(ids)) == 80 and sum(c["primary"] for c in designs) == 24,
        "wrong case registry",
    )
    require(
        manifest["attempted_datasets"] == 80 * 20000
        and manifest["reference_attempted_datasets"] == 80 * 1000,
        "wrong total attempts",
    )
    candidate_env, ref_env = (
        load(root / "candidate-environment.json"),
        load(root / "reference-environment.json"),
    )
    require(
        candidate_env["engine"] == manifest["candidate_version"] == "0.2.7",
        "wrong candidate version",
    )
    hash_maps = [
        (manifest["case_inputs_sha256"], "inputs.jsonl.gz"),
        (candidate_env["case_candidate_sha256"], "candidate.jsonl.gz"),
        (ref_env["case_reference_sha256"], "reference.jsonl"),
        (ref_env["case_distributions_sha256"], "candidate-distributions.json"),
    ]
    for mapping, _ in hash_maps:
        require(set(mapping) == set(ids), "incomplete case hash map")
    require(
        canonical_sha(hash_maps[0][0]) == manifest["inputs_sha256"],
        "aggregate input digest differs",
    )
    summary_rows = {}
    for row in summary["summaries"]:
        key = row["case"], row["method"]
        require(key not in summary_rows, "duplicate case/method summary")
        summary_rows[key] = row
    require(len(summary_rows) == 80 * 7, "wrong number of case/method summaries")
    bounds = load(root / "binomial-bounds.json")
    results, case_digests, ledger = [], {}, []
    for i, case in enumerate(designs):
        case_id = case["id"]
        require(safe_relative(case_id).name == case_id, "unsafe case ID")
        directory = root / "cases" / case_id
        for mapping, name in hash_maps:
            hash_check(directory / name, mapping[case_id])
        part = load(directory / "case-summary.json")
        require(
            part["summaries"] == [r for r in summary["summaries"] if r["case"] == case_id],
            "case and aggregate summaries differ",
        )
        for key in ["fit_summaries", "paired_differences"]:
            require(
                part[key] == [r for r in summary[key] if r["case"] == case_id],
                f"case and aggregate {key} differ",
            )
        case_digests[case_id] = sha(directory / "case-summary.json")
        rows = {
            m: summary_rows[case_id, m]
            for m in [f"{e}:{m}" for e in ["candidate", "reference"] for m in METHODS]
            + ["control:known_variance"]
        }
        result = audit_case(directory, case, manifest, rows, bounds)
        for original in part["fit_summaries"]:
            actual = result["fits"][original["engine"]]
            require(
                actual["converged"] == original["diagnostics"]["converged"]
                and actual["singular"] == original["singular_rate"]["count"]
                and actual["singular_unknown"] == original["singular_unknown"],
                "fit diagnostics differ",
            )
        for original in part["paired_differences"]:
            actual = result["paired"][original["method"]]
            require(actual["common"] == original["common_evaluable"], "paired denominator differs")
            for kind in ["reject", "cover"]:
                require(
                    (original[kind]["max_absolute_difference"] or 0)
                    == (1 if actual[kind + "_disagreements"] else 0),
                    "paired decisions differ",
                )
        for label, row in rows.items():
            interval = row["rejection"]["failure_envelope_simultaneous95"]
            primary_claim = case["primary"] and label in [
                "candidate:satterthwaite",
                "candidate:kenward_roger",
            ]
            ledger.append(
                {
                    "case": case_id,
                    "method": label,
                    "primary_claim": primary_claim,
                    "stratum": case["stratum"],
                    "groups": case["groups"],
                    "sizes": case["sizes"],
                    "generating_rho": case["rho"],
                    "predictor_location": case["location"],
                    "true_slope": case["beta"],
                    "attempted": row["attempted"],
                    "failed": row["failed"],
                    "rejection": row["rejection"],
                    "coverage": row["coverage"],
                    "status": classification(interval) if primary_claim else "descriptive_only",
                }
            )
        results.append(result)
        print(f"evidence audit: {i + 1}/80 cases", flush=True)
    require(
        canonical_sha(case_digests) == summary["artifacts"]["case_summaries_sha256"],
        "aggregate case-summary digest differs",
    )
    decisions = summary["confirmatory_decisions"]
    for method, original in decisions["methods"].items():
        counts = Counter(
            r["status"] for r in ledger if r["primary_claim"] and r["method"] == method
        )
        require(
            sum(counts.values()) == 24
            and all(
                counts[k] == original[k] for k in ["supported", "inconclusive", "contradicted"]
            ),
            "primary decisions differ",
        )
        require(
            original["validated_for_primary_scope"] == (counts["supported"] == 24),
            "global gate differs",
        )
    require(len(decisions["claims"]) == 48, "wrong primary claim count")
    expected_claims = {(r["case"], r["method"]): r for r in ledger if r["primary_claim"]}
    seen_claims = set()
    for claim in decisions["claims"]:
        key = claim["case"], claim["method"]
        require(
            key in expected_claims and key not in seen_claims, "unknown or duplicate primary claim"
        )
        seen_claims.add(key)
        checked = expected_claims[key]
        require(
            claim["decision"] == checked["status"] and claim["complement_verified"] is True,
            "claim decision or complement differs",
        )
        for metric in ["rejection", "coverage"]:
            require(
                claim[f"{metric}_rate"] == checked[metric]["rate"]
                and claim[f"{metric}_envelope_simultaneous95"]
                == checked[metric]["failure_envelope_simultaneous95"],
                "primary claim rate/envelope differs",
            )
    manuscript = (
        Path(__file__).resolve().parents[1] / "scientific/CALIBRATION_CONFIRMATORY_0.2.7.md"
    )
    observed = {
        name: sha(root / name) for name in summary["artifacts"] if name != "case_summaries_sha256"
    }
    observed.update(
        {"summary.json": expected_summary, "case_summaries_sha256": canonical_sha(case_digests)}
    )
    mismatches = []
    for name, recorded in re.findall(
        r"`([^`]+)`:\s*`([0-9a-f]{64})`", manuscript.read_text(encoding="utf-8")
    ):
        if name in observed and observed[name] != recorded:
            mismatches.append(
                {"artifact": name, "report_sha256": recorded, "verified_sha256": observed[name]}
            )
    return {
        "schema_version": 1,
        "scope": "record integrity, binary outcomes and decisions; no fits or bootstrap repeated",
        "source_summary_sha256": expected_summary,
        "auditor_sha256": sha(Path(__file__)),
        "candidate_version": "0.2.7",
        "candidate_native_sha256": candidate_env["native_sha256"],
        "cases": results,
        "decisions": decisions["methods"],
        "manuscript_checksum_errors": mismatches,
        "verified_artifacts": observed,
        "evidence_table": ledger,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-summary-sha256", default=SUMMARY_SHA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "audit report already exists")
    report = audit_run(args.directory, args.expected_summary_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
