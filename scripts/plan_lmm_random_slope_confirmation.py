"""Exact binomial decision-power/resource planning, never campaign execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

try:
    import scipy
    from scipy.stats import beta, binom
except ImportError:
    scipy = beta = binom = None


def require_scipy():
    if scipy is None:
        raise RuntimeError("SciPy is optional; install it to execute the planning calculations")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_rule(alpha, family, band):
    integer(family, "family", 1)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("familywise alpha must lie strictly between zero and one")
    if len(band) != 2 or not all(math.isfinite(x) for x in band) or not 0 <= band[0] < band[1] <= 1:
        raise ValueError("invalid probability band")
    if alpha / (2 * family) == 0:
        raise ValueError("tail probability underflow")


def cp_interval(k, n, alpha, family):
    require_scipy()
    integer(n, "attempted")
    integer(k, "rejections")
    validate_rule(alpha, family, [0, 1])
    if k > n:
        raise ValueError("rejections exceed attempts")
    tail = alpha / (2 * family)
    return [
        0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1)),
        1.0 if k == n else float(beta.isf(tail, k + 1, n - k)),
    ]


def acceptance_counts(n, alpha, family, band):
    """Inclusive counts whose two-sided Bonferroni CP interval is inside band."""
    require_scipy()
    integer(n, "attempted", 1)
    validate_rule(alpha, family, band)
    tail = alpha / (2 * family)
    low, high = band
    # Invert binomial tails independently of the beta-quantile interval routine.
    left, right = 0, n + 1
    if low != 0:
        while left < right:
            mid = (left + right) // 2
            if binom.sf(mid - 1, n, low) <= tail:
                right = mid
            else:
                left = mid + 1
        lower = left
    else:
        lower = 0
    left, right = -1, n
    if high != 1:
        while right - left > 1:
            mid = (left + right) // 2
            if binom.cdf(mid, n, high) <= tail:
                left = mid
            else:
                right = mid
        upper = left
    else:
        upper = n
    return lower, upper


def power(n, failed, p_valid, alpha, family, band):
    """Conditional Bin(n-failed,p_valid) planning, not a missingness assumption for decisions."""
    integer(n, "attempted", 1)
    integer(failed, "failed")
    if failed > n or not math.isfinite(p_valid) or not 0 <= p_valid <= 1:
        raise ValueError("invalid failed count or conditional probability")
    lower, upper = acceptance_counts(n, alpha, family, band)
    valid = n - failed
    admissible_lower, admissible_upper = max(0, lower), min(valid, upper - failed)
    if admissible_lower > admissible_upper:
        failure_probability = 1.0
    else:
        failure_probability = min(
            1.0,
            float(
                binom.cdf(admissible_lower - 1, valid, p_valid)
                + binom.sf(admissible_upper, valid, p_valid)
            ),
        )
    return {
        "n": n,
        "failed": failed,
        "valid": valid,
        "p_valid": p_valid,
        "acceptance_complete_counts": [lower, upper],
        "acceptance_observed_counts": [admissible_lower, admissible_upper],
        "single_claim_failure_probability": failure_probability,
        "single_claim_decision_power": 1 - failure_probability,
        "joint_decision_power_lower_bound": max(0.0, 1 - family * failure_probability),
    }


def validate_inputs(config):
    validate_rule(config["familywise_alpha"], config["family"], config["rejection_band"])
    from lmm_random_slope_v2 import registry

    expected_cases = {c["id"] for c in registry() if c["primary_future"]}
    if set(config["primary_cases"]) != expected_cases or set(config["primary_methods"]) != {
        "satterthwaite",
        "kenward_roger",
    }:
        raise ValueError("primary claims must match the declared candidate-only family")
    if (
        len(set(config["primary_cases"])) != len(config["primary_cases"])
        or len(set(config["primary_methods"])) != len(config["primary_methods"])
        or len(config["primary_cases"]) * len(config["primary_methods"]) != config["family"]
    ):
        raise ValueError("primary family inventory differs")
    grid = config["replicate_grid"]
    if not grid or sorted(set(grid)) != grid:
        raise ValueError("replicate grid must be nonempty, unique and sorted")
    for n in grid:
        integer(n, "replicates", 1)
    integer(config["exploratory_replicates_proposal"], "exploratory replicates", 1)
    integer(config["reference_subset_per_case_proposal"], "reference replicates", 1)
    for name in ["probability_sensitivity", "failure_fraction_sensitivity"]:
        if not config[name] or any(not math.isfinite(p) or not 0 <= p <= 1 for p in config[name]):
            raise ValueError("invalid sensitivity probabilities")
    if (
        not 0 < config["global_decision_power_target"] < 1
        or not 0 <= config["ideal_true_probability"] <= 1
    ):
        raise ValueError("invalid planning target")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_count(n, fraction):
    integer(n, "attempted", 1)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid failure fraction")
    return int((Decimal(n) * Decimal(str(fraction))).to_integral_value(rounding=ROUND_CEILING))


def resources(config, selected, root):
    if selected is None:
        return None
    folder = root / config["pilot_directory"]
    ledger = root / "scientific/RANDOM_SLOPE_PILOT_V2.json"
    if digest(ledger) != config["pilot_summary_sha256"]:
        raise ValueError("pilot resource anchor changed")
    pilot = json.loads(ledger.read_text(encoding="utf-8"))
    import lmm_random_slope_v2 as v2

    manifest = v2.verify_run(folder)
    if digest(folder / "summary.json") != pilot["summary_sha256"]:
        raise ValueError("raw summary changed")
    summary = v2.load(folder / "summary.json")
    for name, expected in summary["artifacts"].items():
        if digest(folder / name) != expected:
            raise ValueError("pilot artifact changed")
    if manifest["attempted"] != pilot["attempted"]:
        raise ValueError("pilot budget differs")
    rates = {}
    for case in v2.load(folder / "designs.json"):
        path = folder / "cases" / case["id"]
        pairs = v2.ordered(path / "paired.jsonl", case["id"], manifest["replicates"], False)
        rates[case["id"]] = {
            e: sum(row[e]["seconds"] for row in pairs) / len(pairs)
            for e in ("candidate", "reference")
        }
        rates[case["id"]].update(
            candidate_bytes=sum(
                (path / n).stat().st_size for n in ("inputs.jsonl.gz", "candidate.jsonl.gz")
            )
            / len(pairs),
            reference_bytes=(path / "paired.jsonl").stat().st_size / len(pairs),
        )
    # Attribute measured phase wall time proportionally to case-specific fit times.
    scale = {
        e: pilot["execution"][e]["seconds"]
        / (manifest["replicates"] * sum(r[e] for r in rates.values()))
        for e in ("candidate", "reference")
    }
    plans = ("primary_only", "primary_plus_secondary", "all_cases_full_n", "all_cases_all_paired")
    result = {}
    for name in plans:
        candidate = reference = hours = raw_bytes = reference_hours = reference_bytes = 0
        for case, rate in rates.items():
            n = (
                selected
                if case in config["primary_cases"] or name.startswith("all_cases")
                else (0 if name == "primary_only" else config["exploratory_replicates_proposal"])
            )
            ref = (
                n
                if name == "all_cases_all_paired"
                else min(n, config["reference_subset_per_case_proposal"])
            )
            candidate += n
            reference += ref
            hours += (
                n * rate["candidate"] * scale["candidate"]
                + ref * rate["reference"] * scale["reference"]
            ) / 3600
            raw_bytes += n * rate["candidate_bytes"] + ref * rate["reference_bytes"]
            reference_hours += ref * rate["reference"] * scale["reference"] / 3600
            reference_bytes += ref * rate["reference_bytes"]
        result[name] = {
            "candidate_datasets": candidate,
            "reference_datasets": reference,
            "pilot_rate_hours": hours,
            "twofold_time_provision_hours": 2 * hours,
            "estimated_input_candidate_pair_bytes": raw_bytes,
            "twice_estimated_raw_bytes": 2 * raw_bytes,
            "four_variant_reference_proxy_hours": hours + 3 * reference_hours,
            "four_variant_twofold_time_provision_hours": 2 * (hours + 3 * reference_hours),
            "four_variant_twice_estimated_raw_bytes": 2 * (raw_bytes + 3 * reference_bytes),
        }
    return {
        "scope": (
            "case-weighted four-worker pilot wall/disk extrapolation; no prediction interval; "
            "excludes numerical audit/analysis/bootstrap and fixed artifacts; "
            "four-variant fields assume four times baseline R cost, not a measured future cost; "
            "paired bytes include a duplicate candidate record; compression may change"
        ),
        "n_primary": selected,
        "case_rates": rates,
        "phase_wall_attribution_scale": scale,
        "scenarios": result,
    }


def run(inputs, output):
    require_scipy()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(inputs.read_text(encoding="utf-8"))
    validate_inputs(config)
    rows = []
    for n in config["replicate_grid"]:
        for fraction in config["failure_fraction_sensitivity"]:
            failed = failure_count(n, fraction)
            for p in config["probability_sensitivity"]:
                rows.append(
                    dict(
                        failure_fraction=fraction,
                        **power(
                            n,
                            failed,
                            p,
                            config["familywise_alpha"],
                            config["family"],
                            config["rejection_band"],
                        ),
                    )
                )
    ideal = [
        power(
            n,
            0,
            config["ideal_true_probability"],
            config["familywise_alpha"],
            config["family"],
            config["rejection_band"],
        )
        for n in config["replicate_grid"]
    ]
    selected = next(
        (
            r["n"]
            for r in ideal
            if r["joint_decision_power_lower_bound"] >= config["global_decision_power_target"]
        ),
        None,
    )
    result = {
        "schema_version": 1,
        "scope": (
            "exact binomial planning, not observed calibration or authorization to run a campaign"
        ),
        "inputs_sha256": digest(inputs),
        "source_sha256": digest(Path(__file__)),
        "runtime": {"python": platform.python_version(), "scipy": scipy.__version__},
        "inputs": config,
        "ideal": ideal,
        "smallest_passing_tested_n": selected,
        "selection_scope": (
            "smallest passing tested grid point; no global-minimum or actual-rate guarantee"
        ),
        "sensitivity": rows,
        "failure_count_rounding": "ceiling(n * decimal representation of fraction)",
        "resources": resources(config, selected, Path(__file__).resolve().parents[1]),
        "resources_proposed_30000": resources(config, 30000, Path(__file__).resolve().parents[1]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)
