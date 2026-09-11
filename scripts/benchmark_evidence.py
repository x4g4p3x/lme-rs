"""Validation and descriptive uncertainty for fit-only benchmark reports.

Intervals describe within-process timing variation, not machine-to-machine or
between-session reproducibility. No speed claim is made without matched LMM fits.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

MIN_REPEATS = 10
OBJECTIVE_ATOL = 1e-5
OBJECTIVE_RTOL = 1e-6
COEFFICIENT_ATOL = 1e-4
COEFFICIENT_RTOL = 1e-4


def samples(result: dict[str, Any], metric: str) -> list[float]:
    values = result[metric]["samples_seconds"]
    if not values or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
        for x in values
    ):
        raise ValueError(f"{metric}: expected finite, positive timing samples")
    if len(values) != result["repeats"]:
        raise ValueError(f"{metric}: sample count differs from requested repeats")
    return values


def check_result(result: dict[str, Any], case: Any, implementation: str, repeats: int) -> None:
    for key, expected in {
        "case": case.name,
        "formula": case.formula,
        "model": case.model,
        "reml": case.reml,
        "implementation": implementation,
        "repeats": repeats,
    }.items():
        if result.get(key) != expected:
            raise ValueError(f"result {key} does not match request: {expected!r}")
    if not isinstance(result.get("n_obs"), int) or result["n_obs"] <= 0:
        raise ValueError("result has no positive observation count")
    for metric in ("cold_fit", "fit_prepared", "prepare_lmer"):
        if metric in result:
            # Recompute summaries from the raw measurements; never trust a stale
            # median supplied by an executable or hand-edited artifact.
            values = samples(result, metric)
            result[metric]["summary"]["median_seconds"] = statistics.median(values)
    checks = result.get("fit_checks", [])
    if len(checks) != repeats:
        raise ValueError("missing fit checks for measured repetitions; rebuild the benchmark")


def finite_fit(check: dict[str, Any]) -> bool:
    values = [check.get("objective"), *(check.get("coefficients") or [])]
    return bool(check.get("coefficients")) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values
    )


def equivalent_fit(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a, b = left["coefficients"], right["coefficients"]
    return (
        len(a) == len(b)
        and math.isclose(
            left["objective"],
            right["objective"],
            abs_tol=OBJECTIVE_ATOL,
            rel_tol=OBJECTIVE_RTOL,
        )
        and all(
            math.isclose(x, y, abs_tol=COEFFICIENT_ATOL, rel_tol=COEFFICIENT_RTOL)
            for x, y in zip(a, b)
        )
    )


def fit_agreement(case: Any, rust: dict[str, Any], julia: dict[str, Any]) -> dict[str, Any]:
    checks = [*rust["fit_checks"], *julia["fit_checks"]]
    reasons = []
    if rust["n_obs"] != julia["n_obs"]:
        reasons.append("observation counts differ")
    if not all(finite_fit(check) for check in checks):
        reasons.append("non-finite or missing fitted values")
    if not all(check.get("converged") is True for check in checks):
        reasons.append("not every measured fit established convergence")
    # GLMM likelihood approximations are not normalized across the two engines.
    # Keep the data, but never turn those timings into a validated speed claim.
    if case.model not in ("lmm", "lmm_weighted"):
        reasons.append("GLMM objective equivalence has not been established")
    elif all(finite_fit(check) for check in checks):
        reference = julia["fit_checks"][0]
        if not all(equivalent_fit(check, reference) for check in checks):
            reasons.append("objective or fixed-effect estimates exceed declared tolerances")
    return {
        "status": "passed" if not reasons else "unverified",
        "reasons": reasons,
        "objective_atol": OBJECTIVE_ATOL,
        "objective_rtol": OBJECTIVE_RTOL,
        "coefficient_atol": COEFFICIENT_ATOL,
        "coefficient_rtol": COEFFICIENT_RTOL,
    }


def ratio_interval(rust: list[float], julia: list[float]) -> list[float]:
    """Deterministic independent percentile bootstrap of the ratio of medians."""
    rng = random.Random(20260911)
    ratios = sorted(
        statistics.median(rng.choices(rust, k=len(rust)))
        / statistics.median(rng.choices(julia, k=len(julia)))
        for _ in range(2000)
    )
    return [ratios[49], ratios[1949]]


def assess_timing(
    rust: dict[str, Any],
    julia: dict[str, Any],
    agreement: dict[str, Any],
    target: float,
) -> dict[str, Any]:
    a, b = samples(rust, "cold_fit"), samples(julia, "cold_fit")
    enough = min(len(a), len(b)) >= MIN_REPEATS
    warmed = min(rust["warmups"], julia["warmups"]) >= 1
    interval = ratio_interval(a, b) if enough and warmed else None
    eligible = agreement["status"] == "passed" and interval is not None
    winner = (
        "unverified"
        if not eligible
        else ("rust" if interval[1] < 1 else "julia" if interval[0] > 1 else "inconclusive")
    )
    target_met = None
    if eligible:
        if interval[1] <= target:
            target_met = True
        elif interval[0] > target:
            target_met = False
    return {
        "eligible_for_speed_claim": eligible,
        "faster_implementation": winner,
        "meets_target": target_met,
        "ratio_interval_95": interval,
        "uncertainty_scope": "within-process independent percentile bootstrap; 2000 resamples",
        "sampling_status": "measured" if enough and warmed else "smoke_only",
        "fit_agreement": agreement,
    }
