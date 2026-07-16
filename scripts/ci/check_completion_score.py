#!/usr/bin/env python3
"""Validate the deterministic repository-completion score and published markers."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "completion_manifest.json"
README = ROOT / "README.md"
REPORT = ROOT / "REPO_COMPLETION_BY_AREA.md"


def rounded_percent(earned: int, possible: int) -> int:
    return int((Decimal(earned) * 100 / Decimal(possible)).quantize(Decimal("1"), ROUND_HALF_UP))


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    areas = payload.get("areas")
    if payload.get("schema_version") != 1 or not isinstance(areas, list):
        raise ValueError("completion manifest must use schema_version 1 and contain areas")

    seen_area_ids: set[int] = set()
    total_earned = total_possible = 0
    expected_area_scores: dict[int, int] = {}
    for area in areas:
        area_id = area.get("id")
        evidence = area.get("evidence")
        criteria = area.get("criteria")
        if (
            not isinstance(area_id, int)
            or area_id in seen_area_ids
            or not isinstance(criteria, list)
            or not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(path, str) or not (ROOT / path).exists() for path in evidence)
        ):
            raise ValueError("every area needs a unique id, existing evidence paths, and a criteria list")
        seen_area_ids.add(area_id)
        criterion_ids: set[str] = set()
        earned = possible = 0
        for criterion in criteria:
            criterion_id = criterion.get("id")
            weight = criterion.get("weight")
            complete = criterion.get("complete")
            if (
                not isinstance(criterion_id, str)
                or not criterion_id
                or criterion_id in criterion_ids
                or not isinstance(weight, int)
                or weight <= 0
                or not isinstance(complete, bool)
            ):
                raise ValueError(f"invalid criterion in area {area_id}")
            criterion_ids.add(criterion_id)
            possible += weight
            earned += weight if complete else 0
        if not possible:
            raise ValueError(f"area {area_id} has no scoreable criteria")
        expected_area_scores[area_id] = rounded_percent(earned, possible)
        total_earned += earned
        total_possible += possible

    overall_percent = rounded_percent(total_earned, total_possible)
    readme = README.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")
    readme_marker = (
        f"**Repository completion (evidence-weighted): {overall_percent}% "
        f"({total_earned}/{total_possible} scope units).**"
    )
    if readme_marker not in readme:
        raise ValueError(f"README completion marker is stale; expected: {readme_marker}")
    report_marker = (
        f"**Evidence-weighted overall: {overall_percent}% "
        f"({total_earned}/{total_possible} scope units).**"
    )
    if report_marker not in report:
        raise ValueError(f"completion report aggregate is stale; expected: {report_marker}")
    for area_id, score in expected_area_scores.items():
        pattern = rf"\| {area_id} \|.*?\| \*\*{score}%\*\* \|"
        if not re.search(pattern, report):
            raise ValueError(f"completion report score for area {area_id} is stale; expected {score}%")

    print(
        f"completion score: {overall_percent}% ({total_earned}/{total_possible} scope units; "
        f"{len(expected_area_scores)} areas)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
