"""Golden-manifest coefficient parity for Python bindings."""

from __future__ import annotations

import json
from pathlib import Path

import lme_python
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "tests" / "data" / "golden_parity_manifest.json"


def _load_manifest() -> dict:
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def _case(case_id: str) -> dict:
    for entry in _load_manifest()["cases"]:
        if entry["id"] == case_id:
            return entry
    raise KeyError(case_id)


def _assert_coefs(case_id: str, fit, checks: list[dict]) -> None:
    names = list(fit.fixed_names)
    coef = list(fit.coefficients)
    by_name = dict(zip(names, coef))
    for check in checks:
        actual = by_name[check["name"]]
        expected = check["value"]
        tol = check["tolerance"]
        assert abs(actual - expected) <= tol, (
            f"{case_id} {check['name']}: actual={actual} expected={expected} tol={tol}"
        )


@pytest.mark.parametrize(
    "case_id,fitter",
    [
        (
            "cbpp_binomial_probit",
            lambda: lme_python.glmer(
                "y ~ period2 + period3 + period4 + (1 | herd)",
                data=pl.read_csv(REPO_ROOT / "tests/data/cbpp_binary.csv"),
                family_name="binomial",
                link_name="probit",
            ),
        ),
        (
            "cbpp_binomial_weighted",
            lambda: lme_python.glmer_weighted(
                "y ~ period2 + period3 + period4 + (1 | herd)",
                data=pl.read_csv(REPO_ROOT / "tests/data/cbpp_binary_weighted.csv"),
                family_name="binomial",
                weights=pl.read_csv(REPO_ROOT / "tests/data/cbpp_binary_weighted.csv")[
                    "prior_w"
                ].to_list(),
            ),
        ),
        (
            "grouseticks_poisson_offset",
            lambda: lme_python.glmer(
                "TICKS ~ YEAR96 + YEAR97 + offset(log_height) + (1 | BROOD)",
                data=pl.read_csv(REPO_ROOT / "tests/data/grouseticks.csv"),
                family_name="poisson",
            ),
        ),
        (
            "sleepstudy_offset_reml",
            lambda: lme_python.lmer(
                "Reaction ~ Days + offset(OffsetDays10) + (Days | Subject)",
                data=pl.read_csv(REPO_ROOT / "tests/data/sleepstudy.csv"),
                reml=True,
            ),
        ),
    ],
)
def test_golden_coefficient_parity(case_id: str, fitter) -> None:
    entry = _case(case_id)
    fit = fitter()
    _assert_coefs(case_id, fit, entry["expected"]["coefficients"])
