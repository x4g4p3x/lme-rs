"""Small independent analytic contracts also run on every native/wheel CI pass."""

import importlib.util
import json
from pathlib import Path

import lme_python as lme
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scientific_validation", ROOT / "scripts/scientific_validation.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)
CONTRACTS = {
    case["id"]: case for case in json.loads((ROOT / "scientific/cases.json").read_text())["cases"]
}


@pytest.mark.parametrize(
    "case_id",
    [
        "ols-rational-v1",
        "balanced-lmm-ml-v1",
        "balanced-lmm-reml-v1",
    ],
)
def test_independent_analytic_contract(case_id):
    checks = validation.CaseChecks(CONTRACTS[case_id])
    validation.numerical_case(case_id, checks, lme, pl, ROOT)
    assert checks.values
    assert all(record["passed"] for record in checks.values), checks.values
