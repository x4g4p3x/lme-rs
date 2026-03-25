"""Pytest entry point: same numerical/API checks as `parity.run_all_checks()`."""

from __future__ import annotations

import parity


def test_sleepstudy_reml() -> None:
    parity.verify_sleepstudy_reml()


def test_sleepstudy_nested_lrt() -> None:
    parity.verify_sleepstudy_nested_lrt()


def test_cbpp_binomial_json() -> None:
    parity.verify_cbpp_binomial_json()


def test_grouseticks_poisson() -> None:
    parity.verify_grouseticks_poisson()


def test_pastes_nested_lmm() -> None:
    parity.verify_pastes_nested_lmm()


def test_penicillin_crossed() -> None:
    parity.verify_penicillin_crossed()


def test_type3_anova_pastes() -> None:
    parity.verify_type3_anova_pastes()


def test_weighted_sleepstudy() -> None:
    parity.verify_weighted_sleepstudy()


def test_confint_simulate() -> None:
    parity.verify_confint_and_simulate()
