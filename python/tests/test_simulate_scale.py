"""Parallel and batched parametric simulation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from lme_python import lmer

SLEEPSTUDY = Path(__file__).resolve().parents[2] / "tests" / "data" / "sleepstudy.csv"


def _sleepstudy() -> pl.DataFrame:
    return pl.read_csv(SLEEPSTUDY)


def test_simulate_parallel_reproducible_with_seed() -> None:
    fit = lmer("Reaction ~ Days + (Days | Subject)", _sleepstudy(), True)
    seq = fit.simulate(40, n_jobs=1, seed=123)
    par = fit.simulate(40, n_jobs=4, seed=123)
    assert len(seq.simulations) == len(par.simulations) == 40
    for a, b in zip(seq.simulations, par.simulations, strict=True):
        assert a == b


def test_simulate_batches_cover_all_draws() -> None:
    fit = lmer("Reaction ~ Days + (Days | Subject)", _sleepstudy(), True)
    nsim = 25
    batch_size = 7
    batches = list(fit.simulate_batches(nsim, batch_size, n_jobs=2, seed=5))
    total = sum(len(b.simulations) for b in batches)
    assert total == nsim
    assert len(batches) == 4  # ceil(25 / 7)


def test_simulate_n_jobs_zero_raises() -> None:
    fit = lmer("Reaction ~ Days + (Days | Subject)", _sleepstudy(), True)
    try:
        fit.simulate(3, n_jobs=0)
    except ValueError as exc:
        assert "n_jobs" in str(exc)
    else:
        raise AssertionError("expected ValueError for n_jobs=0")
