"""Tests for prepare_lmer, fit_prepared, refit_lmer, and cv_grouped."""

from pathlib import Path

import lme_python
import polars as pl
import pytest

SLEEPSTUDY = Path(__file__).resolve().parents[2] / "tests" / "data" / "sleepstudy.csv"


@pytest.fixture
def sleepstudy() -> pl.DataFrame:
    return pl.read_csv(SLEEPSTUDY)


def test_prepare_lmer_fit_prepared(sleepstudy: pl.DataFrame) -> None:
    prep = lme_python.prepare_lmer("Reaction ~ Days + (1 | Subject)", data=sleepstudy)
    assert prep.blocked_kernel in (True, False)
    assert isinstance(prep.blocked_kernel_detail, str)

    fit_reml = lme_python.fit_prepared(prep, reml=True)
    cold = lme_python.lmer("Reaction ~ Days + (1 | Subject)", data=sleepstudy, reml=True)
    assert abs(fit_reml.coefficients[0] - cold.coefficients[0]) < 1e-8


def test_prepare_glmer_fit_prepared() -> None:
    cbpp = pl.read_csv(Path(__file__).resolve().parents[2] / "tests" / "data" / "cbpp_binary.csv")
    prep = lme_python.prepare_glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        data=cbpp,
        family_name="binomial",
        n_agq=1,
    )
    assert prep.n_agq == 1
    assert prep.family_name == "binomial"
    fit = lme_python.fit_prepared_glmer(prep)
    cold = lme_python.glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        data=cbpp,
        family_name="binomial",
        n_agq=1,
    )
    assert abs(fit.coefficients[0] - cold.coefficients[0]) < 1e-6


def test_refit_lmer(sleepstudy: pl.DataFrame) -> None:
    fit = lme_python.refit_lmer("Reaction ~ Days + (1 | Subject)", data=sleepstudy, reml=True)
    assert fit.reml_criterion is not None
    assert len(fit.coefficients) == 2


def test_cv_grouped(sleepstudy: pl.DataFrame) -> None:
    cv = lme_python.cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        data=sleepstudy,
        group="Subject",
        n_splits=5,
        reml=True,
        seed=42,
    )
    assert cv.n_splits == 5
    assert cv.group_col == "Subject"
    assert len(cv.oof_predictions) == sleepstudy.height
    assert len(cv.test_fold) == sleepstudy.height
    assert cv.all_converged
    assert cv.rmse > 0.0
    assert len(cv.folds) == 5

    cv2 = lme_python.cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        data=sleepstudy,
        group="Subject",
        n_splits=5,
        reml=True,
        seed=42,
        n_jobs=1,
    )
    assert cv.oof_predictions == cv2.oof_predictions


def test_cv_grouped_parallel_matches_sequential(sleepstudy: pl.DataFrame) -> None:
    sequential = lme_python.cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        data=sleepstudy,
        group="Subject",
        n_splits=5,
        reml=True,
        seed=99,
        n_jobs=1,
    )
    parallel = lme_python.cv_grouped(
        "Reaction ~ Days + (1 | Subject)",
        data=sleepstudy,
        group="Subject",
        n_splits=5,
        reml=True,
        seed=99,
        n_jobs=4,
    )
    assert sequential.oof_predictions == parallel.oof_predictions
    assert sequential.rmse == parallel.rmse
