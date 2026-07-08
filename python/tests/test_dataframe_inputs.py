"""pandas and PyArrow tabular inputs are accepted at the Python FFI boundary."""

import lme_python
import polars as pl
import pytest

FORMULA = "Reaction ~ Days + (Days | Subject)"
DATA_PATH = "../tests/data/sleepstudy.csv"


def _reference_fit(df_pl: pl.DataFrame) -> lme_python.PyLmeFit:
    return lme_python.lmer(FORMULA, data=df_pl, reml=True)


def test_lmer_accepts_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    df_pl = pl.read_csv(DATA_PATH)
    pdf = df_pl.to_pandas()
    assert isinstance(pdf, pd.DataFrame)

    ref = _reference_fit(df_pl)
    fit = lme_python.lmer(FORMULA, data=pdf, reml=True)

    assert fit.num_obs == ref.num_obs
    assert fit.coefficients == pytest.approx(ref.coefficients, rel=1e-9, abs=1e-6)
    assert len(fit.predict(pdf)) == ref.num_obs


def test_lmer_accepts_pyarrow_table():
    pa = pytest.importorskip("pyarrow")
    df_pl = pl.read_csv(DATA_PATH)
    table = df_pl.to_arrow()
    assert isinstance(table, pa.Table)

    ref = _reference_fit(df_pl)
    fit = lme_python.lmer(FORMULA, data=table, reml=True)

    assert fit.num_obs == ref.num_obs
    assert fit.coefficients == pytest.approx(ref.coefficients, rel=1e-9, abs=1e-6)
    assert len(fit.predict(table)) == ref.num_obs


def test_invalid_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="polars.DataFrame, pandas.DataFrame, or pyarrow.Table"):
        lme_python.lmer(FORMULA, data=[1, 2, 3], reml=True)
