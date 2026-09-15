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


def test_pandas_nullable_categories_and_unused_objects():
    """A repeated-measures frame may carry metadata that cannot be converted to Arrow."""
    pd = pytest.importorskip("pandas")
    df = pl.read_csv(DATA_PATH)
    pdf = pd.DataFrame(df.to_dict(as_series=False)).convert_dtypes()
    pdf["Subject"] = pd.Categorical(pdf["Subject"].astype(str), ordered=True)
    pdf["metadata"] = [object() for _ in range(len(pdf))]
    pdf[17] = [object() for _ in range(len(pdf))]
    original = pdf.copy(deep=True)

    fit = lme_python.lmer(FORMULA, pdf)
    ref = _reference_fit(df)
    assert fit.coefficients == pytest.approx(ref.coefficients, abs=1e-6)
    assert fit.predict(pdf) == pytest.approx(ref.predict(df), abs=1e-6)
    pd.testing.assert_frame_equal(pdf, original)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["Late", "Early"])])
def test_polars_dictionary_fixed_effects(dtype):
    df = pl.read_csv(DATA_PATH).with_columns(
        pl.when(pl.col("Days") < 5).then(pl.lit("Early")).otherwise(pl.lit("Late")).alias("phase")
    )
    categorical = df.with_columns(pl.col("phase").cast(dtype))
    formula = "Reaction ~ phase + (1 | Subject)"
    ref = lme_python.lmer(formula, df)
    fit = lme_python.lmer(formula, categorical)
    assert fit.predict(categorical) == pytest.approx(ref.predict(df), abs=1e-6)


def test_pandas_categorical_fixed_effect_matches_string_input():
    pd = pytest.importorskip("pandas")
    df = pl.read_csv(DATA_PATH).with_columns(
        pl.when(pl.col("Days") < 5).then(pl.lit("Early")).otherwise(pl.lit("Late")).alias("phase")
    )
    pdf = pd.DataFrame(df.to_dict(as_series=False)).convert_dtypes()
    pdf["phase"] = pd.Categorical(pdf["phase"], categories=["Late", "Early"], ordered=True)
    formula = "Reaction ~ phase + (1 | Subject)"
    fit = lme_python.lmer(formula, pdf)
    ref = lme_python.lmer(formula, df)
    assert fit.predict(pdf) == pytest.approx(ref.predict(df), abs=1e-6)
