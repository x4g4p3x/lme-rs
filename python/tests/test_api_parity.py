"""Bindings parity with the public Rust crate surface."""

import math

import lme_python
import polars as pl


def test_model_metadata_getters():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)

    assert fit.formula == "Reaction ~ Days + (Days | Subject)"
    assert fit.family_name is None
    assert fit.link_name is None
    assert fit.family is None
    assert fit.iterations is not None
    assert fit.reml_criterion is not None
    assert fit.beta_t is not None
    assert len(fit.beta_t) == 2


def test_glmer_metadata_getters():
    df = pl.read_csv("../tests/data/cbpp_binary.csv")
    fit = lme_python.glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        data=df,
        family_name="binomial",
    )
    assert fit.family_name == "binomial"
    assert fit.link_name is not None
    assert fit.family == "Binomial"


def test_glmer_explicit_link():
    df = pl.read_csv("../tests/data/cbpp_binary.csv")
    formula = "y ~ period2 + period3 + period4 + (1 | herd)"
    logit = lme_python.glmer(formula, data=df, family_name="binomial")
    probit = lme_python.glmer(formula, data=df, family_name="binomial", link_name="probit")
    assert logit.link_name == "logit"
    assert probit.link_name == "probit"
    diff = sum(abs(a - b) for a, b in zip(logit.coefficients, probit.coefficients))
    assert diff > 1e-4


def test_contrast_matrix_from_names():
    names = ["(Intercept)", "Days"]
    l_mat = lme_python.contrast_matrix_from_names(
        names,
        [[("Days", 1.0)]],
    )
    assert len(l_mat) == 1
    assert len(l_mat[0]) == 2
    assert l_mat[0][1] == 1.0


def test_test_contrast_vs():
    df = pl.read_csv("../tests/data/sleepstudy.csv").head(60)
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    fit.with_satterthwaite(df)

    beta_h = [0.0, 0.0]
    res = fit.test_contrast_vs(
        [[0.0, 1.0]],
        beta_h,
        ddf_method="satterthwaite",
    )
    assert res.num_df == 1.0
    assert res.den_df > 0.0
    assert res.f_value >= 0.0
    assert 0.0 <= res.p_value <= 1.0
    assert "Satterthwaite" in res.method


def test_nlmer_orange():
    df = pl.read_csv("../tests/data/orange.csv")
    start = {"Asym": 200.0, "xmid": 725.0, "scal": 350.0}
    fit = lme_python.nlmer(
        "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
        data=df,
        start=start,
        reml=False,
    )
    assert fit.converged
    assert fit.family_name == "nlmm"
    assert len(fit.coefficients) == 3
    assert fit.theta is not None
    assert all(math.isfinite(x) for x in fit.coefficients)

    pop = fit.predict(df)
    cond = fit.predict_conditional(df, allow_new_levels=False)
    assert len(pop) == df.height
    assert len(cond) == df.height
    assert sum(abs(p - c) for p, c in zip(pop, cond)) > 1.0
    for c, f in zip(cond, fit.fitted):
        assert abs(c - f) < 1e-3


def test_lmer_weighted_python():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    n = df.height
    w = [0.5 + (i % 5) * 0.1 for i in range(n)]
    uw = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    ww = lme_python.lmer_weighted(
        "Reaction ~ Days + (Days | Subject)", data=df, reml=True, weights=w
    )
    assert abs(uw.sigma2 - ww.sigma2) > 1e-6


def test_glmer_weighted_python():
    df = pl.read_csv("../tests/data/cbpp_binary.csv")
    n = df.height
    w = [0.5 + (i % 7) * 0.1 for i in range(n)]
    uw = lme_python.glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        data=df,
        family_name="binomial",
    )
    ww = lme_python.glmer_weighted(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        data=df,
        family_name="binomial",
        weights=w,
    )
    assert (uw.deviance - ww.deviance).__abs__() > 1e-6


def test_anova_type_ii():
    df = pl.read_csv("../tests/data/sleepstudy.csv").head(60)
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    fit.with_satterthwaite(df)
    tab = fit.anova(ddf_method="satterthwaite", anova_type="II")
    assert tab.terms == ["Days"]


def test_anova_type_i_and_linear_hypothesis():
    df = pl.read_csv("../tests/data/pastes.csv")
    fit = lme_python.lmer("strength ~ cask + (1 | batch)", data=df, reml=True)
    fit.with_satterthwaite(df)
    tab_i = fit.anova(ddf_method="satterthwaite", anova_type="I")
    assert tab_i.terms == ["cask"]
    lh = fit.linear_hypothesis("cask")
    tab_iii = fit.anova(ddf_method="satterthwaite", anova_type="III")
    assert abs(lh.f_value - tab_iii.f_value[0]) < 1e-10


def test_lm_matrix():
    y = [1.0, 2.0, 3.0, 4.0]
    x = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    fit = lme_python.lm_matrix(y, x)
    assert len(fit.coefficients) == 2
    assert fit.num_obs == 4
    assert all(math.isfinite(c) for c in fit.coefficients)


def test_contrast_matrix_indices():
    mat = lme_python.contrast_matrix(3, [[(1, 1.0)], [(0, 1.0), (2, -1.0)]])
    assert len(mat) == 2
    assert mat[0][1] == 1.0


def test_extra_getters():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    fit = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    assert fit.b == fit.coefficients
    assert fit.beta_se is not None
    assert fit.u is not None
    assert fit.fixed_term_assign is not None
