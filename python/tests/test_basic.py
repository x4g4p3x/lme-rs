import pytest
import polars as pl
import lme_python
import math

def test_lmer_sleepstudy():
    # Load identical data used in rust tests
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    
    # Fit model natively via rust
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    summary = model.summary()
    
    print("\n--- Model Summary ---")
    print(summary)
    
    # Check that REML objective is embedded in the summary string
    assert "1743.6" in summary
    assert "611.9033" in summary  # Intercept variance
    
    # Check properties
    assert model.converged is True
    assert model.num_obs == 180
    assert len(model.coefficients) == 2
    assert model.fixed_names == ["(Intercept)", "Days"]
    assert model.sigma2 is not None
    assert abs(model.sigma2 - 654.94) < 0.1
    assert model.log_likelihood is not None
    assert model.aic is not None
    assert model.bic is not None
    assert model.deviance is not None
    assert model.ranef is not None
    assert isinstance(model.ranef, list)
    assert len(model.ranef) > 0
    assert len(model.ranef[0]) == 4
    assert model.ranef[0][1] == "308"
    assert model.var_corr is not None
    assert isinstance(model.var_corr, list)
    assert len(model.var_corr) > 0
    assert len(model.var_corr[0]) == 5

    # Test confint
    ci = model.confint(0.95)
    assert len(ci) == 2
    assert ci[0][0] < model.coefficients[0] < ci[0][1]
    
    # Test predictions
    preds = model.predict(df)
    assert len(preds) == 180
    # First prediction for sleepstudy should match R output closely (population level)
    assert abs(preds[0] - 251.405105) < 1e-4

    # Test conditional predictions
    cond_preds = model.predict_conditional(df, allow_new_levels=False)
    assert len(cond_preds) == 180
    assert abs(cond_preds[0] - preds[0]) > 0.1 # Subject level effect should make it differ from population

    # Test residuals and fitted
    assert len(model.residuals) == 180
    assert len(model.fitted) == 180
    
def test_glmer_poisson():
    df = pl.read_csv("../tests/data/grouseticks.csv")
    # Ticks ~ Year + Height + (1|BROOD)
    model = lme_python.glmer("TICKS ~ YEAR + HEIGHT + (1 | BROOD)", data=df, family_name="poisson")
    
    assert model.converged is True
    assert model.sigma2 is None # Poisson has no dispersion
    assert len(model.coefficients) == 3
    assert model.ranef is not None
    assert model.var_corr is not None
    
    preds_link = model.predict(df)
    preds_resp = model.predict_response(df)
    
    # Link scale (log) should be different from response scale
    assert preds_link[0] != preds_resp[0]
    assert abs(math.exp(preds_link[0]) - preds_resp[0]) < 1e-6
    
def test_glmer_binomial():
    df = pl.read_csv("../tests/data/cbpp_binary.csv")
    model = lme_python.glmer("y ~ period2 + period3 + period4 + (1 | herd)", data=df, family_name="binomial")
    
    assert model.converged is True
    assert model.sigma2 is None # Binomial has no dispersion
    assert len(model.coefficients) == 4
    
    preds_resp = model.predict_response(df)
    # Binomial response predictions should be probabilities [0, 1]
    assert all(0.0 <= p <= 1.0 for p in preds_resp)

def test_glmer_predict_conditional_response_poisson():
    df = pl.read_csv("../tests/data/grouseticks.csv")
    model = lme_python.glmer("TICKS ~ YEAR + HEIGHT + (1 | BROOD)", data=df, family_name="poisson")

    preds_pop = model.predict_response(df)
    preds_cond = model.predict_conditional_response(df, allow_new_levels=False)

    assert len(preds_cond) == len(preds_pop) == df.height
    # At least one subject-level/cluster effect should change predictions.
    assert abs(preds_cond[0] - preds_pop[0]) > 1e-6

def test_conditional_response_unseen_levels():
    df = pl.read_csv("../tests/data/grouseticks.csv")
    model = lme_python.glmer("TICKS ~ YEAR + HEIGHT + (1 | BROOD)", data=df, family_name="poisson")

    # Create a dataframe with an entirely unseen group level.
    new_level_df = pl.DataFrame({
        "TICKS": [10.0],
        "YEAR": [2000.0],
        "HEIGHT": [1.2],
        "BROOD": ["UNSEEN_BROOD_999"]
    })

    # allow_new_levels=False should error.
    with pytest.raises(ValueError):
        model.predict_conditional_response(new_level_df, allow_new_levels=False)

    # allow_new_levels=True should fall back to population-level predictions.
    preds_cond = model.predict_conditional_response(new_level_df, allow_new_levels=True)
    preds_pop = model.predict_response(new_level_df)
    assert len(preds_cond) == 1
    assert abs(preds_cond[0] - preds_pop[0]) < 1e-6

def test_invalid_family():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    with pytest.raises(ValueError, match="Unsupported or invalid family"):
        lme_python.glmer("Reaction ~ Days + (Days | Subject)", data=df, family_name="invalid_family")

def test_missing_column():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    with pytest.raises(ValueError):
         # MissingCol is not in sleepstudy
        lme_python.lmer("Reaction ~ MissingCol + (1 | Subject)", data=df)

def test_lmer_ml_estimation():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=False)
    assert model.converged is True
    # The REML objective is larger than ML objective for the same model
    assert "REML criterion" not in model.summary()

def test_glmer_gamma():
    df = pl.read_csv("../tests/data/dyestuff.csv")
    model = lme_python.glmer("Yield ~ 1 + (1 | Batch)", data=df, family_name="gamma")
    assert model.converged is True
    assert model.sigma2 is not None # Gamma has dispersion
    assert len(model.coefficients) == 1

def test_predictions_with_new_data():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    
    # Subset to just 2 rows
    subset_df = df.head(2)
    preds = model.predict(subset_df)
    assert len(preds) == 2
    
def test_conditional_predictions_unseen_levels():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    
    # Create a dataframe with an entirely unseen subject level
    new_level_df = pl.DataFrame({
        "Reaction": [250.0],
        "Days": [0],
        "Subject": ["UNSEEN_SUBJECT_999"]
    })
    
    # Should throw an error natively when allow_new_levels=False
    with pytest.raises(ValueError):
        model.predict_conditional(new_level_df, allow_new_levels=False)
        
    # Should fallback to population level securely when allow_new_levels=True
    cond_preds = model.predict_conditional(new_level_df, allow_new_levels=True)
    pop_preds = model.predict(new_level_df)
    assert len(cond_preds) == 1
    assert abs(cond_preds[0] - pop_preds[0]) < 1e-6
    
def test_predict_missing_columns():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    
    bad_df = pl.DataFrame({"Reaction": [250.0], "Subject": ["308"]}) # Missing 'Days'
    with pytest.raises(ValueError):
        model.predict(bad_df)

def test_std_errors():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)
    
    se = model.std_errors
    assert se is not None
    assert len(se) == 2
    assert se[0] > 0
    assert se[1] > 0

def test_inference_getters_default_none():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)

    assert model.robust_se is None
    assert model.robust_t is None
    assert model.robust_p_values is None
    assert model.satterthwaite_dfs is None
    assert model.satterthwaite_p_values is None
    assert model.kenward_roger_dfs is None
    assert model.kenward_roger_p_values is None

def test_simulate():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=True)

    sims = model.simulate(3)
    assert len(sims) == 3
    assert all(len(v) == model.num_obs for v in sims)
    # Spot-check finiteness (avoid scanning the entire simulation tensor).
    for v in sims:
        for x in v[:5]:
            assert math.isfinite(x)

def test_simulate_poisson_nonnegative_counts():
    df = pl.read_csv("../tests/data/grouseticks.csv")
    model = lme_python.glmer("TICKS ~ YEAR + HEIGHT + (1 | BROOD)", data=df, family_name="poisson")

    sims = model.simulate(3)
    assert len(sims) == 3
    for v in sims:
        for x in v[:20]:
            assert x >= 0.0
            assert abs(x - round(x)) < 1e-9

def test_simulate_binomial_binary():
    df = pl.read_csv("../tests/data/cbpp_binary.csv")
    model = lme_python.glmer("y ~ period2 + period3 + period4 + (1 | herd)", data=df, family_name="binomial")

    sims = model.simulate(3)
    assert len(sims) == 3
    for v in sims:
        for x in v[:20]:
            assert x in (0.0, 1.0)

def test_simulate_gamma_positive():
    df = pl.read_csv("../tests/data/dyestuff.csv")
    model = lme_python.glmer("Yield ~ 1 + (1 | Batch)", data=df, family_name="gamma")

    sims = model.simulate(3)
    assert len(sims) == 3
    for v in sims:
        for x in v[:20]:
            assert x > 0.0
            assert math.isfinite(x)

def test_anova_lrt_nested_lmer():
    df = pl.read_csv("../tests/data/sleepstudy.csv")

    fit0 = lme_python.lmer("Reaction ~ 1 + (1 | Subject)", data=df, reml=False)
    fit1 = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=df, reml=False)

    res = lme_python.anova(fit0, fit1)
    (n_params_0, n_params_1, dev0, dev1, chi_sq, df_diff, p_value, formula_0, formula_1) = res

    assert n_params_0 < n_params_1
    assert df_diff > 0
    assert chi_sq >= 0.0
    assert 0.0 <= p_value <= 1.0
    assert isinstance(formula_0, str) and isinstance(formula_1, str)

def test_with_robust_se():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_robust_se(subset_df, cluster_col=None)
    summary = model.summary()
    assert "[Robust]" in summary

def test_with_cluster_robust_se():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_robust_se(subset_df, cluster_col="Subject")

    robust_se = model.robust_se
    robust_t = model.robust_t
    robust_p = model.robust_p_values

    assert robust_se is not None and robust_t is not None and robust_p is not None
    assert len(robust_se) == len(model.coefficients)
    assert len(robust_t) == len(model.coefficients)
    assert len(robust_p) == len(model.coefficients)

def test_with_satterthwaite():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_satterthwaite(subset_df)
    summary = model.summary()
    assert "Satterthwaite" in summary

    assert model.satterthwaite_dfs is not None
    assert model.satterthwaite_p_values is not None
    assert len(model.satterthwaite_dfs) == len(model.coefficients)
    assert len(model.satterthwaite_p_values) == len(model.coefficients)

def test_with_kenward_roger():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_kenward_roger(subset_df)
    summary = model.summary()
    assert "Kenward-Roger" in summary

    assert model.kenward_roger_dfs is not None
    assert model.kenward_roger_p_values is not None
    assert len(model.kenward_roger_dfs) == len(model.coefficients)
    assert len(model.kenward_roger_p_values) == len(model.coefficients)

def test_robust_inference_values():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_robust_se(subset_df, cluster_col=None)

    robust_se = model.robust_se
    robust_t = model.robust_t
    robust_p = model.robust_p_values

    assert robust_se is not None and robust_t is not None and robust_p is not None
    assert len(robust_se) == len(model.coefficients)
    assert len(robust_t) == len(model.coefficients)
    assert len(robust_p) == len(model.coefficients)
    assert all(math.isfinite(x) for x in robust_se[:])
    assert all(0.0 <= p <= 1.0 for p in robust_p)

def test_anova_satterthwaite_fixed_effects():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_satterthwaite(subset_df)

    terms, num_df, den_df, f_value, p_value, method = model.anova("satterthwaite")

    assert terms == ["Days"]
    assert len(num_df) == len(den_df) == len(f_value) == len(p_value) == 1
    assert 0.0 <= p_value[0] <= 1.0
    assert den_df[0] > 0.0
    assert isinstance(method, str)

def test_anova_kenward_roger_fixed_effects():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_kenward_roger(subset_df)

    terms, num_df, den_df, f_value, p_value, method = model.anova("kenward_roger")

    assert terms == ["Days"]
    assert len(num_df) == len(den_df) == len(f_value) == len(p_value) == 1
    assert 0.0 <= p_value[0] <= 1.0
    assert den_df[0] > 0.0
    assert isinstance(method, str)

def test_anova_invalid_ddf_method():
    df = pl.read_csv("../tests/data/sleepstudy.csv")
    subset_df = df.head(60)

    model = lme_python.lmer("Reaction ~ Days + (Days | Subject)", data=subset_df, reml=True)
    model.with_satterthwaite(subset_df)

    with pytest.raises(ValueError, match="Unsupported ddf_method"):
        model.anova("not_a_real_method")

if __name__ == "__main__":
    test_lmer_sleepstudy()
    test_glmer_poisson()
    test_glmer_binomial()
    test_glmer_predict_conditional_response_poisson()
    test_conditional_response_unseen_levels()
    test_lmer_ml_estimation()
    test_glmer_gamma()
    test_predictions_with_new_data()
    test_conditional_predictions_unseen_levels()
    test_predict_missing_columns()
    test_std_errors()
    test_inference_getters_default_none()
    test_simulate()
    test_simulate_poisson_nonnegative_counts()
    test_simulate_binomial_binary()
    test_simulate_gamma_positive()
    test_anova_lrt_nested_lmer()
    test_with_robust_se()
    test_with_cluster_robust_se()
    test_with_satterthwaite()
    test_with_kenward_roger()
    test_robust_inference_values()
    test_anova_satterthwaite_fixed_effects()
    test_anova_kenward_roger_fixed_effects()
    test_anova_invalid_ddf_method()
    print("All tests passed natively")
