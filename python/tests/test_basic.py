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

if __name__ == "__main__":
    test_lmer_sleepstudy()
    test_glmer_poisson()
    test_glmer_binomial()
    test_lmer_ml_estimation()
    test_glmer_gamma()
    test_predictions_with_new_data()
    test_conditional_predictions_unseen_levels()
    test_predict_missing_columns()
    test_std_errors()
    print("All tests passed natively")
