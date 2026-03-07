import pytest
import polars as pl
import lme_python

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
    
    # Test predictions
    preds = model.predict(df)
    assert len(preds) == 180
    # First prediction for sleepstudy should match R output closely
    assert abs(preds[0] - 251.405105) < 1e-4

if __name__ == "__main__":
    test_lmer_sleepstudy()
