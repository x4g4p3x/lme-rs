import pandas as pd
import statsmodels.formula.api as smf

# Load data
data = pd.read_csv("tests/data/sleepstudy.csv")
print(f"Loaded {len(data)} rows from tests/data/sleepstudy.csv\n")

# Fit ML model
print("Fitting model: Reaction ~ Days + (Days | Subject)")
print("Evaluating Maximum Likelihood (REML = False)...")
model = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"], re_formula="~Days")
result = model.fit(reml=False)

# Print summary
print("\n=== Model Summary ===")
print(result.summary())

# Predict
newdata = pd.DataFrame({
    "Days": [0, 1, 5, 10],
    "Subject": [308, 308, 308, 308]
})
preds = result.predict(newdata)
print("\n=== Predictions ===")
print(preds)
