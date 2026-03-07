import os
import sys

try:
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    print("Please install pandas and statsmodels to run this example:")
    print("pip install pandas statsmodels")
    sys.exit(1)

def main():
    file_path = os.path.join("tests", "data", "sleepstudy.csv")
    if not os.path.exists(file_path):
        print(f"Could not find the dataset at {file_path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    print(f"Loading data from {file_path}...")
    
    # 1. Load the dataset
    data = pd.read_csv(file_path)
    print(f"Successfully loaded {len(data)} rows.")

    # 2. Fit the model
    # In statsmodels, equivalent to lme4's: Reaction ~ Days + (Days | Subject)
    print("\nFitting model: Reaction ~ Days + (Days | Subject)")
    print("Evaluating Restricted Maximum Likelihood (REML)...")
    
    # groups specifies the outer grouping factor (Subject)
    # re_formula specifies the random slopes (Days)
    model = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"], re_formula="~Days")
    
    # For closer parity with lme4 bounded optimization, we can just use defaults, which try very hard
    result = model.fit(reml=True)

    # 3. Print the summary
    print("\n=== Model Summary ===")
    print(result.summary())
    
    # Also print the variance components explicitly to be clear:
    print("\nVariance Components:")
    print(result.cov_re.to_string())
    print(f"Residual Variance: {result.scale:.4f}")

    # 4. Generate Predictions
    print("\n=== Predictions ===")
    print("Generating predictions for Subject 308 at Days 0, 1, 5, and 10...")

    newdata = pd.DataFrame({
        'Intercept': [1.0, 1.0, 1.0, 1.0], # statsmodels often expects the intercept in exog
        'Days': [0.0, 1.0, 5.0, 10.0],
        'Subject': ["308", "308", "308", "308"]
    })

    # Predict population-level expectations
    preds = result.predict(exog=newdata)
    
    print("\nPredictions:")
    print(preds.to_string(index=False))

if __name__ == "__main__":
    main()
