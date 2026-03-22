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
    file_path = os.path.join("tests", "data", "grouseticks.csv")
    if not os.path.exists(file_path):
        print(f"Could not find the dataset at {file_path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    print(f"Loading data from {file_path}...")
    
    # 1. Load the dataset
    data = pd.read_csv(file_path)
    print(f"Successfully loaded {len(data)} rows.")

    # 2. Fit the Poisson GLMM model
    print("\nFitting Poisson GLMM: TICKS ~ YEAR + HEIGHT + (1 | BROOD)")
    print("Evaluating Maximum Likelihood...")
    
    # For Poisson MixedLM in statsmodels:
    try:
        model = sm.PoissonBayesMixedGLM.from_formula(
            "TICKS ~ YEAR + HEIGHT",
            "BROOD",
            data
        )
        # Note: statsmodels uses a different penalized framework for GLMMs 
        # (BayesMixedGLM) instead of Laplace approximation, so the math might
        # drift more significantly than LMMs, but it's the closest analog.
        result = model.fit_vb()
        print("\n=== Model Summary ===")
        print(result.summary())
    except Exception as e:
        print(f"\nStatsmodels Poisson GLMM failed: {e}")
        # Sometimes statsmodels struggles with GLMM convergence
        
if __name__ == "__main__":
    main()
