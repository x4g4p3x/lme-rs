import polars as pl
import lme_python as lme

def main():
    print("Loading data from tests/data/cbpp_binary.csv...")
    try:
        df = pl.read_csv("tests/data/cbpp_binary.csv")
    except FileNotFoundError:
        print("Could not find the dataset. Please run from the root of lme-rs.")
        return


    print("\nFitting Binomial GLMM: y ~ period2 + period3 + period4 + (1 | herd)")
    
    # Fit the Generalized Linear Mixed Model
    model = lme.glmer("y ~ period2 + period3 + period4 + (1 | herd)", df, "binomial")
    
    print("\n=== Model Summary ===")
    print(model.summary())

    print("\n=== Predictions (Probabilities) ===")
    print("Generating predictions for herd 1 across periods...")
    
    newdata = pl.DataFrame({
        "herd": ["1", "1", "1", "1"],
        "period2": [0.0, 1.0, 0.0, 0.0],
        "period3": [0.0, 0.0, 1.0, 0.0],
        "period4": [0.0, 0.0, 0.0, 1.0]
    })

    import numpy as np
    
    # lme_python predict currently outputs on the link scale (log-odds)
    # We apply the inverse-logit function to get probabilities
    def invlogit(x):
        return 1.0 / (1.0 + np.exp(-x))
        
    eta = model.predict(newdata)
    preds = [invlogit(x) for x in eta]
    
    print("Predictions:")
    print(preds)

if __name__ == "__main__":
    main()
