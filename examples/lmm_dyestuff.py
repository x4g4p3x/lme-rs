import polars as pl
import lme_python as lme

def main():
    print("Loading data from tests/data/dyestuff.csv...")
    try:
        df = pl.read_csv("tests/data/dyestuff.csv")
    except FileNotFoundError:
        print("Could not find the dataset. Please run from the root of lme-rs.")
        return

    print("\nFitting model: Yield ~ 1 + (1 | Batch)")
    
    # Fit the Linear Mixed Model
    model = lme.lmer("Yield ~ 1 + (1 | Batch)", df, True)
    
    print("\n=== Model Summary ===")
    print(model.summary())

if __name__ == "__main__":
    main()
