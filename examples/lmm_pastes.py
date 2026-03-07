import polars as pl
import lme_python as lme

def main():
    print("Loading data from tests/data/pastes.csv...")
    try:
        df = pl.read_csv("tests/data/pastes.csv")
    except FileNotFoundError:
        print("Could not find the dataset. Please run from the root of lme-rs.")
        return

    print("\nFitting model: strength ~ 1 + (1 | batch/cask)")
    
    # Fit the Linear Mixed Model
    model = lme.lmer("strength ~ 1 + (1 | batch/cask)", df, True)
    
    print("\n=== Model Summary ===")
    print(model.summary())

    print("\n=== Predictions ===")
    print("Generating predictions for population-level...")
    
    newdata = pl.DataFrame({
        "batch": ["A", "B", "C"],
        "cask": ["a", "b", "c"]
    })

    # lme_python predict defaults to population-level
    preds = model.predict(newdata)
    
    print("Predictions:")
    print(preds)

if __name__ == "__main__":
    main()
