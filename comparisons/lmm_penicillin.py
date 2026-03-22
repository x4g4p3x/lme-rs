import polars as pl
import lme_python as lme

def main():
    print("Loading data from tests/data/penicillin.csv...")
    try:
        df = pl.read_csv("tests/data/penicillin.csv")
    except FileNotFoundError:
        print("Could not find the dataset. Please run from the root of lme-rs.")
        return

    print("\nFitting model: diameter ~ 1 + (1 | plate) + (1 | sample)")
    
    # Fit the Linear Mixed Model
    model = lme.lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", df, True)
    
    print("\n=== Model Summary ===")
    print(model.summary())

    print("\n=== Predictions ===")
    print("Generating predictions for new plates and samples...")
    
    newdata = pl.DataFrame({
        "plate": ["a", "b", "c", "d"],
        "sample": ["A", "C", "E", "F"]
    })

    # lme_python predict defaults to population-level
    preds = model.predict(newdata)
    
    print("Predictions:")
    print(preds)

if __name__ == "__main__":
    main()
