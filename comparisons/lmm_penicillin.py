"""
Penicillin crossed random effects using statsmodels.

`lme4::lmer(diameter ~ 1 + (1 | plate) + (1 | sample))` has two crossed random
intercepts. `MixedLM` allows one grouping factor per fit; we fit the plate
stratum only as a runnable benchmark (not identical to the full crossed model).
"""

from __future__ import annotations

import os
import sys

try:
    import pandas as pd
    import statsmodels.formula.api as smf
except ImportError:
    print("Please install pandas and statsmodels:")
    print("pip install pandas statsmodels")
    sys.exit(1)


def main() -> None:
    path = os.path.join("tests", "data", "penicillin.csv")
    if not os.path.exists(path):
        print(f"Could not find the dataset at {path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    data = pd.read_csv(path)
    print("\nFitting model: diameter ~ 1 + (1 | plate)  [single grouping; crossed RE not in one MixedLM]")
    model = smf.mixedlm("diameter ~ 1", data, groups=data["plate"])
    result = model.fit(reml=True)
    print("\n=== Model Summary ===")
    print(result.summary())

    print("\n=== Predictions (population-level) ===")
    fe = result.fe_params
    print("Intercept:", float(fe["Intercept"]))


if __name__ == "__main__":
    main()
