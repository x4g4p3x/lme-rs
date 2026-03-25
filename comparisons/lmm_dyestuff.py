"""Dyestuff intercept-only LMM using statsmodels (cf. `comparisons/lmm_dyestuff.rs`)."""

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
    path = os.path.join("tests", "data", "dyestuff.csv")
    if not os.path.exists(path):
        print(f"Could not find the dataset at {path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    data = pd.read_csv(path)
    print("\nFitting model: Yield ~ 1 + (1 | Batch)")
    model = smf.mixedlm("Yield ~ 1", data, groups=data["Batch"])
    result = model.fit(reml=True)
    print("\n=== Model Summary ===")
    print(result.summary())


if __name__ == "__main__":
    main()
