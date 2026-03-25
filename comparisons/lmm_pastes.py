"""
Pastes LMM using statsmodels.

`lme4::lmer(strength ~ 1 + (1 | batch/cask))` uses two variance components
(batch and batch:cask). `statsmodels` `MixedLM` supports only one grouping
column per fit. We group by the `sample` column (`batch:cask` labels), which
matches the inner random intercept stratum only — not the full nested model.
For exact nested parity, use R, Julia, or Rust examples in this folder.
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
    path = os.path.join("tests", "data", "pastes.csv")
    if not os.path.exists(path):
        print(f"Could not find the dataset at {path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    data = pd.read_csv(path)
    print("\nFitting model: strength ~ 1 + (1 | sample)  [sample = batch:cask level]")
    print("(statsmodels single-grouping proxy for nested batch/cask)")

    model = smf.mixedlm("strength ~ 1", data, groups=data["sample"])
    result = model.fit(reml=True)
    print("\n=== Model Summary ===")
    print(result.summary())

    print("\n=== Predictions (population-level) ===")
    fe = result.fe_params
    print("Intercept (population mean):", float(fe["Intercept"]))


if __name__ == "__main__":
    main()
