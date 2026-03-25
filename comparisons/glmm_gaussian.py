"""
Gaussian GLMM on sleepstudy using statsmodels MixedLM.

For the identity link and Gaussian noise, `glmer(..., family = gaussian)` in R
is closely related to a linear mixed model with the same formula. Here we fit
`Reaction ~ Days + (1 | Subject)` as a random-intercept MixedLM (REML), which
is the standard statsmodels analogue for this comparison.
"""

from __future__ import annotations

import os
import sys

try:
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
except ImportError:
    print("Please install pandas and statsmodels to run this example:")
    print("pip install pandas statsmodels")
    sys.exit(1)


def main() -> None:
    path = os.path.join("tests", "data", "sleepstudy.csv")
    if not os.path.exists(path):
        print(f"Could not find the dataset at {path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    data = pd.read_csv(path)
    print(f"Loading data from {path} ... ({len(data)} rows)")

    print("\nFitting model: Reaction ~ Days + (1 | Subject) [REML, statsmodels MixedLM]")
    model = smf.mixedlm(
        "Reaction ~ Days",
        data,
        groups=data["Subject"],
        re_formula="~1",
    )
    result = model.fit(reml=True)
    print("\n=== Model Summary ===")
    print(result.summary())

    print("\n=== Predictions (population-level, fixed part only) ===")
    fe = result.fe_params
    days = np.array([0.0, 1.0])
    preds = fe["Intercept"] + fe["Days"] * days
    print(preds)


if __name__ == "__main__":
    main()
