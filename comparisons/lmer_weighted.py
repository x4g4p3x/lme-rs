"""
Weighted LMM parity script using statsmodels.

`statsmodels.regression.mixed_linear_model.MixedLM.fit` does not apply prior
case weights the way `lme4::lmer(..., weights=)` does. To keep this script
self-contained (no `lme_python` wheel), we approximate prior weights by
replicating each row a number of times proportional to its weight, then fit
REML MixedLM on the expanded table. Coefficients will not match R/Rust exactly;
use those implementations for numerical parity.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import pandas as pd
    import statsmodels.formula.api as smf
except ImportError:
    print("Please install pandas and statsmodels to run this example:")
    print("pip install pandas statsmodels numpy")
    sys.exit(1)


def main() -> None:
    path = os.path.join("tests", "data", "sleepstudy.csv")
    if not os.path.exists(path):
        print(f"Could not find the dataset at {path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    data = pd.read_csv(path)
    n = len(data)
    w = np.array([0.5 + (i % 5) * 0.1 for i in range(n)], dtype=float)
    mult = np.maximum(1, (w / w.min() * 8.0).round().astype(int))
    expanded = data.loc[np.repeat(data.index.values, mult)].reset_index(drop=True)

    print("Loading tests/data/sleepstudy.csv ...")
    print(f"Expanded row count for approximate weights: {len(expanded)} (from {n})")
    print("\nFitting model: Reaction ~ Days + (Days | Subject) [REML, expanded-data proxy]")

    model = smf.mixedlm(
        "Reaction ~ Days",
        expanded,
        groups=expanded["Subject"],
        re_formula="~Days",
    )
    result = model.fit(reml=True)
    print("\n=== Model Summary ===")
    print(result.summary())

    print("\n=== Predictions (population-level, fixed part only) ===")
    fe = result.fe_params
    days = np.array([0.0, 1.0, 5.0, 10.0])
    preds = fe["Intercept"] + fe["Days"] * days
    print(preds)


if __name__ == "__main__":
    main()
