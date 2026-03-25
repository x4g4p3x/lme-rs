"""
Binomial GLMM on `cbpp_binary.csv` using statsmodels `BinomialBayesMixedGLM`.

This is a native Python mixed binomial fit (variational Bayes), not Laplace / AGQ
like `lme4::glmer`. It exists so `scripts/run_cross_language_benchmarks.py` runs
without installing the `lme_python` extension. For Laplace parity, use the Rust
example or R script in this folder.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def invlogit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    path = os.path.join("tests", "data", "cbpp_binary.csv")
    if not os.path.exists(path):
        print(f"Could not find the dataset at {path}")
        print("Please run this example from the root of the lme-rs repository.")
        sys.exit(1)

    print(f"Loading data from {path} ...")
    data = pd.read_csv(path)
    data["herd"] = data["herd"].astype(str)

    print("\nFitting Binomial mixed GLM: y ~ period2 + period3 + period4 + (1 | herd)")
    print("(statsmodels variational Bayes - not identical to lme4 Laplace / AGQ)")

    model = BinomialBayesMixedGLM.from_formula(
        "y ~ period2 + period3 + period4",
        {"herd": "0 + C(herd)"},
        data,
    )
    result = model.fit_vb()

    print("\n=== Model Summary (VB) ===")
    print(result.summary())

    print("\n=== Approximate population-level probabilities (linear predictor from fixed part) ===")
    fe_mean = result.fe_mean
    # Design row for herd 1, four period dummies
    rows = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=float,
    )
    eta = rows @ fe_mean
    print("Eta (log-odds, fixed part only):", eta)
    print("Invlogit:", invlogit(eta))


if __name__ == "__main__":
    main()
