"""Ordered factors, adjusted comparisons, and an unconditional null bootstrap."""

import math

import lme_python as lme
import polars as pl


def main():
    rows = []
    for subject in range(18):
        for visit in range(4):
            # Keep observed visits from incomplete subjects explicitly.
            if subject % 4 == 0 and visit == 2:
                continue
            arm = subject % 3
            rows.append(
                {
                    "subject": f"s{subject}",
                    "arm": ["control", "low", "high"][arm],
                    "visit": float(visit),
                    "response": 5
                    + 0.4 * arm
                    + 0.3 * visit
                    + 0.1 * arm * visit
                    + math.sin(subject * 1.37)
                    + 0.4 * math.cos(subject * 2.1 + visit * 1.8),
                }
            )
    data = pl.DataFrame(rows)
    factors = {"arm": lme.FactorSpec(["control", "low", "high"], "sum")}
    full = lme.prepare_lmer("response ~ arm*visit + (1|subject)", data, factors=factors)
    null = lme.prepare_lmer("response ~ arm + visit + (1|subject)", data, factors=factors)
    fit = full.fit()
    fit.with_satterthwaite(data)
    means = fit.emmeans_grid(["arm"], data, at={"visit": 2.0}, ddf_method="satterthwaite")
    assert [cell["arm"] for cell in means.cells] == ["control", "low", "high"]
    comparisons = fit.emmeans_grid_pairs(
        ["arm"], data, at={"visit": 2.0}, adjust="holm", ddf_method="satterthwaite"
    )
    assert len(comparisons.p_adjust) == 3
    # Nineteen replicates only exercise the API; choose a production count for
    # the desired Monte Carlo precision and inspect all failures.
    bootstrap = lme.bootstrap_lrt(full, null, 19, seed=127, n_jobs=2)
    assert bootstrap.valid == bootstrap.requested, bootstrap.errors
    print("Adjusted means at visit 2:", means.estimate)
    print("Holm p-values:", comparisons.p_adjust)
    print("Null-bootstrap p-value:", bootstrap.p_value, "valid:", bootstrap.valid)


if __name__ == "__main__":
    main()
