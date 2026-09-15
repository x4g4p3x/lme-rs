"""Repeated measurements with explicit sum coding and a null bootstrap.

Run with an installed lme-python; only Polars and the standard library are needed.
Inputs here are already aggregated to one row per animal and condition. This
example simulates new animals under the null model. The conditional-response
bootstrap offered by fit.boot() answers a different resampling question.
"""

from itertools import product
from math import isfinite, sqrt
from random import Random

import lme_python as lme
import polars as pl

BETWEEN = ["Control", "Treatment"]
WITHIN = ["Day 1", "Day 2", "Day 10"]


def sum_code(frame, column, levels, prefix):
    """Encode k declared levels into k-1 columns; the last level is all -1.

    Keeping the same level lists for fitting and prediction preserves the
    declared order and Patsy Sum semantics without requiring its formula parser.
    """
    values = frame[column].to_list()
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ValueError("levels must contain at least two distinct labels")
    if any(value not in levels for value in values):
        raise ValueError(f"missing or unknown level in {column}")
    names = [f"{prefix}{i}" for i in range(len(levels) - 1)]
    expressions = [
        pl.when(pl.col(column) == level)
        .then(1.0)
        .when(pl.col(column) == levels[-1])
        .then(-1.0)
        .otherwise(0.0)
        .alias(name)
        for name, level in zip(names, levels[:-1])
    ]
    return frame.with_columns(expressions), names


def encode(frame):
    frame, between = sum_code(frame, "between", BETWEEN, "b")
    frame, within = sum_code(frame, "within", WITHIN, "w")
    interactions = [f"{b}:{w}" for b in between for w in within]
    return frame, between, within, interactions


def sample_data(seed=42):
    rng = Random(seed)
    rows = []
    for animal in range(24):
        group = BETWEEN[animal // 12]
        animal_effect = rng.gauss(0, 2)
        for day, phase in enumerate(WITHIN):
            rows.append(
                {
                    "subject_id": str(animal),
                    "between": group,
                    "within": phase,
                    "dv": 10 + 3 * (group == "Treatment") + day + animal_effect + rng.gauss(0, 1),
                }
            )
    return pl.DataFrame(rows)


def simulate_null(fit, data, rng):
    """Random-intercept Gaussian null draws with fresh animal effects."""
    variance = next(row[3] for row in fit.var_corr if row[0] == "subject_id" and row[1] == row[2])
    subjects = data["subject_id"].to_list()
    effects = {subject: rng.gauss(0, sqrt(max(variance, 0))) for subject in dict.fromkeys(subjects)}
    sigma = sqrt(max(fit.sigma2, 0))
    return [
        mean + effects[subject] + rng.gauss(0, sigma)
        for mean, subject in zip(fit.predict(data), subjects)
    ]


def bootstrap_lrt(data, full_formula, null_formula, replicates=19, seed=42):
    """Reuse two fixed designs; compare ML fits on identical simulated rows.

    The default replicate count is a quick demonstration, not adequate precision
    for an analysis. Only successfully converged finite replicate pairs count.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    control = lme.FitControl(require_convergence=True)
    full = lme.prepare_lmer(full_formula, data)
    null = lme.prepare_lmer(null_formula, data)
    full_fit = full.fit(reml=False, control=control)
    null_fit = null.fit(reml=False, control=control)
    observed = 2 * (full_fit.log_likelihood - null_fit.log_likelihood)
    if not isfinite(observed) or observed < -1e-6:
        raise RuntimeError("invalid reference likelihood-ratio statistic")
    observed = max(observed, 0)
    rng = Random(seed)
    statistics = []
    for _ in range(replicates):
        response = simulate_null(null_fit, data, rng)
        try:
            f = full.fit(y=response, reml=False, control=control)
            n = null.fit(y=response, reml=False, control=control)
        except RuntimeError:
            continue
        statistic = 2 * (f.log_likelihood - n.log_likelihood)
        if isfinite(statistic) and statistic >= -1e-6:
            statistics.append(max(statistic, 0))
    if not statistics:
        raise RuntimeError("no valid bootstrap replicate pairs")
    return {
        "p": (1 + sum(value >= observed for value in statistics)) / (1 + len(statistics)),
        "valid": len(statistics),
        "requested": replicates,
        "seed": seed,
    }


def main():
    data, between, within, interactions = encode(sample_data())
    formula = "dv ~ " + " + ".join(between + within + interactions) + " + (1 | subject_id)"
    fit = lme.lmer(formula, data, control=lme.FitControl(require_convergence=True))
    grid = pl.DataFrame(
        [{"between": group, "within": phase} for group, phase in product(BETWEEN, WITHIN)]
    )
    grid, _, _, _ = encode(grid)
    matrix = fit.design_matrix(grid)
    # Treatment minus Control, averaging equally over the declared within levels.
    n = len(WITHIN)
    contrast = [
        sum(row[j] for row in matrix[n:]) / n - sum(row[j] for row in matrix[:n]) / n
        for j in range(len(fit.coefficients))
    ]
    estimate = sum(weight * beta for weight, beta in zip(contrast, fit.coefficients))
    assert isfinite(estimate)
    print("Adjusted cell means:", fit.predict(grid))
    print("Treatment minus Control:", estimate)
    # Main-effect null comparisons use the additive model; the
    # interaction is tested separately against that additive model.
    additive = "dv ~ " + " + ".join(between + within) + " + (1 | subject_id)"
    reduced = "dv ~ " + " + ".join(within) + " + (1 | subject_id)"
    print("New-animal null bootstrap:", bootstrap_lrt(data, additive, reduced))


if __name__ == "__main__":
    main()
