#!/usr/bin/env python3
"""Correctness-gated statsmodels/lme_python comparison on shared synthetic data.

No production backend selection is changed. See docs/MODEL_WORKFLOW_BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
# Fixed before collecting measurements; inference differences are not optimizer tolerances.
TOLERANCES = {
    "objective": (1e-5, 1e-7),
    "beta": (2e-4, 2e-4),
    "variance": (2e-4, 5e-4),
    "prediction": (2e-4, 2e-4),
    "residual": (2e-4, 2e-4),
    "blup": (2e-4, 2e-4),
    "se": (1e-5, 1e-3),
    "ci": (2e-4, 1e-3),
    "p": (1e-4, 1e-3),
    "lrt": (2e-4, 1e-4),
}
LME_TOLERANCE = 1e-6
CASES = {
    "balanced_none": (12, 4, "none", False, 1.2),
    "balanced_within": (24, 4, "within", False, 1.2),
    "incomplete_decomposed": (30, 6, "decomposed", True, 1.2),
    "incomplete_baseline": (24, 4, "baseline", True, 1.2),
    "large_within": (240, 6, "within", True, 1.2),
    "near_boundary": (24, 4, "within", False, 0.0),
}


def dependencies():
    """Import numerical runtimes only after the worker's thread environment is set."""
    global np, pd, patsy, pg, sm, smf, lme, stats, minimize_scalar, threadpool_info
    import lme_python as lme
    import numpy as np
    import pandas as pd
    import patsy
    import pingouin as pg
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats
    from scipy.optimize import minimize_scalar
    from threadpoolctl import threadpool_info


def dataset(name):
    subjects, visits, mode, incomplete, random_sd = CASES[name]
    rng = np.random.default_rng(260915 + list(CASES).index(name))
    levels = ["control", "high", "low"]
    times = ["day0", "day2", "day10", "day12", "day20", "day22"][:visits]
    records = []
    for i in range(subjects):
        group = i % 3
        intercept = rng.normal(0, random_sd)
        baseline = rng.normal(0, 1) + 0.4 * group
        for j in range(visits):
            x = baseline + rng.normal(0, 0.6) + 0.1 * j
            y = (
                4
                + 0.25 * group
                + 0.15 * j
                + 0.035 * group * j
                + (0.0 if name == "near_boundary" else 0.45) * baseline
                - 0.3 * (x - baseline)
                + intercept
                + rng.normal(0, 0.6)
            )
            # Keep the baseline and at least three rows; missingness independent of response.
            if incomplete and j > 1 and rng.random() < (0.1 + 0.1 * group):
                continue
            records.append((f"s{i:04}", levels[group], times[j], x, y))
    data = pd.DataFrame(records, columns=["id", "a", "b", "x", "y"])
    data["a"] = pd.Categorical(data.a, categories=levels, ordered=True)
    data["b"] = pd.Categorical(data.b, categories=times, ordered=True)
    mean = data.groupby("id", observed=True).x.transform("mean")
    data["xw"] = data.x - mean
    data["xb"] = mean - mean.mean()
    base = data.groupby("id", observed=True).x.transform("first")
    data["xbase"] = base - base.mean()
    columns = {"none": [], "within": ["xw"], "decomposed": ["xb", "xw"], "baseline": ["xbase"]}[
        mode
    ]
    return data, columns


def specification(data, columns, null=False):
    suffix = "".join(" + " + c for c in columns)
    fixed = ("a + b" if null else "a*b") + suffix
    formula = ("C(a, Sum) + C(b, Sum)" if null else "C(a, Sum)*C(b, Sum)") + suffix
    factors = {c: lme.FactorSpec(list(data[c].cat.categories), "sum") for c in ("a", "b")}
    return "y ~ " + formula, "y ~ " + fixed + " + (1|id)", factors


def fit_model(backend, data, columns, reml, null=False):
    formula, rust_formula, factors = specification(data, columns, null)
    if backend == "statsmodels":
        # Keep the default optimizer sequence used by formula-based MixedLM callers.
        return smf.mixedlm(formula, data, groups=data.id, re_formula="~1").fit(reml=reml)
    return lme.lmer(
        rust_formula,
        data,
        reml=reml,
        factors=factors,
        control=lme.FitControl(tolerance=LME_TOLERANCE),
    )


def grid_for(data, columns):
    grid = pd.DataFrame(
        [(a, b) for b in data.b.cat.categories for a in data.a.cat.categories], columns=["a", "b"]
    )
    for c in ("a", "b"):
        grid[c] = pd.Categorical(grid[c], categories=data[c].cat.categories, ordered=True)
    for c in columns:
        grid[c] = 0.0
    return grid


def canonical_design(data, columns, null=False):
    formula, _, _ = specification(data, columns, null)
    return patsy.dmatrix(formula.split("~", 1)[1], data, return_type="dataframe")


def column_permutation(reference, actual):
    """Require equal design columns, allowing only permutation (not just equal rank)."""
    if reference.shape != actual.shape:
        raise ValueError("Design dimensions differ")
    matches = []
    for column in reference.T:
        indices = np.flatnonzero(
            np.all(np.isclose(actual, column[:, None], atol=1e-12, rtol=0), axis=0)
        )
        if len(indices) != 1:
            raise ValueError("Design coding differs or is not uniquely identifiable")
        matches.append(int(indices[0]))
    if len(set(matches)) != len(matches):
        raise ValueError("Duplicate design column")
    return matches


def gaussian_reference(data, x, reml):
    """Independent profiled Gaussian likelihood for a random intercept.

    R = I + rho ZZ'; invert each subject's small dense block directly. Optimize
    log(rho), then compare against the exact rho=0 boundary. No mixed-model engine.
    """
    x = np.asarray(x)
    y = data.y.to_numpy()
    groups = list(data.groupby("id", sort=True, observed=True).indices.values())
    n, p = x.shape
    if np.linalg.matrix_rank(x) != p:
        raise ValueError("Reference requires a full-rank design")
    df = n - p if reml else n

    def evaluate(rho, details=False):
        rix = np.empty_like(x)
        riy = np.empty_like(y)
        logdet = 0.0
        for indices in groups:
            covariance = np.eye(len(indices)) + rho * np.ones((len(indices), len(indices)))
            solved = np.linalg.solve(covariance, np.column_stack((x[indices], y[indices])))
            rix[indices], riy[indices] = solved[:, :-1], solved[:, -1]
            logdet += np.linalg.slogdet(covariance)[1]
        information = x.T @ rix
        beta = np.linalg.solve(information, x.T @ riy)
        residual = y - x @ beta
        rss = float(residual @ (riy - rix @ beta))
        sigma2 = rss / df
        objective = df * (np.log(2 * np.pi * sigma2) + 1) + logdet
        if reml:
            objective += np.linalg.slogdet(information)[1]
        if not details:
            return objective
        blup = np.array([rho * residual[ix].sum() / (1 + rho * len(ix)) for ix in groups])
        conditional = residual.copy()
        for ix, value in zip(groups, blup):
            conditional[ix] -= value
        return {
            "objective": float(objective),
            "beta": beta,
            "variance": [rho * sigma2, sigma2],
            "prediction": x @ beta,
            "residual": conditional,
            "blup": blup,
            "covariance": np.linalg.inv(information) * sigma2,
        }

    optimum = minimize_scalar(
        lambda z: evaluate(np.exp(z)), bounds=(-25, 20), method="bounded", options={"xatol": 1e-10}
    )
    if not optimum.success:
        raise RuntimeError("Independent likelihood optimization failed")
    rho = float(np.exp(optimum.x))
    return evaluate(0.0 if evaluate(0.0) <= optimum.fun else rho, True)


def snapshot(backend, fit, data, columns, reference_design=None):
    x = canonical_design(data, columns) if reference_design is None else reference_design
    x = np.asarray(x)
    if not bool(fit.converged):
        raise ValueError("Fit did not converge")
    if backend == "statsmodels":
        perm = column_permutation(x, fit.model.exog)
        beta = np.asarray(fit.fe_params)[perm]
        cov = np.asarray(fit.cov_params())[: len(perm), : len(perm)][np.ix_(perm, perm)]
        variance = [float(np.asarray(fit.cov_re)[0, 0]), float(fit.scale)]
        blup = [float(np.asarray(fit.random_effects[key])[0]) for key in sorted(data.id.unique())]
        residual = np.asarray(fit.resid)
        objective = -2 * float(fit.llf)
    else:
        perm = column_permutation(x, np.asarray(fit.design_matrix(data)))
        beta = np.asarray(fit.coefficients)[perm]
        cov = np.asarray(fit.v_beta_unscaled)[np.ix_(perm, perm)] * fit.sigma2
        variance = [float(fit.var_corr[0][3]), float(fit.sigma2)]
        effects = {row[1]: row[3] for row in fit.ranef}
        blup = [effects[key] for key in sorted(data.id.unique())]
        residual = np.asarray(fit.residuals)
        objective = float(fit.deviance)
    result = {
        "objective": objective,
        "beta": beta,
        "covariance": cov,
        "variance": variance,
        "prediction": x @ beta,
        "residual": residual,
        "blup": blup,
    }
    if not all(np.isfinite(value).all() for value in map(np.asarray, result.values())):
        raise ValueError("Nonfinite model output")
    return result


def holm(probabilities):
    """Step-down family-wise correction, independently checked against public APIs."""
    p = np.asarray(probabilities)
    order = np.argsort(p)
    corrected = np.empty_like(p)
    corrected[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected


def inference(result, data, columns):
    """Preserve asymptotic Wald inference; changing df methods is a separate decision."""
    design = canonical_design(data, columns)
    grid = grid_for(data, columns)
    gx = np.asarray(patsy.build_design_matrices([design.design_info], grid)[0])
    beta, covariance = result["beta"], result["covariance"]
    means = gx @ beta
    mean_se = np.sqrt(np.einsum("ij,jk,ik->i", gx, covariance, gx))
    pair_est, pair_se, pair_p, adjusted = [], [], [], []
    for b in data.b.cat.categories:
        rows = np.flatnonzero(grid.b == b)
        family = []
        for j in range(1, len(rows)):
            for i in range(j):
                vector = gx[rows[j]] - gx[rows[i]]
                estimate = float(vector @ beta)
                se = float(np.sqrt(vector @ covariance @ vector))
                p = float(2 * stats.norm.sf(abs(estimate / se)))
                pair_est.append(estimate)
                pair_se.append(se)
                pair_p.append(p)
                family.append(p)
        adjusted.extend(holm(family))
    term_p = []
    for name, section in design.design_info.term_name_slices.items():
        if name == "Intercept":
            continue
        coefficients = beta[section]
        statistic = coefficients @ np.linalg.solve(covariance[section, section], coefficients)
        term_p.append(float(stats.chi2.sf(statistic, len(coefficients))))
    se = np.concatenate([np.sqrt(np.diag(covariance)), mean_se, pair_se])
    estimates = np.concatenate([beta, means, pair_est])
    half = stats.norm.ppf(0.975) * se
    return {
        "se": se,
        "ci": np.column_stack((estimates - half, estimates + half)),
        "p": np.concatenate([term_p, pair_p, adjusted]),
        "means": means,
        "pair_est": pair_est,
        "pair_se": pair_se,
        "adjusted": adjusted,
    }


def difference(a, b, fields):
    checks = {}
    for field in fields:
        av, bv = np.asarray(a[field]), np.asarray(b[field])
        atol, rtol = TOLERANCES[field]
        finite = av.shape == bv.shape and np.isfinite(av).all() and np.isfinite(bv).all()
        checks[field] = {
            "pass": bool(finite and np.allclose(av, bv, atol=atol, rtol=rtol)),
            "max_absolute": float(np.max(np.abs(av - bv))) if finite else None,
        }
        if field == "p":
            checks[field]["alpha_005_decisions_equal"] = bool(
                finite and np.array_equal(av < 0.05, bv < 0.05)
            )
            checks[field]["pass"] &= checks[field]["alpha_005_decisions_equal"]
    return checks


def native_inference(backend, fit, data, columns):
    """Adjusted means and within-visit comparisons using the public APIs."""
    if backend == "lme":
        means = fit.emmeans_grid(["a"], data, by=["b"], at=dict.fromkeys(columns, 0.0))
        pairs = fit.emmeans_grid_pairs(
            ["a"], data, by=["b"], at=dict.fromkeys(columns, 0.0), adjust="holm"
        )
        native = [means.estimate, pairs.estimate, pairs.std_error, pairs.p_adjust]
    else:
        design = statsmodels_design(fit, grid_for(data, columns))
        rows = []
        size = len(data.a.cat.categories)
        for start in range(0, len(design), size):
            for j in range(1, size):
                for i in range(j):
                    rows.append(design[start + j] - design[start + i])
        test = fit.t_test(np.asarray(rows))
        p = np.asarray(test.pvalue).reshape(-1, size * (size - 1) // 2)
        adjusted = np.concatenate([pg.multicomp(row, method="holm")[1] for row in p])
        native = [
            fit.predict(grid_for(data, columns)),
            np.ravel(test.effect),
            np.ravel(test.sd),
            adjusted,
        ]
    return native


def statsmodels_design(fit, data):
    """Preserve the fitted Patsy encoding across statsmodels 0.14 and 0.15."""
    metadata = fit.model.data
    info = getattr(metadata, "design_info", None)
    if info is None:
        info = getattr(metadata, "model_spec", None)
    if not isinstance(info, patsy.DesignInfo):
        raise ValueError("The comparison requires the fitted Patsy design specification")
    return np.asarray(patsy.build_design_matrices([info], data)[0])


def compare_native_inference(backend, fit, data, columns, values):
    """Check the benchmark adapter against each engine's actual public inference API."""
    for actual, key in zip(
        native_inference(backend, fit, data, columns), ["means", "pair_est", "pair_se", "adjusted"]
    ):
        np.testing.assert_allclose(actual, values[key], atol=1e-9, rtol=1e-8)


def checked_call(call):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        start = time.perf_counter()
        try:
            result = call()
            elapsed = time.perf_counter() - start
            error = None
        except Exception as exc:
            result, elapsed, error = (
                None,
                time.perf_counter() - start,
                f"{type(exc).__name__}: {exc}",
            )
    return result, elapsed, sorted({str(w.message) for w in captured}), error


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def measured(call, validate, warmups, repeats):
    records = []
    for i in range(warmups + repeats):
        result, elapsed, notes, error = checked_call(call)
        evidence = None
        if error is None:
            try:
                evidence = validate(result)
            except Exception as exc:
                error = f"Validation {type(exc).__name__}: {exc}"
        records.append(
            {
                "warmup": i < warmups,
                "seconds": elapsed,
                "warnings": notes,
                "error": error,
                "checks": evidence,
            }
        )
    return records


def work_case(backend, name, warmups, repeats, replicates):
    data, columns = dataset(name)
    design = canonical_design(data, columns)
    data_hash = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    result = {
        "case": name,
        "rows": len(data),
        "subjects": data.id.nunique(),
        "input_sha256": data_hash,
        "columns": columns,
        "fits": {},
        "timings": {},
    }
    references = {}
    for reml in (False, True):
        label = "reml" if reml else "ml"
        oracle = gaussian_reference(data, design, reml)
        references[label] = oracle
        result["fits"][label] = {"oracle": serializable(oracle)}

        def validate(fit):
            values = snapshot(backend, fit, data, columns, design)
            infer = inference(values, data, columns)
            compare_native_inference(backend, fit, data, columns, infer)
            repeat_checks = {}
            if "inference" in result["fits"][label]:
                repeat_checks = {
                    "repeat_" + k: v
                    for k, v in difference(
                        infer, result["fits"][label]["inference"], ["se", "ci", "p"]
                    ).items()
                }
            result["fits"][label]["fit"] = serializable(values)
            result["fits"][label]["inference"] = serializable(infer)
            return (
                difference(
                    values,
                    oracle,
                    ["objective", "beta", "variance", "prediction", "residual", "blup"],
                )
                | repeat_checks
            )

        result["timings"]["cold_" + label] = measured(
            lambda: fit_model(backend, data, columns, reml), validate, warmups, repeats
        )

    fitted, _, _, fit_error = checked_call(lambda: fit_model(backend, data, columns, True))
    if fit_error is None and "inference" in result["fits"]["reml"]:
        expected_inference = result["fits"]["reml"]["inference"]

        def validate_inference(values):
            for value, key in zip(values, ["means", "pair_est", "pair_se", "adjusted"]):
                np.testing.assert_allclose(value, expected_inference[key], atol=1e-9, rtol=1e-8)
            return {"native_inference": {"pass": True}}

        result["timings"]["adjusted_comparisons"] = measured(
            lambda: native_inference(backend, fitted, data, columns),
            validate_inference,
            warmups,
            repeats,
        )

    # Identical null-generated responses: random-number differences cannot hide refit errors.
    null_design = canonical_design(data, columns, True)
    null_oracle = gaussian_reference(data, null_design, False)
    rng = np.random.default_rng(190927)
    codes = pd.factorize(data.id, sort=True)[0]
    responses = [
        np.asarray(null_oracle["prediction"])
        + rng.normal(0, np.sqrt(null_oracle["variance"][0]), data.id.nunique())[codes]
        + rng.normal(0, np.sqrt(null_oracle["variance"][1]), len(data))
        for _ in range(replicates)
    ]
    result["response_sha256"] = hashlib.sha256(np.asarray(responses).tobytes()).hexdigest()
    result["observed_lrt"] = null_oracle["objective"] - references["ml"]["objective"]
    # Large case measures fits; repeated-workflow cases keep all subject deletions tractable.
    if data.id.nunique() > 48:
        result["refit_skip"] = "Fit scaling case; repeated workflows restricted to <=48 subjects"
        return result

    prepared = []
    started = time.perf_counter()
    for null in (False, True):
        if backend == "lme":
            _, formula, factors = specification(data, columns, null)
            prepared.append(lme.prepare_lmer(formula, data, factors=factors))
        else:
            prepared.append(np.asarray(canonical_design(data, columns, null)))
    result["refit_setup_seconds"] = time.perf_counter() - started
    shared_oracles = []
    for response in responses:
        sample = data.assign(y=response)
        shared_oracles.append([gaussian_reference(sample, x, False) for x in (design, null_design)])

    def refits():
        pairs = []
        for response in responses:
            if backend == "lme":
                pairs.append(
                    [
                        model.fit(
                            response.tolist(),
                            reml=False,
                            control=lme.FitControl(tolerance=LME_TOLERANCE),
                        )
                        for model in prepared
                    ]
                )
            else:
                pairs.append(
                    [sm.MixedLM(response, x, groups=data.id).fit(reml=False) for x in prepared]
                )
        return pairs

    def validate_refits(pairs):
        statistics = []
        all_checks = []
        for response, pair, oracles in zip(responses, pairs, shared_oracles):
            values = []
            for fit, oracle, x in zip(pair, oracles, (design, null_design)):
                values.append(snapshot(backend, fit, data.assign(y=response), columns, x))
                all_checks.append(difference(values[-1], oracle, ["objective", "beta", "variance"]))
            statistics.append(values[1]["objective"] - values[0]["objective"])
        expected = [o[1]["objective"] - o[0]["objective"] for o in shared_oracles]
        checks = difference({"lrt": statistics}, {"lrt": expected}, ["lrt"])
        checks["all_refits"] = {"pass": all(c["pass"] for row in all_checks for c in row.values())}
        result["shared_refit_checks"] = all_checks
        result["shared_statistics"] = statistics
        result["shared_oracle_statistics"] = expected
        result["shared_p"] = (1 + sum(v >= result["observed_lrt"] for v in statistics)) / (
            1 + len(statistics)
        )
        return checks

    result["timings"]["shared_null_refits"] = measured(refits, validate_refits, warmups, repeats)
    result["timings"]["formula_null_refits"] = measured(
        lambda: [
            [
                fit_model(backend, data.assign(y=response), columns, False, null)
                for null in (False, True)
            ]
            for response in responses
        ],
        validate_refits,
        warmups,
        repeats,
    )
    subsets = [data.loc[data.id != subject].copy() for subject in sorted(data.id.unique())]
    loo_oracles = [
        gaussian_reference(subset, canonical_design(subset, columns), True) for subset in subsets
    ]

    def validate_loo(fits):
        checks = []
        effects = []
        for fit, subset, oracle in zip(fits, subsets, loo_oracles):
            value = snapshot(backend, fit, subset, columns)
            checks.append(difference(value, oracle, ["objective", "beta", "variance"]))
            effects.append(inference(value, subset, columns)["p"])
        result["loo_checks"] = checks
        result["loo_p"] = serializable(effects)
        return {"all_deletions": {"pass": all(c["pass"] for row in checks for c in row.values())}}

    result["timings"]["leave_one_subject_out"] = measured(
        lambda: [fit_model(backend, subset, columns, True) for subset in subsets],
        validate_loo,
        warmups,
        repeats,
    )
    if backend == "lme":
        result["native_bootstrap"] = {}
        for jobs in (1, 2):
            value, elapsed, notes, error = checked_call(
                lambda: lme.bootstrap_lrt(
                    prepared[0],
                    prepared[1],
                    replicates,
                    seed=190927,
                    n_jobs=jobs,
                    control=lme.FitControl(tolerance=LME_TOLERANCE),
                )
            )
            result["native_bootstrap"][str(jobs)] = {
                "seconds": elapsed,
                "warnings": notes,
                "error": error,
            }
            if value is not None:
                result["native_bootstrap"][str(jobs)].update(
                    {
                        key: getattr(value, key)
                        for key in (
                            "observed",
                            "requested",
                            "valid",
                            "statistics",
                            "errors",
                            "p_value",
                            "mc_se",
                        )
                    }
                )
    return result


def metadata():
    import lme_python.lme_python as extension

    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "lme_python",
                "numpy",
                "pandas",
                "scipy",
                "statsmodels",
                "patsy",
                "pingouin",
                "polars",
                "pyarrow",
            )
        },
        "threads": {key: os.environ.get(key) for key in THREAD_VARS},
        "lme_tolerance": LME_TOLERANCE,
        "threadpools": threadpool_info(),
    }


def summarize(runs, names):
    if not runs:
        raise ValueError("No worker reports")
    for run in runs:
        for key in (
            "revision",
            "harness_sha256",
            "extension_sha256",
            "versions",
            "threads",
            "lme_tolerance",
        ):
            if run["metadata"][key] != runs[0]["metadata"][key]:
                raise ValueError("Worker provenance differs: " + key)
    rows = []
    for name in names:
        selected = [(run, next(c for c in run["cases"] if c["case"] == name)) for run in runs]
        errors = []
        hashes = {case["input_sha256"] for _, case in selected}
        if len(hashes) != 1:
            raise ValueError("Inputs changed across workers")
        if len({case["response_sha256"] for _, case in selected}) != 1:
            raise ValueError("Simulated responses changed across workers")
        base = {
            backend: next(c for run, c in selected if run["backend"] == backend)
            for backend in ("statsmodels", "lme")
        }
        agreement = {}
        for method in ("ml", "reml"):
            try:
                left, right = [base[b]["fits"][method] for b in ("statsmodels", "lme")]
                agreement[method] = difference(
                    left["fit"],
                    right["fit"],
                    ["objective", "beta", "variance", "prediction", "residual", "blup"],
                )
                agreement[method].update(
                    difference(left["inference"], right["inference"], ["se", "ci", "p"])
                )
            except KeyError:
                errors.append("Missing validated " + method + " fit")
        for run, case in selected:
            for metric, records in case["timings"].items():
                for record in records:
                    if (
                        record["error"]
                        or not record["checks"]
                        or not all(c["pass"] for c in record["checks"].values())
                    ):
                        errors.append(run["backend"] + ":" + metric + ":fit validation failed")
            # Check every independent process, including inference, against its reference worker.
            for method in ("ml", "reml"):
                if (
                    "inference" in case["fits"][method]
                    and "inference" in base[run["backend"]]["fits"][method]
                ):
                    repeated = difference(
                        case["fits"][method]["inference"],
                        base[run["backend"]]["fits"][method]["inference"],
                        ["se", "ci", "p"],
                    )
                    if not all(c["pass"] for c in repeated.values()):
                        errors.append("Inference changed between independent processes")
            if "native_bootstrap" in case:
                a, b = [case["native_bootstrap"][str(j)] for j in (1, 2)]
                if (
                    a.get("error")
                    or b.get("error")
                    or a.get("valid", 0) != a.get("requested")
                    or b.get("valid", 0) != b.get("requested")
                    or a.get("statistics") != b.get("statistics")
                    or any(a.get("errors", ["missing"]))
                    or any(b.get("errors", ["missing"]))
                ):
                    errors.append("Native bootstrap failed or differs across worker counts")
                elif not np.isclose(a["observed"], case["observed_lrt"], atol=2e-4, rtol=1e-4) or a[
                    "p_value"
                ] != (1 + sum(v >= a["observed"] for v in a["statistics"])) / (1 + a["valid"]):
                    errors.append("Native bootstrap observed statistic or p-value contract differs")
        refit_agreement = None
        if "shared_statistics" in base["statsmodels"] and "shared_statistics" in base["lme"]:
            refit_agreement = difference(
                {"lrt": base["statsmodels"]["shared_statistics"]},
                {"lrt": base["lme"]["shared_statistics"]},
                ["lrt"],
            )
            if (
                not refit_agreement["lrt"]["pass"]
                or base["statsmodels"]["shared_p"] != base["lme"]["shared_p"]
            ):
                errors.append("Shared-response bootstrap differs")
        loo_agreement = None
        if "loo_p" in base["statsmodels"] and "loo_p" in base["lme"]:
            loo_agreement = difference(
                {"p": base["statsmodels"]["loo_p"]}, {"p": base["lme"]["loo_p"]}, ["p"]
            )
            if not loo_agreement["p"]["pass"]:
                errors.append("Leave-one-subject-out inference differs")
        comparable = not errors and all(
            c["pass"] for method in agreement.values() for c in method.values()
        )
        timings = {}
        for metric in sorted({key for _, c in selected for key in c["timings"]}):
            if any(metric not in c["timings"] for _, c in selected):
                timings[metric] = {"qualified": False, "error": "Missing metric from a worker"}
                comparable = False
                continue
            blocks = sorted({run["block"] for run, _ in selected})
            ratios, medians = [], {"statsmodels": [], "lme": []}
            for block in blocks:
                for backend in medians:
                    process_medians = []
                    for run, case in selected:
                        if run["block"] == block and run["backend"] == backend:
                            samples = [
                                r["seconds"] for r in case["timings"][metric] if not r["warmup"]
                            ]
                            process_medians.append(float(np.median(samples)))
                    medians[backend].append(float(np.median(process_medians)))
                ratios.append(medians["statsmodels"][-1] / medians["lme"][-1])
            rng = np.random.default_rng(260915)
            interval = np.quantile(
                [np.median(rng.choice(ratios, len(ratios), replace=True)) for _ in range(2000)],
                [0.025, 0.975],
            ).tolist()
            timings[metric] = {
                "median_seconds": {b: float(np.median(v)) for b, v in medians.items()},
                "statsmodels_over_lme": float(np.median(ratios)),
                "block_ratios": ratios,
                "block_bootstrap_95_interval": interval,
                "qualified": comparable,
            }
        if not comparable:
            for timing in timings.values():
                timing["qualified"] = False
        rows.append(
            {
                "case": name,
                "agreement": agreement,
                "errors": sorted(set(errors)),
                "compatible": comparable,
                "refit_agreement": refit_agreement,
                "loo_agreement": loo_agreement,
                "timings": timings,
            }
        )
    return rows


def main():
    global LME_TOLERANCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark-results/model-workflows.json"
    )
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=9)
    parser.add_argument("--lme-tolerance", type=float, default=1e-6)
    parser.add_argument("--worker", choices=["statsmodels", "lme"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    LME_TOLERANCE = args.lme_tolerance
    names = args.cases.split(",")
    if not names or any(name not in CASES for name in names) or len(set(names)) != len(names):
        parser.error("Choose unique known cases: " + ",".join(CASES))
    if min(args.blocks, args.repeats, args.replicates) < 1 or args.warmups < 0:
        parser.error("Counts must be positive (warmups may be zero)")
    if not 0 < args.lme_tolerance < float("inf"):
        parser.error("The fit tolerance must be finite and positive")
    for key in THREAD_VARS:
        os.environ[key] = "1"
    dependencies()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        report = {"backend": args.worker, "metadata": metadata(), "cases": []}
        for name in names:
            print(args.worker, name, flush=True)
            report["cases"].append(
                work_case(args.worker, name, args.warmups, args.repeats, args.replicates)
            )
            args.output.write_text(
                json.dumps(serializable(report), indent=2, allow_nan=False), encoding="utf-8"
            )
        return 0
    runs = []
    for block in range(args.blocks):
        order = ["statsmodels", "lme"] if block % 2 == 0 else ["lme", "statsmodels"]
        for backend in order:
            path = args.output.with_name(f"{args.output.stem}-block{block}-{backend}.json")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                "--output",
                str(path),
                "--cases",
                args.cases,
                "--warmups",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--replicates",
                str(args.replicates),
                "--lme-tolerance",
                str(args.lme_tolerance),
            ]
            subprocess.run(command, check=True, cwd=ROOT)
            run = json.loads(path.read_text(encoding="utf-8"))
            run["block"] = block
            runs.append(run)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tolerances": TOLERANCES,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summarize(runs, names),
        "runs": runs,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for row in report["summary"]:
        print(
            row["case"], "COMPATIBLE" if row["compatible"] else "NOT DROP-IN COMPATIBLE", flush=True
        )
    return 0 if all(row["compatible"] for row in report["summary"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
