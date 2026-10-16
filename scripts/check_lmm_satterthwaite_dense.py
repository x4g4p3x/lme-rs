"""Independent dense observed-REML derivatives at saved parameter values.

No model is fitted. NumPy is optional until this diagnostic is executed.
Physical coordinates are eta=(sigma2, Sigma00, Sigma01, Sigma11). Derivatives
extend locally over positive-definite V, including around covariance boundaries.
The transformed coordinates match the library's rho=(theta, sigma2/base_sigma2).
These are numerical diagnostics, not calibration evidence or valid inference for
a source fit that did not converge.

With P=V^-1-V^-1 X Phi X' V^-1, r=Py and Kj=dV/deta_j, the deviance
score is tr(PKj)-r'Kjr and its Hessian is
-tr(PKjPKk)+2r'KjPKkr. The slope-variance gradient is t'Kjt, where
t=V^-1 X Phi (0,1)'. Under eta(rho), the Hessian includes both J'HJ and
sum_j score_j * d2eta_j/drho2; that second term matters away from stationarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def require_numpy():
    if np is None:
        raise RuntimeError(
            "NumPy is optional; install it to execute the dense derivative diagnostic"
        )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class DenseReml:
    """Direct dense GLS and analytic derivatives, independent of LMM fit APIs."""

    def __init__(self, predictor, response, groups):
        require_numpy()
        self.x = np.asarray(predictor, dtype=float)
        self.y = np.asarray(response, dtype=float)
        group = np.asarray(groups)
        if (
            self.x.ndim != 1
            or self.y.shape != self.x.shape
            or group.shape != self.x.shape
            or len(self.x) <= 2
            or not np.isfinite(self.x).all()
            or not np.isfinite(self.y).all()
        ):
            raise ValueError("finite matching one-dimensional observations required")
        self.design = np.column_stack((np.ones(len(self.x)), self.x))
        if np.linalg.matrix_rank(self.design) != 2:
            raise ValueError("fixed design must have rank two")
        same = (group[:, None] == group[None, :]).astype(float)
        self.components = [
            np.eye(len(self.x)),
            same,
            same * (self.x[:, None] + self.x[None, :]),
            same * self.x[:, None] * self.x[None, :],
        ]

    def state(self, eta):
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (4,) or not np.isfinite(eta).all() or eta[0] <= 0:
            raise ValueError("four finite physical parameters with positive sigma2 required")
        variance = sum(v * k for v, k in zip(eta, self.components))
        # This validates positive definiteness without adding jitter.
        chol = np.linalg.cholesky(variance)
        inverse = np.linalg.solve(variance, np.eye(len(self.x)))
        wx = inverse @ self.design
        precision = self.design.T @ wx
        covariance = np.linalg.solve(precision, np.eye(2))
        beta = covariance @ wx.T @ self.y
        projection = inverse - wx @ covariance @ wx.T
        projection = (projection + projection.T) / 2
        residual_precision = projection @ self.y
        criterion = (
            (len(self.x) - 2) * math.log(2 * math.pi)
            + 2 * np.log(np.diag(chol)).sum()
            + np.linalg.slogdet(precision)[1]
            + self.y @ residual_precision
        )
        return {
            "criterion": float(criterion),
            "beta": beta,
            "covariance": covariance,
            "projection": projection,
            "residual_precision": residual_precision,
            "slope_weight": wx @ covariance[:, 1],
        }

    def evaluate(self, eta):
        state = self.state(eta)
        projection = state["projection"]
        residual = state["residual_precision"]
        pk = [projection @ k for k in self.components]
        kr = [k @ residual for k in self.components]
        score = np.array([np.trace(p) - residual @ r for p, r in zip(pk, kr)])
        hessian = np.array(
            [
                [-np.trace(pk[j] @ pk[k]) + 2 * kr[j] @ projection @ kr[k] for k in range(4)]
                for j in range(4)
            ]
        )
        slope_weight = state["slope_weight"]
        gradient = np.array([slope_weight @ k @ slope_weight for k in self.components])
        return {
            **state,
            "score": score,
            "hessian": hessian,
            "variance_gradient": gradient,
        }


def relative_map(parameters, base_sigma2):
    """Physical eta, Jacobian and second derivatives for (a,b,c,relative_sigma2)."""
    require_numpy()
    parameters = np.asarray(parameters, dtype=float)
    if (
        parameters.shape != (4,)
        or not np.isfinite(parameters).all()
        or parameters[3] <= 0
        or not math.isfinite(base_sigma2)
        or base_sigma2 <= 0
    ):
        raise ValueError("finite relative coordinates and positive variance scales required")
    a, b, c, relative = parameters
    s = base_sigma2 * relative
    eta = np.array([s, s * a * a, s * a * b, s * (b * b + c * c)])
    jacobian = np.array(
        [
            [0, 0, 0, base_sigma2],
            [2 * s * a, 0, 0, base_sigma2 * a * a],
            [s * b, s * a, 0, base_sigma2 * a * b],
            [0, 2 * s * b, 2 * s * c, base_sigma2 * (b * b + c * c)],
        ]
    )
    second = np.zeros((4, 4, 4))
    second[1, 0, 0] = 2 * s
    second[1, 0, 3] = second[1, 3, 0] = 2 * base_sigma2 * a
    second[2, 0, 1] = second[2, 1, 0] = s
    second[2, 0, 3] = second[2, 3, 0] = base_sigma2 * b
    second[2, 1, 3] = second[2, 3, 1] = base_sigma2 * a
    second[3, 1, 1] = second[3, 2, 2] = 2 * s
    second[3, 1, 3] = second[3, 3, 1] = 2 * base_sigma2 * b
    second[3, 2, 3] = second[3, 3, 2] = 2 * base_sigma2 * c
    return eta, jacobian, second


def variance_df(variance, gradient, hessian):
    """Observed-information Satterthwaite diagnostic; never substitutes a fallback."""
    try:
        denominator = float(gradient @ (2 * np.linalg.solve(hessian, gradient)))
    except np.linalg.LinAlgError:
        return None
    return 2 * variance * variance / denominator if denominator > 0 else None


def relative_derivatives(model, parameters, base_sigma2):
    eta, jacobian, second = relative_map(parameters, base_sigma2)
    physical = model.evaluate(eta)
    curvature = np.einsum("j,jab->ab", physical["score"], second)
    hessian = jacobian.T @ physical["hessian"] @ jacobian + curvature
    gradient = jacobian.T @ physical["variance_gradient"]
    variance = float(physical["covariance"][1, 1])
    return {
        "eta": eta,
        "criterion": physical["criterion"],
        "beta": physical["beta"],
        "slope_variance": variance,
        "physical_score": physical["score"],
        "physical_hessian": physical["hessian"],
        "physical_variance_gradient": physical["variance_gradient"],
        "relative_score": jacobian.T @ physical["score"],
        "relative_hessian": hessian,
        "relative_variance_gradient": gradient,
        "transformation_curvature": curvature,
        "relative_hessian_eigenvalues": np.linalg.eigvalsh(hessian),
        "relative_hessian_condition": float(np.linalg.cond(hessian)),
        "relative_df": variance_df(variance, gradient, hessian),
        "physical_df": variance_df(variance, physical["variance_gradient"], physical["hessian"]),
    }


def read_evidence(directory):
    controls = json.loads((directory / "controls.json").read_text(encoding="utf-8"))
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    candidates = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    if digest(directory / "inputs.json") != controls["inputs_sha256"]:
        raise ValueError("saved input hash differs from controls receipt")
    expected_inputs = {tuple(value) for value in controls["selected"]}
    observed_inputs = [(value["case"]["id"], value["replicate"]) for value in inputs]
    if len(observed_inputs) != len(set(observed_inputs)) or set(observed_inputs) != expected_inputs:
        raise ValueError("missing, duplicate or unknown saved input")
    expected = {
        (case, replicate, canonical_digest(control))
        for case, replicate in expected_inputs
        for control in controls["controls"]
    }
    observed = [
        (value["case"], value["replicate"], canonical_digest(value["control"]))
        for value in candidates
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("missing, duplicate or unknown saved candidate")
    return controls, {(v["case"]["id"], v["replicate"]): v for v in inputs}, candidates


def json_value(value):
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def check(directory, output):
    require_numpy()
    controls, inputs, candidates = read_evidence(directory)
    results = []
    for row in candidates:
        raw = inputs[row["case"], row["replicate"]]
        fit = row["candidate"]
        result = {
            "case": row["case"],
            "replicate": row["replicate"],
            "control": row["control"],
            "source_converged": fit.get("converged") is True,
            "scope": "local derivative diagnostic; nonconverged fits remain unevaluable",
        }
        if "theta" not in fit or "sigma2" not in fit:
            results.append({**result, "unavailable": fit.get("error", "missing parameters")})
            continue
        model = DenseReml(raw["case"]["x"], raw["y"], raw["case"]["group"])
        dense = relative_derivatives(model, [*fit["theta"], 1.0], fit["sigma2"])
        reported = fit.get("methods", {}).get("satterthwaite", {})
        result.update(
            dense=dense,
            criterion_difference=fit.get("criterion", dense["criterion"]) - dense["criterion"],
            reported_criterion=fit.get("criterion"),
            reported_satterthwaite=reported,
            retained_reference_satterthwaite=raw.get("retained_reference", {})
            .get("methods", {})
            .get("satterthwaite"),
            df_difference=reported["df"] - dense["relative_df"]
            if reported.get("df") is not None and dense["relative_df"] is not None
            else None,
        )
        results.append(json_value(result))
    result = {
        "schema_version": 1,
        "scope": "independent dense derivatives at saved points; no fitting or calibration claim",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "executable": sys.executable,
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "source_sha256": digest(__file__),
        "evidence_sha256": {
            name: digest(directory / name)
            for name in ("inputs.json", "controls.json", "candidate.json")
        },
        "source_environment": controls.get("environment"),
        "results_sha256": canonical_digest(results),
        "results_hash_definition": (
            "SHA256 of sorted-key compact UTF-8 JSON of results, allow_nan=false"
        ),
        "results": results,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="saved controls-selected evidence directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.directory / "dense-derivatives.json"
    report = check(args.directory, output)
    print(f"dense derivatives: {len(report['results'])} saved points; {output}")


if __name__ == "__main__":
    main()
