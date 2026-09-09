//! Python model fitting and resampling entry points.
use super::*;
#[pyfunction]
#[pyo3(signature = (formula, data, reml=true, *, control=None))]
pub fn lmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    reml: bool,
    control: Option<&PyFitControl>,
) -> PyResult<PyLmeFit> {
    let control = control_value(control);
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    match py.detach(|| lme_rs::prepare_lmer(formula, &df).and_then(|p| p.fit(None, reml, &control)))
    {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(controls::model_error(e)),
    }
}

/// Prepare an LMM for repeated fits on the same formula and data.
#[pyfunction]
#[pyo3(signature = (formula, data, *, weights=None))]
pub fn prepare_lmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    weights: Option<Vec<f64>>,
) -> PyResult<PyLmerPrepared> {
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    match py.detach(|| {
        lme_rs::prepare_lmer_weighted(formula, &df, weights.map(ndarray::Array1::from_vec))
    }) {
        Ok(prepared) => Ok(PyLmerPrepared { inner: prepared }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "prepare_lmer failed: {e}"
        ))),
    }
}

/// Fit a prepared LMM (amortized hot path after [`prepare_lmer`]).
#[pyfunction]
#[pyo3(signature = (prepared, reml=true, *, y=None, control=None))]
pub fn fit_prepared(
    py: Python<'_>,
    prepared: &PyLmerPrepared,
    reml: bool,
    y: Option<Vec<f64>>,
    control: Option<&PyFitControl>,
) -> PyResult<PyLmeFit> {
    prepared.fit(py, y, reml, control)
}

/// Prepare a GLMM for repeated fits on the same formula and data.
#[pyfunction]
#[pyo3(signature = (formula, data, family_name, n_agq=1, weights=None, link_name=None))]
pub fn prepare_glmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    family_name: &str,
    n_agq: usize,
    weights: Option<Vec<f64>>,
    link_name: Option<&str>,
) -> PyResult<PyGlmerPrepared> {
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    let family = parse_family(family_name)?;
    let link = parse_link(link_name, family)?;
    let w = weights.map(ndarray::Array1::from_vec);
    match py
        .detach(|| lme_rs::prepare_glmer_weighted_with_link(formula, &df, family, link, n_agq, w))
    {
        Ok(prepared) => Ok(PyGlmerPrepared { inner: prepared }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "prepare_glmer failed: {e}"
        ))),
    }
}

/// Fit a prepared GLMM (amortized hot path after [`prepare_glmer`]).
#[pyfunction]
#[pyo3(signature = (prepared, *, y=None, control=None))]
pub fn fit_prepared_glmer(
    py: Python<'_>,
    prepared: &PyGlmerPrepared,
    y: Option<Vec<f64>>,
    control: Option<&PyFitControl>,
) -> PyResult<PyLmeFit> {
    prepared.fit(py, y, control)
}

/// Refit an LMM on the same formula and data (prepare + fit).
#[pyfunction]
#[pyo3(signature = (formula, data, reml=true))]
pub fn refit_lmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    reml: bool,
) -> PyResult<PyLmeFit> {
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    match py.detach(|| lme_rs::refit_lmer(formula, &df, reml)) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "refit_lmer failed: {e}"
        ))),
    }
}

/// Group-structure-preserving k-fold cross-validation for LMMs.
#[pyfunction]
#[pyo3(signature = (formula, data, group, n_splits=5, reml=true, seed=None, n_jobs=None))]
pub fn cv_grouped<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    group: &str,
    n_splits: usize,
    reml: bool,
    seed: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<PyCvGroupedResult> {
    let df = input::read_dataframe(py, data, Some(formula), &[group])?;
    match py.detach(|| lme_rs::cv_grouped(formula, &df, group, n_splits, reml, seed, n_jobs)) {
        Ok(res) => Ok(cv_result_to_py(res)),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cv_grouped failed: {e}"
        ))),
    }
}

fn cv_result_to_py(res: lme_rs::CvGroupedResult) -> PyCvGroupedResult {
    PyCvGroupedResult {
        oof_predictions: res.oof_predictions.to_vec(),
        test_fold: res.test_fold.to_vec(),
        rmse: res.rmse,
        mae: res.mae,
        mean_log_loss: res.mean_log_loss,
        folds: res
            .folds
            .into_iter()
            .map(|f| PyCvFoldMetric {
                fold: f.fold,
                n_train_groups: f.n_train_groups,
                n_test_groups: f.n_test_groups,
                n_train_obs: f.n_train_obs,
                n_test_obs: f.n_test_obs,
                rmse: f.rmse,
                mae: f.mae,
                mean_log_loss: f.mean_log_loss,
                converged: f.converged,
            })
            .collect(),
        all_converged: res.all_converged,
        n_splits: res.n_splits,
        group_col: res.group_col,
    }
}

/// Group-structure-preserving k-fold CV for GLMMs (response-scale OOF metrics).
#[pyfunction]
#[pyo3(signature = (
    formula,
    data,
    group,
    family_name,
    n_splits=5,
    n_agq=1,
    weights=None,
    link_name=None,
    seed=None,
    n_jobs=None
))]
pub fn cv_grouped_glmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    group: &str,
    family_name: &str,
    n_splits: usize,
    n_agq: usize,
    weights: Option<Vec<f64>>,
    link_name: Option<&str>,
    seed: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<PyCvGroupedResult> {
    let df = input::read_dataframe(py, data, Some(formula), &[group])?;
    let family = parse_family(family_name)?;
    let link = parse_link(link_name, family)?;
    let w = weights.map(ndarray::Array1::from_vec);
    match py.detach(|| {
        lme_rs::cv_grouped_glmer(
            formula, &df, group, n_splits, family, link, n_agq, w, seed, n_jobs,
        )
    }) {
        Ok(res) => Ok(cv_result_to_py(res)),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cv_grouped_glmer failed: {e}"
        ))),
    }
}

/// Parametric or residual bootstrap refits for LMMs (`bootMer`-style).
#[pyfunction]
#[pyo3(signature = (formula, data, fit, nsim=200, method="parametric", reml=true, seed=None, n_jobs=None))]
pub fn boot_lmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    fit: &PyLmeFit,
    nsim: usize,
    method: &str,
    reml: bool,
    seed: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<PyBootLmerResult> {
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    let boot_method = parse_boot_method(method)?;
    match py.detach(|| {
        lme_rs::boot_lmer(
            formula,
            &df,
            &fit.inner,
            nsim,
            boot_method,
            reml,
            seed,
            n_jobs,
        )
    }) {
        Ok(res) => Ok(PyBootLmerResult { inner: res }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "boot_lmer failed: {e}"
        ))),
    }
}

/// Parametric bootstrap refits for GLMMs (`bootMer`-style).
#[pyfunction]
#[pyo3(signature = (formula, data, fit, nsim=200, method="parametric", seed=None, n_jobs=None))]
pub fn boot_glmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    fit: &PyLmeFit,
    nsim: usize,
    method: &str,
    seed: Option<u64>,
    n_jobs: Option<usize>,
) -> PyResult<PyBootLmerResult> {
    let df = input::read_dataframe(py, data, None, &[])?;
    let boot_method = parse_boot_method(method)?;
    match py
        .detach(|| lme_rs::boot_glmer(formula, &df, &fit.inner, nsim, boot_method, seed, n_jobs))
    {
        Ok(res) => Ok(PyBootLmerResult { inner: res }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "boot_glmer failed: {e}"
        ))),
    }
}

fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    if obj.hasattr("tolist")? {
        return obj.call_method0("tolist")?.extract();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a 1-D numeric sequence",
    ))
}

fn extract_f64_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(m) = obj.extract::<Vec<Vec<f64>>>() {
        return Ok(m);
    }
    if obj.hasattr("tolist")? {
        let listed = obj.call_method0("tolist")?;
        if let Ok(m) = listed.extract::<Vec<Vec<f64>>>() {
            return Ok(m);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a 2-D numeric matrix (list of rows)",
    ))
}

/// Fit a fixed-effects-only linear model.
///
/// Two calling conventions:
///
/// * ``lm(formula, data)`` — Wilkinson formula and a tabular frame
///   (Polars / pandas / PyArrow).
/// * ``lm(y, x)`` — numeric response and dense design matrix (same as Rust
///   ``lm(y, x)``). ``x`` is ``n_obs × p`` as a list of rows, or any object
///   with ``.tolist()`` (NumPy arrays).
///
/// ``lm_matrix(y, x)`` remains an explicit alias for the numeric path.
#[pyfunction]
#[pyo3(signature = (formula_or_y, data))]
pub fn lm<'py>(
    py: Python<'py>,
    formula_or_y: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyLmeFit> {
    if let Ok(formula) = formula_or_y.extract::<&str>() {
        let df = input::read_dataframe(py, data, Some(formula), &[])?;
        return match py.detach(|| lme_rs::lm_df(formula, &df)) {
            Ok(fit) => Ok(PyLmeFit { inner: fit }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
        };
    }
    let y = extract_f64_vec(formula_or_y).map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "lm() expects lm(formula, data) or lm(y, x) with a numeric response vector",
        )
    })?;
    let x = extract_f64_matrix(data).map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "when the first argument is numeric, the second must be a 2-D design matrix",
        )
    })?;
    lm_matrix(py, y, x)
}

/// Fit OLS from numeric **y** and design matrix **X** (mirrors Rust `lm(y, x)`).
///
/// `x` is `n_obs × p` as a list of rows.
#[pyfunction]
pub fn lm_matrix(py: Python<'_>, y: Vec<f64>, x: Vec<Vec<f64>>) -> PyResult<PyLmeFit> {
    let n = y.len();
    if x.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least one row",
        ));
    }
    let p = x[0].len();
    if p == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least one column",
        ));
    }
    if x.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y has length {n} but x has {} rows",
            x.len()
        )));
    }
    let mut mat = Array2::<f64>::zeros((n, p));
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Row {i} has length {} but expected {p}",
                row.len()
            )));
        }
        for (j, &v) in row.iter().enumerate() {
            mat[[i, j]] = v;
        }
    }
    let y_arr = ndarray::Array1::from_vec(y);
    match py.detach(|| lme_rs::lm(&y_arr, &mat)) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(controls::model_error(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, family_name, n_agq=1, link_name=None, *, control=None))]
pub fn glmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    family_name: &str,
    n_agq: usize,
    link_name: Option<&str>,
    control: Option<&PyFitControl>,
) -> PyResult<PyLmeFit> {
    glmer_weighted(
        py,
        formula,
        data,
        family_name,
        n_agq,
        None,
        link_name,
        control,
    )
}

#[pyfunction]
#[pyo3(signature = (formula, data, family_name, n_agq=1, weights=None, link_name=None, *, control=None))]
pub fn glmer_weighted<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    family_name: &str,
    n_agq: usize,
    weights: Option<Vec<f64>>,
    link_name: Option<&str>,
    control: Option<&PyFitControl>,
) -> PyResult<PyLmeFit> {
    let control = control_value(control);
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    let family = parse_family(family_name)?;
    let link = parse_link(link_name, family)?;
    let weights_arr = weights.map(ndarray::Array1::from_vec);

    match py.detach(|| {
        lme_rs::prepare_glmer_weighted_with_link(formula, &df, family, link, n_agq, weights_arr)
            .and_then(|p| p.fit(None, &control))
    }) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(controls::model_error(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, reml=true, weights=None, *, control=None))]
pub fn lmer_weighted<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    reml: bool,
    weights: Option<Vec<f64>>,
    control: Option<&PyFitControl>,
) -> PyResult<PyLmeFit> {
    let control = control_value(control);
    let df = input::read_dataframe(py, data, Some(formula), &[])?;
    let weights_arr = weights.map(ndarray::Array1::from_vec);
    match py.detach(|| {
        lme_rs::prepare_lmer_weighted(formula, &df, weights_arr)
            .and_then(|p| p.fit(None, reml, &control))
    }) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(controls::model_error(e)),
    }
}

/// Build a **q × p** contrast matrix from named coefficient weights.
///
/// Each row is a list of `(coefficient_name, weight)` pairs, aligned with `fixed_names`.
#[pyfunction]
#[pyo3(name = "contrast_matrix_from_names")]
pub(super) fn contrast_matrix_from_names_py(
    fixed_names: Vec<String>,
    rows: Vec<Vec<(String, f64)>>,
) -> PyResult<Vec<Vec<f64>>> {
    let p = fixed_names.len();
    let mut index_rows = Vec::with_capacity(rows.len());
    for row in rows {
        let mut idx_row = Vec::with_capacity(row.len());
        for (name, w) in row {
            let j = fixed_names.iter().position(|n| n == &name).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown coefficient name '{name}' in contrast"
                ))
            })?;
            idx_row.push((j, w));
        }
        index_rows.push(idx_row);
    }
    let mat = contrast_matrix(p, &index_rows);
    Ok((0..mat.nrows()).map(|i| mat.row(i).to_vec()).collect())
}

/// Build a **q × p** contrast matrix from `(column_index, weight)` rows (Rust `contrast_matrix`).
#[pyfunction]
#[pyo3(name = "contrast_matrix")]
pub fn contrast_matrix_py(p: usize, rows: Vec<Vec<(usize, f64)>>) -> PyResult<Vec<Vec<f64>>> {
    let mat = contrast_matrix(p, &rows);
    Ok((0..mat.nrows()).map(|i| mat.row(i).to_vec()).collect())
}

/// Fit a nonlinear mixed-effects model (`SSlogis` mean; random effect on one NL parameter).
#[pyfunction]
#[pyo3(signature = (formula, data, start=None, reml=false, n_agq=1, lower=None, upper=None, group_lower=None, group_upper=None, *, max_inner=120, max_outer_iters=500))]
pub fn nlmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    start: Option<&Bound<'py, PyDict>>,
    reml: bool,
    n_agq: usize,
    lower: Option<&Bound<'py, PyDict>>,
    upper: Option<&Bound<'py, PyDict>>,
    group_lower: Option<&Bound<'py, PyDict>>,
    group_upper: Option<&Bound<'py, PyDict>>,
    max_inner: usize,
    max_outer_iters: u64,
) -> PyResult<PyLmeFit> {
    let df = input::read_dataframe(py, data, None, &[])?;
    let start_map = parse_nlmm_start(start)?;
    let opts = lme_rs::NlmerOptions {
        reml,
        start: start_map,
        n_agq,
        lower: parse_optional_nlmm_bounds(lower)?,
        upper: parse_optional_nlmm_bounds(upper)?,
        group_lower: parse_optional_nlmm_bounds(group_lower)?,
        group_upper: parse_optional_nlmm_bounds(group_upper)?,
        max_inner,
        max_outer_iters,
    };
    match py.detach(|| lme_rs::nlmer_with_options(formula, &df, &opts)) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(controls::model_error(e)),
    }
}

/// Fit a nonlinear mixed model with a user-defined mean function.
///
/// ``formula`` uses the custom-mean layout ``response ~ covariate ~ re | group``
/// (middle segment is the covariate column, not an ``SS*`` call). ``mean_fn(x, params)``
/// must return ``(mu, grad)`` where ``grad`` has one partial derivative per name in
/// ``param_names``.
#[pyfunction]
#[pyo3(signature = (formula, data, mean_fn, param_names, start=None, reml=false, n_agq=1, lower=None, upper=None, group_lower=None, group_upper=None, *, max_inner=120, max_outer_iters=500))]
pub fn nlmer_with_mean<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    mean_fn: &Bound<'py, PyAny>,
    param_names: Vec<String>,
    start: Option<&Bound<'py, PyDict>>,
    reml: bool,
    n_agq: usize,
    lower: Option<&Bound<'py, PyDict>>,
    upper: Option<&Bound<'py, PyDict>>,
    group_lower: Option<&Bound<'py, PyDict>>,
    group_upper: Option<&Bound<'py, PyDict>>,
    max_inner: usize,
    max_outer_iters: u64,
) -> PyResult<PyLmeFit> {
    if param_names.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "param_names must name at least one nonlinear parameter",
        ));
    }
    let mean = validate_py_nlmm_mean(mean_fn, param_names.len())?;
    let parsed = parse_nlmer_custom_formula(formula, &param_names).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid nlmer formula: {e}"))
    })?;
    let df = input::read_dataframe(py, data, None, &[])?;
    let start_map = parse_nlmm_start(start)?;
    let opts = lme_rs::NlmerOptions {
        reml,
        start: start_map,
        n_agq,
        lower: parse_optional_nlmm_bounds(lower)?,
        upper: parse_optional_nlmm_bounds(upper)?,
        group_lower: parse_optional_nlmm_bounds(group_lower)?,
        group_upper: parse_optional_nlmm_bounds(group_upper)?,
        max_inner,
        max_outer_iters,
    };
    let callback_error = mean.callback_error.clone();
    let fit_result =
        py.detach(|| lme_rs::nlmer_with_mean(&parsed, Arc::new(mean), &df, Some(formula), &opts));
    if let Ok(mut error) = callback_error.lock() {
        if let Some(error) = error.take() {
            return Err(error);
        }
    }
    match fit_result {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(controls::model_error(e)),
    }
}

/// Likelihood ratio test between two nested fitted models.
#[pyfunction]
pub fn anova(
    py: Python<'_>,
    fit_a: &PyLmeFit,
    fit_b: &PyLmeFit,
) -> PyResult<PyLikelihoodRatioAnova> {
    match py.detach(|| lme_rs::anova(&fit_a.inner, &fit_b.inner)) {
        Ok(res) => Ok(PyLikelihoodRatioAnova {
            n_params_0: res.n_params_0,
            n_params_1: res.n_params_1,
            deviance_0: res.deviance_0,
            deviance_1: res.deviance_1,
            chi_sq: res.chi_sq,
            df: res.df,
            p_value: res.p_value,
            formula_0: res.formula_0,
            formula_1: res.formula_1,
        }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "anova failed: {}",
            e
        ))),
    }
}
