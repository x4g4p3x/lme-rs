//! Python numerical controls and diagnostic output.
use super::*;

#[pyclass(name = "FitControl", from_py_object)]
#[derive(Clone)]
pub struct PyFitControl {
    pub(super) inner: lme_rs::FitControl,
}

#[pymethods]
impl PyFitControl {
    #[new]
    #[pyo3(signature=(*, max_iterations=1000, tolerance=1e-6, max_inner_iterations=None, start=None, require_convergence=false))]
    fn new(
        max_iterations: u64,
        tolerance: f64,
        max_inner_iterations: Option<usize>,
        start: Option<Vec<f64>>,
        require_convergence: bool,
    ) -> PyResult<Self> {
        let control = lme_rs::FitControl {
            max_iterations,
            tolerance,
            max_inner_iterations,
            start: start.map(ndarray::Array1::from_vec),
            require_convergence,
        };
        control
            .validate(control.start.as_ref().map_or(0, |s| s.len()))
            .map_err(model_error)?;
        Ok(Self { inner: control })
    }
}

pub(super) fn control_value(control: Option<&PyFitControl>) -> lme_rs::FitControl {
    control.map(|c| c.inner.clone()).unwrap_or_default()
}

pub(super) fn model_error(error: lme_rs::LmeError) -> PyErr {
    match error {
        lme_rs::LmeError::NonConvergence { .. } | lme_rs::LmeError::LinearAlgebra { .. } => {
            pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
        _ => pyo3::exceptions::PyValueError::new_err(error.to_string()),
    }
}

pub(super) fn diagnostic_dict(py: Python<'_>, fit: &LmeFit) -> PyResult<Option<Py<PyAny>>> {
    let Some(d) = &fit.diagnostics else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("termination", format!("{:?}", d.termination))?;
    dict.set_item("outer_iterations", d.outer_iterations)?;
    dict.set_item("inner_iterations", d.inner_iterations)?;
    dict.set_item("objective", d.objective)?;
    dict.set_item("requested_n_agq", d.requested_n_agq)?;
    dict.set_item("effective_n_agq", d.effective_n_agq)?;
    dict.set_item("warnings", &d.warnings)?;
    Ok(Some(dict.into_any().unbind()))
}
