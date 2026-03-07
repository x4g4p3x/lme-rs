use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use polars::prelude::*;
use std::io::Cursor;
use lme_rs::LmeFit;
use lme_rs::family::Family;

#[pyclass]
pub struct PyLmeFit {
    inner: LmeFit,
}

fn get_ipc_bytes<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
    let io = py.import("io")?;
    let bytes_io = io.call_method0("BytesIO")?;
    data.call_method1("write_ipc", (&bytes_io,))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DataFrame must have a write_ipc method (e.g., polars.DataFrame): {}", e)))?;
    let py_bytes = bytes_io.call_method0("getvalue")?;
    let bytes: &Bound<'py, PyBytes> = py_bytes.downcast()?;
    Ok(bytes.as_bytes().to_vec())
}

fn read_ipc_bytes(data: &[u8]) -> PyResult<DataFrame> {
    let cursor = Cursor::new(data);
    IpcReader::new(cursor).finish()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to parse dataframe from IPC: {}", e)))
}

#[pymethods]
impl PyLmeFit {
    pub fn summary(&self) -> String {
        format!("{}", self.inner)
    }
    
    fn __str__(&self) -> String {
        self.summary()
    }

    pub fn predict<'py>(&self, py: Python<'py>, newdata: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
        let bytes = get_ipc_bytes(py, newdata)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.predict(&df) {
            Ok(arr) => Ok(arr.to_vec()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Predict failed: {}", e))),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, reml=true))]
pub fn lmer<'py>(py: Python<'py>, formula: &str, data: &Bound<'py, PyAny>, reml: bool) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    match lme_rs::lmer(formula, &df, reml) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Model fit failed: {}", e))),
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, family_name))]
pub fn glmer<'py>(py: Python<'py>, formula: &str, data: &Bound<'py, PyAny>, family_name: &str) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    let family = match family_name.to_lowercase().as_str() {
        "binomial" => Family::Binomial,
        "poisson" => Family::Poisson,
        "gamma" => Family::Gamma,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported or invalid family: {}", family_name))),
    };

    match lme_rs::glmer(formula, &df, family) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Model fit failed: {}", e))),
    }
}

#[pymodule]
fn lme_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLmeFit>()?;
    m.add_function(wrap_pyfunction!(lmer, m)?)?;
    m.add_function(wrap_pyfunction!(glmer, m)?)?;
    Ok(())
}
