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
    /// Return the R-style model summary.
    pub fn summary(&self) -> String {
        format!("{}", self.inner)
    }
    
    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!("PyLmeFit(formula={:?}, n_obs={})", 
            self.inner.formula.as_deref().unwrap_or("?"),
            self.inner.num_obs)
    }

    /// Fixed-effects coefficients (β).
    #[getter]
    pub fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients.to_vec()
    }

    /// Names of fixed effects.
    #[getter]
    pub fn fixed_names(&self) -> Option<Vec<String>> {
        self.inner.fixed_names.clone()
    }

    /// Residual variance (σ²). None for GLMMs without dispersion.
    #[getter]
    pub fn sigma2(&self) -> Option<f64> {
        self.inner.sigma2
    }

    /// Akaike Information Criterion.
    #[getter]
    pub fn aic(&self) -> Option<f64> {
        self.inner.aic
    }

    /// Bayesian Information Criterion.
    #[getter]
    pub fn bic(&self) -> Option<f64> {
        self.inner.bic
    }

    /// Log-likelihood.
    #[getter]
    pub fn log_likelihood(&self) -> Option<f64> {
        self.inner.log_likelihood
    }

    /// Model deviance.
    #[getter]
    pub fn deviance(&self) -> Option<f64> {
        self.inner.deviance
    }

    /// Whether the optimizer converged.
    #[getter]
    pub fn converged(&self) -> Option<bool> {
        self.inner.converged
    }

    /// Number of observations.
    #[getter]
    pub fn num_obs(&self) -> usize {
        self.inner.num_obs
    }

    /// Standard errors of fixed effects.
    #[getter]
    pub fn std_errors(&self) -> Option<Vec<f64>> {
        self.inner.beta_se.as_ref().map(|se| se.to_vec())
    }

    /// Residuals (y - fitted).
    #[getter]
    pub fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.to_vec()
    }

    /// Fitted values.
    #[getter]
    pub fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.to_vec()
    }

    /// Population-level predictions (Xβ).
    pub fn predict<'py>(&self, py: Python<'py>, newdata: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
        let bytes = get_ipc_bytes(py, newdata)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.predict(&df) {
            Ok(arr) => Ok(arr.to_vec()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Predict failed: {}", e))),
        }
    }

    /// Conditional predictions including random effects (Xβ + Zb).
    #[pyo3(signature = (newdata, allow_new_levels=false))]
    pub fn predict_conditional<'py>(&self, py: Python<'py>, newdata: &Bound<'py, PyAny>, allow_new_levels: bool) -> PyResult<Vec<f64>> {
        let bytes = get_ipc_bytes(py, newdata)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.predict_conditional(&df, allow_new_levels) {
            Ok(arr) => Ok(arr.to_vec()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Predict failed: {}", e))),
        }
    }

    /// Population-level predictions on the response scale (for GLMMs).
    pub fn predict_response<'py>(&self, py: Python<'py>, newdata: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
        let bytes = get_ipc_bytes(py, newdata)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.predict_response(&df) {
            Ok(arr) => Ok(arr.to_vec()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Predict failed: {}", e))),
        }
    }

    /// Wald confidence intervals for fixed effects.
    /// Returns a list of (lower, upper) tuples.
    #[pyo3(signature = (level=0.95))]
    pub fn confint(&self, level: f64) -> PyResult<Vec<(f64, f64)>> {
        match self.inner.confint(level) {
            Ok(ci) => {
                let mut result = Vec::new();
                for i in 0..ci.lower.len() {
                    result.push((ci.lower[i], ci.upper[i]));
                }
                Ok(result)
            },
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("confint failed: {}", e))),
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
