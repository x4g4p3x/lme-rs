use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use polars::prelude::*;
use std::io::Cursor;
use lme_rs::contrast::contrast_matrix;
use lme_rs::family::Family;
use lme_rs::nlmm::NlmmStart;
use lme_rs::{AnovaType, DdfMethod, LmeFit};
use ndarray::Array2;

#[pyclass]
pub struct PyLmeFit {
    inner: LmeFit,
}

type RanefRow = (String, String, String, f64);
type VarCorrRow = (String, String, String, f64, f64);

/// Wald confidence intervals (`LmeFit::confint`).
#[pyclass]
#[derive(Clone)]
pub struct PyConfintResult {
    #[pyo3(get)]
    pub lower: Vec<f64>,
    #[pyo3(get)]
    pub upper: Vec<f64>,
    #[pyo3(get)]
    pub names: Vec<String>,
    #[pyo3(get)]
    pub level: f64,
}

#[pymethods]
impl PyConfintResult {
    fn __str__(&self) -> String {
        format!("{}", lme_rs::ConfintResult {
            lower: ndarray::Array1::from_vec(self.lower.clone()),
            upper: ndarray::Array1::from_vec(self.upper.clone()),
            names: self.names.clone(),
            level: self.level,
        })
    }

    /// List of `(lower, upper)` pairs (legacy tuple API).
    fn as_tuples(&self) -> Vec<(f64, f64)> {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(&lo, &hi)| (lo, hi))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.lower.len()
    }

    fn __getitem__(&self, index: usize) -> PyResult<(f64, f64)> {
        self.lower
            .get(index)
            .zip(self.upper.get(index))
            .map(|(&lo, &hi)| (lo, hi))
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "confint index {index} out of range"
                ))
            })
    }
}

/// Parametric simulation draws (`LmeFit::simulate`).
#[pyclass]
#[derive(Clone)]
pub struct PySimulateResult {
    #[pyo3(get)]
    pub simulations: Vec<Vec<f64>>,
}

#[pymethods]
impl PySimulateResult {
    fn __len__(&self) -> usize {
        self.simulations.len()
    }
}

/// Fixed-effects ANOVA table (`LmeFit::anova_typed`).
#[pyclass]
#[derive(Clone)]
pub struct PyFixedEffectsAnova {
    #[pyo3(get)]
    pub anova_type: String,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub terms: Vec<String>,
    #[pyo3(get)]
    pub num_df: Vec<f64>,
    #[pyo3(get)]
    pub den_df: Vec<f64>,
    #[pyo3(get)]
    pub f_value: Vec<f64>,
    #[pyo3(get)]
    pub p_value: Vec<f64>,
}

#[pymethods]
impl PyFixedEffectsAnova {
    fn __str__(&self) -> String {
        format!(
            "Fixed-effects ANOVA (type {}, {}):\n  terms: {:?}",
            self.anova_type, self.method, self.terms
        )
    }
}

/// Wald contrast test (`LmeFit::test_contrast`).
#[pyclass]
#[derive(Clone)]
pub struct PyContrastTest {
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub num_df: f64,
    #[pyo3(get)]
    pub den_df: f64,
    #[pyo3(get)]
    pub f_value: f64,
    #[pyo3(get)]
    pub p_value: f64,
}

#[pymethods]
impl PyContrastTest {
    fn __str__(&self) -> String {
        format!(
            "Contrast test ({}, num_df={}, den_df={}, F={:.4}, p={:.4})",
            self.method, self.num_df, self.den_df, self.f_value, self.p_value
        )
    }
}

/// Likelihood-ratio test between nested models (`lme_rs::anova`).
#[pyclass]
#[derive(Clone)]
pub struct PyLikelihoodRatioAnova {
    #[pyo3(get)]
    pub n_params_0: usize,
    #[pyo3(get)]
    pub n_params_1: usize,
    #[pyo3(get)]
    pub deviance_0: f64,
    #[pyo3(get)]
    pub deviance_1: f64,
    #[pyo3(get)]
    pub chi_sq: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub formula_0: String,
    #[pyo3(get)]
    pub formula_1: String,
}

#[pymethods]
impl PyLikelihoodRatioAnova {
    fn __str__(&self) -> String {
        format!("{}", lme_rs::AnovaResult {
            n_params_0: self.n_params_0,
            n_params_1: self.n_params_1,
            deviance_0: self.deviance_0,
            deviance_1: self.deviance_1,
            chi_sq: self.chi_sq,
            df: self.df,
            p_value: self.p_value,
            formula_0: self.formula_0.clone(),
            formula_1: self.formula_1.clone(),
        })
    }
}

/// GLMM / LMM family selector (mirrors [`lme_rs::family::Family`]).
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyFamily {
    Binomial,
    Poisson,
    Gamma,
    Gaussian,
}

impl From<PyFamily> for Family {
    fn from(f: PyFamily) -> Self {
        match f {
            PyFamily::Binomial => Family::Binomial,
            PyFamily::Poisson => Family::Poisson,
            PyFamily::Gamma => Family::Gamma,
            PyFamily::Gaussian => Family::Gaussian,
        }
    }
}

fn parse_family(family_name: &str) -> PyResult<Family> {
    match family_name.to_lowercase().as_str() {
        "binomial" => Ok(Family::Binomial),
        "poisson" => Ok(Family::Poisson),
        "gamma" => Ok(Family::Gamma),
        "gaussian" => Ok(Family::Gaussian),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported or invalid family: {other}"
        ))),
    }
}

fn parse_anova_type(anova_type: &str) -> PyResult<AnovaType> {
    match anova_type.to_uppercase().as_str() {
        "III" | "3" | "TYPE3" | "TYPE III" => Ok(AnovaType::Type3),
        "II" | "2" | "TYPE2" | "TYPE II" => Ok(AnovaType::Type2),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported anova_type '{other}'"
        ))),
    }
}

fn fixed_effects_anova_to_py(res: lme_rs::FixedEffectsAnovaResult) -> PyFixedEffectsAnova {
    PyFixedEffectsAnova {
        anova_type: format!("{:?}", res.anova_type),
        method: format!("{:?}", res.method),
        terms: res.terms,
        num_df: res.num_df.to_vec(),
        den_df: res.den_df.to_vec(),
        f_value: res.f_value.to_vec(),
        p_value: res.p_value.to_vec(),
    }
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

fn parse_nlmm_start(start: Option<&Bound<'_, PyDict>>) -> PyResult<NlmmStart> {
    let mut map = NlmmStart::new();
    if let Some(d) = start {
        for (k, v) in d {
            map.insert(k.extract()?, v.extract()?);
        }
    }
    Ok(map)
}

fn parse_ddf_method(ddf_method: &str) -> PyResult<DdfMethod> {
    match ddf_method.to_lowercase().as_str() {
        "satterthwaite" => Ok(DdfMethod::Satterthwaite),
        "kenward_roger" | "kenward-roger" | "kenwardroger" => Ok(DdfMethod::KenwardRoger),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported ddf_method '{other}'"
        ))),
    }
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

    /// Random-effects covariance parameters (θ) on the Cholesky scale (see `lme4::getME(., "theta")`).
    #[getter]
    pub fn theta(&self) -> Option<Vec<f64>> {
        self.inner.theta.as_ref().map(|t| t.to_vec())
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

    /// Formula string used to fit the model.
    #[getter]
    pub fn formula(&self) -> Option<String> {
        self.inner.formula.clone()
    }

    /// GLMM response family name (`"gaussian"`, `"binomial"`, …). `None` for LMMs.
    #[getter]
    pub fn family_name(&self) -> Option<String> {
        self.inner.family_name.clone()
    }

    /// GLMM link name. `None` for LMMs.
    #[getter]
    pub fn link_name(&self) -> Option<String> {
        self.inner.link_name.clone()
    }

    /// Family enum label when set (`"Binomial"`, `"Poisson"`, …).
    #[getter]
    pub fn family(&self) -> Option<String> {
        self.inner.family.map(|f| format!("{f:?}"))
    }

    /// Optimizer iteration count when available.
    #[getter]
    pub fn iterations(&self) -> Option<u64> {
        self.inner.iterations
    }

    /// REML criterion at convergence (REML LMMs only).
    #[getter]
    pub fn reml_criterion(&self) -> Option<f64> {
        self.inner.reml
    }

    /// Fixed-effect t- or z-values (LMM: t; GLMM: z).
    #[getter]
    pub fn beta_t(&self) -> Option<Vec<f64>> {
        self.inner.beta_t.as_ref().map(|t| t.to_vec())
    }

    /// Standard errors of fixed effects.
    #[getter]
    pub fn std_errors(&self) -> Option<Vec<f64>> {
        self.inner.beta_se.as_ref().map(|se| se.to_vec())
    }

    /// Alias for [`std_errors`](Self::std_errors) (matches Rust `LmeFit::beta_se`).
    #[getter]
    pub fn beta_se(&self) -> Option<Vec<f64>> {
        self.std_errors()
    }

    /// Fixed-effects coefficient vector (alias for [`coefficients`](Self::coefficients)).
    #[getter]
    pub fn b(&self) -> Vec<f64> {
        self.inner.coefficients.to_vec()
    }

    /// Random-effects modes **u** (when available).
    #[getter]
    pub fn u(&self) -> Option<Vec<f64>> {
        self.inner.u.as_ref().map(|u| u.to_vec())
    }

    /// Term label for each fixed-effect column (e.g. `"Days"`).
    #[getter]
    pub fn fixed_term_assign(&self) -> Option<Vec<String>> {
        self.inner.fixed_term_assign.clone()
    }

    /// Observed levels for each categorical predictor.
    #[getter]
    pub fn categorical_levels(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.categorical_levels {
            None => Ok(None),
            Some(map) => {
                let dict = PyDict::new(py);
                for (k, v) in map {
                    dict.set_item(k, v)?;
                }
                Ok(Some(dict.into()))
            }
        }
    }

    /// Unscaled covariance of **β** (V matrix before df scaling).
    #[getter]
    pub fn v_beta_unscaled(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.v_beta_unscaled.as_ref().map(|v| {
            (0..v.nrows())
                .map(|i| v.row(i).to_vec())
                .collect()
        })
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

    /// Random effects modes as a Python-friendly list of rows.
    ///
    /// Returns `None` if the fit did not compute random effects.
    /// Each row is: `(Grouping, Group, Effect, Value)`.
    #[getter]
    pub fn ranef(&self) -> PyResult<Option<Vec<RanefRow>>> {
        let df = match &self.inner.ranef {
            Some(df) => df,
            None => return Ok(None),
        };

        // Split into separate `let` bindings to avoid borrow-checker issues
        // from chaining polars iterators/borrowed chunked arrays.
        let grouping_series = df
            .column("Grouping")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e)))?;
        let grouping_series = grouping_series
            .cast(&DataType::String)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "ranef: failed casting Grouping to String: {}",
                    e
                ))
            })?;
        let grouping = grouping_series.str().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e))
        })?;

        let group_series = df
            .column("Group")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e)))?;
        let group_series = group_series.cast(&DataType::String).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "ranef: failed casting Group to String: {}",
                e
            ))
        })?;
        let group = group_series.str().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e))
        })?;

        let effect_series = df
            .column("Effect")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e)))?;
        let effect_series = effect_series.cast(&DataType::String).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "ranef: failed casting Effect to String: {}",
                e
            ))
        })?;
        let effect = effect_series.str().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e))
        })?;

        let value_series = df
            .column("Value")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e)))?;
        let value_series = value_series.cast(&DataType::Float64).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "ranef: failed casting Value to Float64: {}",
                e
            ))
        })?;
        let value = value_series.f64().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ranef: {}", e))
        })?;

        let n = df.height();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let g0 = grouping.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("ranef: missing row {}", i))
            })?;
            let g1 = group.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("ranef: missing row {}", i))
            })?;
            let e = effect.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("ranef: missing row {}", i))
            })?;
            let v = value.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("ranef: missing row {}", i))
            })?;

            out.push((
                g0.to_string(),
                g1.to_string(),
                e.to_string(),
                v,
            ));
        }

        Ok(Some(out))
    }

    /// Random-effects variance / covariance summary.
    ///
    /// Returns `None` if the fit did not compute variance-covariance information.
    /// Each row is: `(Group, Effect1, Effect2, Variance, StdDev)`.
    #[getter]
    pub fn var_corr(&self) -> PyResult<Option<Vec<VarCorrRow>>> {
        let df = match &self.inner.var_corr {
            Some(df) => df,
            None => return Ok(None),
        };

        // Split into separate `let` bindings to avoid borrow-checker issues
        // from chained polars iterators/borrowed chunked arrays.
        let group_series = df
            .column("Group")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e)))?;
        let group_series = group_series.cast(&DataType::String).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "var_corr: failed casting Group to String: {}",
                e
            ))
        })?;
        let group = group_series.str().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e))
        })?;

        let effect1_series = df
            .column("Effect1")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e)))?;
        let effect1_series = effect1_series.cast(&DataType::String).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "var_corr: failed casting Effect1 to String: {}",
                e
            ))
        })?;
        let effect1 = effect1_series.str().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e))
        })?;

        let effect2_series = df
            .column("Effect2")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e)))?;
        let effect2_series = effect2_series.cast(&DataType::String).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "var_corr: failed casting Effect2 to String: {}",
                e
            ))
        })?;
        let effect2 = effect2_series.str().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e))
        })?;

        let variance_series = df
            .column("Variance")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e)))?;
        let variance_series = variance_series.cast(&DataType::Float64).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "var_corr: failed casting Variance to Float64: {}",
                e
            ))
        })?;
        let variance = variance_series.f64().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e))
        })?;

        let stddev_series = df
            .column("StdDev")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e)))?;
        let stddev_series = stddev_series.cast(&DataType::Float64).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "var_corr: failed casting StdDev to Float64: {}",
                e
            ))
        })?;
        let stddev = stddev_series.f64().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("var_corr: {}", e))
        })?;

        let n = df.height();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let g = group.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("var_corr: missing row {}", i))
            })?;
            let e1 = effect1.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("var_corr: missing row {}", i))
            })?;
            let e2 = effect2.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("var_corr: missing row {}", i))
            })?;
            let v = variance.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("var_corr: missing row {}", i))
            })?;
            let s = stddev.get(i).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("var_corr: missing row {}", i))
            })?;

            out.push((g.to_string(), e1.to_string(), e2.to_string(), v, s));
        }

        Ok(Some(out))
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

    /// Conditional predictions on the response scale (for GLMMs).
    ///
    /// Includes random effects (Xβ + Zb), then applies the inverse link to move from
    /// the linear predictor scale to the response scale.
    ///
    /// For LMMs (identity link), this matches `predict_conditional()`.
    #[pyo3(signature = (newdata, allow_new_levels=false))]
    pub fn predict_conditional_response<'py>(
        &self,
        py: Python<'py>,
        newdata: &Bound<'py, PyAny>,
        allow_new_levels: bool,
    ) -> PyResult<Vec<f64>> {
        let bytes = get_ipc_bytes(py, newdata)?;
        let df = read_ipc_bytes(&bytes)?;
        match self
            .inner
            .predict_conditional_response(&df, allow_new_levels)
        {
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
    #[pyo3(signature = (level=0.95))]
    pub fn confint(&self, level: f64) -> PyResult<PyConfintResult> {
        match self.inner.confint(level) {
            Ok(ci) => Ok(PyConfintResult {
                lower: ci.lower.to_vec(),
                upper: ci.upper.to_vec(),
                names: ci.names,
                level: ci.level,
            }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("confint failed: {}", e))),
        }
    }

    /// Robust standard errors (requires `with_robust_se()`).
    #[getter]
    pub fn robust_se(&self) -> Option<Vec<f64>> {
        self.inner
            .robust
            .as_ref()
            .map(|r| r.robust_se.to_vec())
    }

    /// Robust t-values (requires `with_robust_se()`).
    #[getter]
    pub fn robust_t(&self) -> Option<Vec<f64>> {
        self.inner
            .robust
            .as_ref()
            .map(|r| r.robust_t.to_vec())
    }

    /// Robust p-values (requires `with_robust_se()`).
    #[getter]
    pub fn robust_p_values(&self) -> Option<Vec<f64>> {
        self.inner
            .robust
            .as_ref()
            .and_then(|r| r.robust_p_values.as_ref().map(|p| p.to_vec()))
    }

    /// Satterthwaite denominator degrees of freedom (requires `with_satterthwaite()`).
    #[getter]
    pub fn satterthwaite_dfs(&self) -> Option<Vec<f64>> {
        self.inner
            .satterthwaite
            .as_ref()
            .map(|r| r.dfs.to_vec())
    }

    /// Satterthwaite p-values (requires `with_satterthwaite()`).
    #[getter]
    pub fn satterthwaite_p_values(&self) -> Option<Vec<f64>> {
        self.inner
            .satterthwaite
            .as_ref()
            .map(|r| r.p_values.to_vec())
    }

    /// Kenward-Roger denominator degrees of freedom (requires `with_kenward_roger()`).
    #[getter]
    pub fn kenward_roger_dfs(&self) -> Option<Vec<f64>> {
        self.inner
            .kenward_roger
            .as_ref()
            .map(|r| r.dfs.to_vec())
    }

    /// Kenward-Roger p-values (requires `with_kenward_roger()`).
    #[getter]
    pub fn kenward_roger_p_values(&self) -> Option<Vec<f64>> {
        self.inner
            .kenward_roger
            .as_ref()
            .map(|r| r.p_values.to_vec())
    }

    /// Fixed-effects ANOVA table (requires a denominator df path).
    ///
    /// `ddf_method` can be:
    /// - `"satterthwaite"`
    /// - `"kenward_roger"` / `"kenward-roger"`
    ///
    /// `anova_type` can be `"III"` (default) or `"II"`.
    #[pyo3(signature = (ddf_method="satterthwaite", anova_type="III"))]
    pub fn anova(
        &self,
        ddf_method: &str,
        anova_type: &str,
    ) -> PyResult<PyFixedEffectsAnova> {
        let method = parse_ddf_method(ddf_method)?;
        let atype = parse_anova_type(anova_type)?;
        match self.inner.anova_typed(atype, method) {
            Ok(res) => Ok(fixed_effects_anova_to_py(res)),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "anova failed: {}",
                e
            ))),
        }
    }

    /// Wald F-test for a user-defined contrast matrix **L** (H₀: L β = 0).
    ///
    /// `l_matrix` is a list of rows; each row is a list of length `p` (number of fixed effects),
    /// aligned with `fit.fixed_names` order.
    ///
    #[pyo3(signature = (l_matrix, ddf_method="satterthwaite"))]
    pub fn test_contrast(
        &self,
        l_matrix: Vec<Vec<f64>>,
        ddf_method: &str,
    ) -> PyResult<PyContrastTest> {
        let method = parse_ddf_method(ddf_method)?;
        let p = self.inner.coefficients.len();
        let l = matrix_from_rows(l_matrix, p)?;
        self.run_test_contrast(&l, None, method)
    }

    /// Wald F-test for **H₀: L β = β_h** with a full-length null vector **β_h** (length **p**).
    #[pyo3(signature = (l_matrix, beta_h, ddf_method="satterthwaite"))]
    pub fn test_contrast_vs(
        &self,
        l_matrix: Vec<Vec<f64>>,
        beta_h: Vec<f64>,
        ddf_method: &str,
    ) -> PyResult<PyContrastTest> {
        let method = parse_ddf_method(ddf_method)?;
        let p = self.inner.coefficients.len();
        if beta_h.len() != p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "beta_h has length {} but the model has {} coefficients",
                beta_h.len(),
                p
            )));
        }
        let l = matrix_from_rows(l_matrix, p)?;
        let h = ndarray::Array1::from_vec(beta_h);
        self.run_test_contrast(&l, Some(&h), method)
    }

    /// Parametric simulation (bootstrap) from the fitted model.
    ///
    pub fn simulate(&self, nsim: usize) -> PyResult<PySimulateResult> {
        match self.inner.simulate(nsim) {
            Ok(res) => Ok(PySimulateResult {
                simulations: res
                    .simulations
                    .into_iter()
                    .map(|arr| arr.to_vec())
                    .collect(),
            }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("simulate failed: {}", e))),
        }
    }

    /// Compute robust (Sandwich) standard errors and p-values.
    ///
    /// If `cluster_col` is provided, computes cluster-robust standard errors.
    #[pyo3(signature = (data, cluster_col=None))]
    pub fn with_robust_se<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        cluster_col: Option<&str>,
    ) -> PyResult<()> {
        let bytes = get_ipc_bytes(py, data)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.with_robust_se(&df, cluster_col) {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("with_robust_se failed: {}", e))),
        }
    }

    /// Compute Satterthwaite degrees of freedom and p-values.
    #[pyo3(signature = (data))]
    pub fn with_satterthwaite<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let bytes = get_ipc_bytes(py, data)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.with_satterthwaite(&df) {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("with_satterthwaite failed: {}", e))),
        }
    }

    /// Compute Kenward-Roger degrees of freedom and p-values.
    ///
    /// Results match R's `pbkrtest` on covered LMM configurations.
    #[pyo3(signature = (data))]
    pub fn with_kenward_roger<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let bytes = get_ipc_bytes(py, data)?;
        let df = read_ipc_bytes(&bytes)?;
        match self.inner.with_kenward_roger(&df) {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("with_kenward_roger failed: {}", e))),
        }
    }
}

impl PyLmeFit {
    fn run_test_contrast(
        &self,
        l: &ndarray::Array2<f64>,
        beta_h: Option<&ndarray::Array1<f64>>,
        method: DdfMethod,
    ) -> PyResult<PyContrastTest> {
        let res = if let Some(h) = beta_h {
            self.inner.test_contrast_vs(l, h, method)
        } else {
            self.inner.test_contrast(l, method)
        };
        match res {
            Ok(r) => Ok(PyContrastTest {
                method: format!("{:?}", r.method),
                num_df: r.num_df,
                den_df: r.den_df,
                f_value: r.f_value,
                p_value: r.p_value,
            }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "test_contrast failed: {e}"
            ))),
        }
    }
}

fn matrix_from_rows(l_matrix: Vec<Vec<f64>>, p: usize) -> PyResult<ndarray::Array2<f64>> {
    let q = l_matrix.len();
    if q == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "l_matrix must have at least one row",
        ));
    }
    let mut l = ndarray::Array2::<f64>::zeros((q, p));
    for (i, row) in l_matrix.iter().enumerate() {
        if row.len() != p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Row {i} has length {} but the model has {p} coefficients",
                row.len()
            )));
        }
        for (j, &v) in row.iter().enumerate() {
            l[[i, j]] = v;
        }
    }
    Ok(l)
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

/// Fit a fixed-effects-only linear model from a Wilkinson formula string.
///
/// Parameters
/// ----------
/// formula : str
///     Wilkinson formula, e.g. ``"y ~ x1 + x2"`` or ``"y ~ 1"``.
/// data : polars.DataFrame
///     DataFrame containing the variables referenced in the formula.
///
/// Returns
/// -------
/// PyLmeFit
///     Fitted model with ``.coefficients``, ``.fixed_names``, ``.aic``,
///     ``.bic``, ``.log_likelihood``, ``.fitted``, ``.residuals``,
///     ``.std_errors``, and ``.summary()``.
#[pyfunction]
#[pyo3(signature = (formula, data))]
pub fn lm<'py>(py: Python<'py>, formula: &str, data: &Bound<'py, PyAny>) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    match lme_rs::lm_df(formula, &df) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Model fit failed: {}", e))),
    }
}

/// Fit OLS from numeric **y** and design matrix **X** (mirrors Rust `lm(y, x)`).
///
/// `x` is `n_obs × p` as a list of rows.
#[pyfunction]
pub fn lm_matrix(y: Vec<f64>, x: Vec<Vec<f64>>) -> PyResult<PyLmeFit> {
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
    match lme_rs::lm(&y_arr, &mat) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Model fit failed: {e}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, family_name, n_agq=1))]
pub fn glmer<'py>(py: Python<'py>, formula: &str, data: &Bound<'py, PyAny>, family_name: &str, n_agq: usize) -> PyResult<PyLmeFit> {
    glmer_weighted(py, formula, data, family_name, n_agq, None)
}

#[pyfunction]
#[pyo3(signature = (formula, data, family_name, n_agq=1, weights=None))]
pub fn glmer_weighted<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    family_name: &str,
    n_agq: usize,
    weights: Option<Vec<f64>>,
) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    let family = parse_family(family_name)?;
    let weights_arr = weights.map(ndarray::Array1::from_vec);

    match lme_rs::glmer_weighted(formula, &df, family, n_agq, weights_arr) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Model fit failed: {}",
            e
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, reml=true, weights=None))]
pub fn lmer_weighted<'py>(py: Python<'py>, formula: &str, data: &Bound<'py, PyAny>, reml: bool, weights: Option<Vec<f64>>) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    let weights_arr = weights.map(ndarray::Array1::from_vec);
    match lme_rs::lmer_weighted(formula, &df, reml, weights_arr) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Model fit failed: {}", e))),
    }
}

/// Build a **q × p** contrast matrix from named coefficient weights.
///
/// Each row is a list of `(coefficient_name, weight)` pairs, aligned with `fixed_names`.
#[pyfunction]
#[pyo3(name = "contrast_matrix_from_names")]
fn contrast_matrix_from_names_py(
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
    Ok((0..mat.nrows())
        .map(|i| mat.row(i).to_vec())
        .collect())
}

/// Build a **q × p** contrast matrix from `(column_index, weight)` rows (Rust `contrast_matrix`).
#[pyfunction]
#[pyo3(name = "contrast_matrix")]
pub fn contrast_matrix_py(p: usize, rows: Vec<Vec<(usize, f64)>>) -> PyResult<Vec<Vec<f64>>> {
    let mat = contrast_matrix(p, &rows);
    Ok((0..mat.nrows())
        .map(|i| mat.row(i).to_vec())
        .collect())
}

/// Fit a nonlinear mixed-effects model (`SSlogis` mean; random effect on one NL parameter).
#[pyfunction]
#[pyo3(signature = (formula, data, start=None, reml=false))]
pub fn nlmer<'py>(
    py: Python<'py>,
    formula: &str,
    data: &Bound<'py, PyAny>,
    start: Option<&Bound<'py, PyDict>>,
    reml: bool,
) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    let start_map = parse_nlmm_start(start)?;
    match lme_rs::nlmer(formula, &df, start_map, reml) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Model fit failed: {e}"
        ))),
    }
}

/// Likelihood ratio test between two nested fitted models.
#[pyfunction]
pub fn anova(fit_a: &PyLmeFit, fit_b: &PyLmeFit) -> PyResult<PyLikelihoodRatioAnova> {
    match lme_rs::anova(&fit_a.inner, &fit_b.inner) {
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
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("anova failed: {}", e))),
    }
}

#[pymodule]
fn lme_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLmeFit>()?;
    m.add_class::<PyConfintResult>()?;
    m.add_class::<PySimulateResult>()?;
    m.add_class::<PyFixedEffectsAnova>()?;
    m.add_class::<PyContrastTest>()?;
    m.add_class::<PyLikelihoodRatioAnova>()?;
    m.add_class::<PyFamily>()?;
    m.add_function(wrap_pyfunction!(lm, m)?)?;
    m.add_function(wrap_pyfunction!(lm_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(lmer, m)?)?;
    m.add_function(wrap_pyfunction!(lmer_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(glmer, m)?)?;
    m.add_function(wrap_pyfunction!(glmer_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(nlmer, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_matrix_from_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(anova, m)?)?;
    m.setattr(
        "__all__",
        [
            "PyLmeFit",
            "PyConfintResult",
            "PySimulateResult",
            "PyFixedEffectsAnova",
            "PyContrastTest",
            "PyLikelihoodRatioAnova",
            "PyFamily",
            "lm",
            "lm_matrix",
            "lmer",
            "lmer_weighted",
            "glmer",
            "glmer_weighted",
            "nlmer",
            "anova",
            "contrast_matrix",
            "contrast_matrix_from_names",
        ],
    )?;
    Ok(())
}
