use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use polars::prelude::*;
use std::io::Cursor;
use lme_rs::LmeFit;
use lme_rs::family::Family;
use lme_rs::DdfMethod;

#[pyclass]
pub struct PyLmeFit {
    inner: LmeFit,
}

type RanefRow = (String, String, String, f64);
type VarCorrRow = (String, String, String, f64, f64);
type FixedEffectsAnovaPy = (Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, String);
type LikelihoodRatioAnovaPy = (usize, usize, f64, f64, f64, usize, f64, String, String);

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

    /// Type III fixed-effects ANOVA table (requires a denominator df path).
    ///
    /// `ddf_method` can be:
    /// - `"satterthwaite"`
    /// - `"kenward_roger"` / `"kenward-roger"`
    #[pyo3(signature = (ddf_method="satterthwaite"))]
    pub fn anova(
        &self,
        ddf_method: &str,
    ) -> PyResult<FixedEffectsAnovaPy> {
        let method = match ddf_method.to_lowercase().as_str() {
            "satterthwaite" => DdfMethod::Satterthwaite,
            "kenward_roger" | "kenward-roger" | "kenwardroger" => DdfMethod::KenwardRoger,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported ddf_method '{}'",
                    other
                )))
            }
        };

        match self.inner.anova(method) {
            Ok(res) => Ok((
                res.terms,
                res.num_df.to_vec(),
                res.den_df.to_vec(),
                res.f_value.to_vec(),
                res.p_value.to_vec(),
                format!("{:?}", res.method),
            )),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "anova failed: {}",
                e
            ))),
        }
    }

    /// Parametric simulation (bootstrap) from the fitted model.
    ///
    /// Returns a list of response vectors. Each element has length `num_obs`.
    pub fn simulate(&self, nsim: usize) -> PyResult<Vec<Vec<f64>>> {
        match self.inner.simulate(nsim) {
            Ok(res) => Ok(res
                .simulations
                .into_iter()
                .map(|arr| arr.to_vec())
                .collect()),
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

#[pyfunction]
#[pyo3(signature = (formula, data, family_name, n_agq=1))]
pub fn glmer<'py>(py: Python<'py>, formula: &str, data: &Bound<'py, PyAny>, family_name: &str, n_agq: usize) -> PyResult<PyLmeFit> {
    let bytes = get_ipc_bytes(py, data)?;
    let df = read_ipc_bytes(&bytes)?;
    let family = match family_name.to_lowercase().as_str() {
        "binomial" => Family::Binomial,
        "poisson" => Family::Poisson,
        "gamma" => Family::Gamma,
        "gaussian" => Family::Gaussian,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported or invalid family: {}", family_name))),
    };

    match lme_rs::glmer(formula, &df, family, n_agq) {
        Ok(fit) => Ok(PyLmeFit { inner: fit }),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Model fit failed: {}", e))),
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

/// Likelihood ratio test between two fitted models.
///
/// Returns a tuple:
/// `(n_params_0, n_params_1, deviance_0, deviance_1, chi_sq, df, p_value, formula_0, formula_1)`.
#[pyfunction]
pub fn anova(fit_a: &PyLmeFit, fit_b: &PyLmeFit) -> PyResult<LikelihoodRatioAnovaPy> {
    match lme_rs::anova(&fit_a.inner, &fit_b.inner) {
        Ok(res) => Ok((
            res.n_params_0,
            res.n_params_1,
            res.deviance_0,
            res.deviance_1,
            res.chi_sq,
            res.df,
            res.p_value,
            res.formula_0,
            res.formula_1,
        )),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("anova failed: {}", e))),
    }
}

#[pymodule]
fn lme_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLmeFit>()?;
    m.add_function(wrap_pyfunction!(lm, m)?)?;
    m.add_function(wrap_pyfunction!(lmer, m)?)?;
    m.add_function(wrap_pyfunction!(lmer_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(glmer, m)?)?;
    m.add_function(wrap_pyfunction!(anova, m)?)?;
    Ok(())
}
