//! Explicit categorical designs, reference grids, and null-bootstrap interfaces.
use super::*;
use std::collections::{BTreeMap, HashMap};

/// Ordered fixed-factor levels and treatment/sum coding.
#[pyclass(name = "FactorSpec", from_py_object)]
#[derive(Clone)]
pub struct PyFactorSpec {
    inner: lme_rs::FactorSpec,
}

#[pymethods]
impl PyFactorSpec {
    #[new]
    #[pyo3(signature=(levels, coding="treatment"))]
    fn new(levels: Vec<String>, coding: &str) -> PyResult<Self> {
        let coding = match coding {
            "treatment" => lme_rs::FactorCoding::Treatment,
            "sum" => lme_rs::FactorCoding::Sum,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "coding must be treatment or sum",
                ))
            }
        };
        if levels.is_empty()
            || levels
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != levels.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "levels must be nonempty and unique",
            ));
        }
        Ok(Self {
            inner: lme_rs::FactorSpec { levels, coding },
        })
    }
    #[getter]
    fn levels(&self) -> Vec<String> {
        self.inner.levels.clone()
    }
    #[getter]
    fn coding(&self) -> &'static str {
        match self.inner.coding {
            lme_rs::FactorCoding::Treatment => "treatment",
            lme_rs::FactorCoding::Sum => "sum",
        }
    }
}

pub fn factor_values(
    factors: Option<HashMap<String, PyFactorSpec>>,
) -> HashMap<String, lme_rs::FactorSpec> {
    factors
        .unwrap_or_default()
        .into_iter()
        .map(|(name, spec)| (name, spec.inner))
        .collect()
}

pub(super) fn grid(
    terms: Vec<String>,
    by: Option<Vec<String>>,
    at: Option<BTreeMap<String, f64>>,
    weights: &str,
) -> PyResult<lme_rs::ReferenceGrid> {
    Ok(lme_rs::ReferenceGrid {
        terms,
        by: by.unwrap_or_default(),
        at: at.unwrap_or_default(),
        weights: match weights {
            "equal" => lme_rs::GridWeights::Equal,
            "proportional" => lme_rs::GridWeights::Proportional,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "weights must be equal or proportional",
                ))
            }
        },
    })
}

/// Per-replicate evidence for an unconditional null-model likelihood-ratio bootstrap.
#[pyclass(name = "NullBootstrapResult", get_all)]
pub struct PyNullBootstrapResult {
    observed: f64,
    requested: usize,
    valid: usize,
    exceedances: usize,
    p_value: Option<f64>,
    mc_se: Option<f64>,
    seed: u64,
    statistics: Vec<Option<f64>>,
    errors: Vec<Option<String>>,
}

#[pyfunction]
#[pyo3(signature=(full, null, nsim, *, seed, n_jobs=1, control=None))]
pub fn bootstrap_lrt(
    py: Python<'_>,
    full: &PyLmerPrepared,
    null: &PyLmerPrepared,
    nsim: usize,
    seed: u64,
    n_jobs: usize,
    control: Option<&PyFitControl>,
) -> PyResult<PyNullBootstrapResult> {
    let control = control_value(control);
    let r = py
        .detach(|| lme_rs::bootstrap_lrt(&full.inner, &null.inner, nsim, seed, n_jobs, &control))
        .map_err(model_error)?;
    Ok(PyNullBootstrapResult {
        observed: r.observed,
        requested: r.requested,
        valid: r.valid,
        exceedances: r.exceedances,
        p_value: r.p_value,
        mc_se: r.mc_se,
        seed: r.seed,
        statistics: r.statistics,
        errors: r.errors,
    })
}
