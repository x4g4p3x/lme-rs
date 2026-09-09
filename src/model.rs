//! Shared model metadata, numerical controls, and fit diagnostics.
use crate::{formula, LmeError, LmeFit, Result};
use ndarray::Array1;

/// Reconstruct the fitted LMM likelihood for post-fit inference.
pub(crate) fn inference_lmm_data(
    fit: &LmeFit,
    matrices: &crate::model_matrix::DesignMatrices,
) -> Result<crate::math::LmmData> {
    crate::validate_observation_weights(fit.weights.as_ref(), matrices.y.len())?;
    let adjusted_y = match &matrices.offset {
        Some(offset) => &matrices.y - offset,
        None => matrices.y.clone(),
    };
    Ok(crate::math::LmmData::new_weighted(
        matrices.x.clone(),
        matrices.zt.clone(),
        adjusted_y,
        matrices.re_blocks.clone(),
        fit.weights.clone(),
    ))
}

/// Controls for LMM/GLMM optimization. Defaults retain the specialized optimizers.
#[derive(Debug, Clone, PartialEq)]
pub struct FitControl {
    /// Maximum outer optimizer iterations per search stage.
    pub max_iterations: u64,
    /// Standard deviation tolerance for the Nelder-Mead objective.
    pub tolerance: f64,
    /// Optional inner PIRLS iteration limit; `None` uses the family-specific default.
    pub max_inner_iterations: Option<usize>,
    /// Optional starting covariance parameters (theta).
    pub start: Option<Array1<f64>>,
    /// Return an error instead of a nonconverged fit when true.
    pub require_convergence: bool,
}

impl Default for FitControl {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            max_inner_iterations: None,
            start: None,
            require_convergence: false,
        }
    }
}

impl FitControl {
    /// Validate numerical limits and optional starting parameters.
    pub fn validate(&self, theta_len: usize) -> Result<()> {
        if self.max_iterations == 0
            || self.max_inner_iterations == Some(0)
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
        {
            return Err(LmeError::InvalidInput {
                message: "iteration limits and tolerance must be positive and finite".into(),
            });
        }
        if self
            .start
            .as_ref()
            .is_some_and(|x| x.len() != theta_len || !x.iter().all(|v| v.is_finite()))
        {
            return Err(LmeError::InvalidInput {
                message: format!("start must contain {theta_len} finite theta values"),
            });
        }
        Ok(())
    }

    pub(crate) fn default_search(&self) -> bool {
        self.max_iterations == 1000 && self.tolerance == 1e-6 && self.start.is_none()
    }
}

/// Why numerical fitting terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Convergence criterion was satisfied.
    Converged,
    /// An iteration budget was exhausted.
    IterationLimit,
    /// No improving step was found before convergence could be established.
    NoProgress,
    /// A valid finite numerical solution could not be obtained.
    NumericalFailure,
}

/// Structured numerical diagnostics, including approximation actually used.
#[derive(Debug, Clone)]
pub struct FitDiagnostics {
    /// Numerical termination reason.
    pub termination: TerminationReason,
    /// Outer iterations or evaluations reported by the selected optimizer.
    pub outer_iterations: u64,
    /// Inner iterations at the final covariance parameters, when applicable.
    pub inner_iterations: Option<usize>,
    /// Final objective value.
    pub objective: f64,
    /// Requested quadrature order; one means Laplace.
    pub requested_n_agq: usize,
    /// Effective quadrature order; one means Laplace.
    pub effective_n_agq: usize,
    /// Additional numerical/approximation warnings.
    pub warnings: Vec<String>,
}

impl FitDiagnostics {
    pub(crate) fn from_optimizer(opt: &crate::optimizer::OptimizeResult) -> Self {
        Self {
            termination: if opt.converged {
                TerminationReason::Converged
            } else {
                TerminationReason::IterationLimit
            },
            outer_iterations: opt.iterations,
            inner_iterations: None,
            objective: opt.final_cost,
            requested_n_agq: 1,
            effective_n_agq: 1,
            warnings: Vec::new(),
        }
    }
}

/// A borrowed view of the authoritative fitted model metadata.
///
/// Uses the fit's existing fields directly, avoiding conflicting copied metadata.
pub struct ModelSpec<'a> {
    /// Original formula, if fitted through a formula API.
    pub formula: Option<&'a str>,
    /// Observation precision/trial weights.
    pub weights: Option<&'a Array1<f64>>,
    /// Whether the model was estimated with REML.
    pub reml: bool,
    /// Fixed-effect names and order.
    pub fixed_names: Option<&'a [String]>,
    /// Number of training observations.
    pub num_obs: usize,
    fit: &'a LmeFit,
}

impl ModelSpec<'_> {
    /// Parse the formula with its original offset and variable roles.
    pub fn formula_model(&self) -> Result<formula::FormulaModel> {
        formula::parse(self.formula.ok_or_else(|| LmeError::InvalidInput {
            message: "fit has no formula metadata".into(),
        })?)
    }

    /// Validate resampling input against the fitted design specification.
    pub fn validate_refit(&self, formula: &str, num_obs: usize) -> Result<()> {
        if num_obs != self.num_obs {
            return Err(LmeError::InvalidInput {
                message: "refit data row count differs from the reference fit".into(),
            });
        }
        let mut expected = self.formula_model()?;
        let mut actual = formula::parse(formula)?;
        expected.formula.clear();
        actual.formula.clear();
        if expected != actual {
            return Err(LmeError::InvalidInput {
                message: "refit formula differs from the reference fit".into(),
            });
        }
        crate::validate_observation_weights(self.weights, num_obs)
    }

    /// Build fixed effects using the training categorical and basis encodings.
    pub fn prediction_matrix(
        &self,
        data: &polars::prelude::DataFrame,
    ) -> Result<(ndarray::Array2<f64>, Option<Array1<f64>>)> {
        let ast = self.formula_model()?;
        let response = ast
            .columns
            .iter()
            .find(|(_, c)| c.has_role(formula::ColumnRole::Response))
            .map(|(name, _)| name.as_str())
            .unwrap_or("");
        let (x, names, _, _, _) = crate::model_matrix::build_x_matrix(
            &ast,
            data,
            response,
            data.height(),
            self.fit.categorical_levels.as_ref(),
            self.fit.basis_encodings.as_ref(),
        )?;
        if Some(names.as_slice()) != self.fixed_names {
            return Err(LmeError::InvalidInput {
                message: "prediction columns do not match training columns".into(),
            });
        }
        let offset = ast
            .offset
            .as_ref()
            .map(|expr| crate::model_matrix::eval_numeric_expr(expr, data, data.height()))
            .transpose()?;
        if x.iter().any(|v| !v.is_finite())
            || offset
                .as_ref()
                .is_some_and(|o| o.iter().any(|v| !v.is_finite()))
        {
            return Err(LmeError::InvalidInput {
                message: "prediction design and offset must be finite".into(),
            });
        }
        Ok((x, offset))
    }
}

impl LmeFit {
    /// Reject numerical results that have not established convergence before inference.
    pub fn ensure_converged(&self) -> Result<()> {
        if self.converged == Some(false) {
            return Err(LmeError::NonConvergence {
                message: "fit did not converge; refit with appropriate controls before inference"
                    .into(),
            });
        }
        Ok(())
    }

    /// Inspect the metadata shared by prediction, simulation, and refitting.
    pub fn model_spec(&self) -> ModelSpec<'_> {
        ModelSpec {
            formula: self.formula.as_deref(),
            weights: self.weights.as_ref(),
            reml: self.reml.is_some(),
            fixed_names: self.fixed_names.as_deref(),
            num_obs: self.num_obs,
            fit: self,
        }
    }
}
