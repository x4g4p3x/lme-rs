//! Independent reusable LMM response workspaces.
use crate::{math, FitControl, LmeError, LmeFit, LmerPrepared, Result};
use ndarray::Array1;
use std::sync::Arc;

/// Reusable numerical buffers for one worker fitting a prepared LMM.
///
/// Create one workspace per worker. Immutable design data is shared, while
/// response products and solver state are owned by this workspace.
pub struct LmerWorkspace<'a> {
    pub(crate) prepared: &'a LmerPrepared,
    lmm: Arc<math::LmmData>,
}

impl LmerPrepared {
    /// Create an independent workspace for repeated responses.
    pub fn workspace(&self) -> LmerWorkspace<'_> {
        LmerWorkspace {
            prepared: self,
            lmm: Arc::new(self.lmm.with_response(self.lmm.y.clone())),
        }
    }

    /// Fit an optional new response with numerical controls.
    pub fn fit(&self, y: Option<Array1<f64>>, reml: bool, control: &FitControl) -> Result<LmeFit> {
        crate::fit_prepared_with_control(self, y, reml, control)
    }
}

impl LmerWorkspace<'_> {
    /// Replace the response, retain solver buffers, and fit covariance parameters.
    pub fn fit_response(
        &mut self,
        y: Array1<f64>,
        reml: bool,
        control: &FitControl,
    ) -> Result<LmeFit> {
        control.validate(self.prepared.init_theta.len())?;
        if y.len() != self.lmm.y.len() || y.iter().any(|v| !v.is_finite()) {
            return Err(LmeError::InvalidInput {
                message: "response must be finite and match the prepared row count".into(),
            });
        }
        let adjusted = match &self.prepared.matrices.offset {
            Some(o) => &y - o,
            None => y.clone(),
        };
        Arc::get_mut(&mut self.lmm)
            .ok_or_else(|| LmeError::InvalidInput {
                message: "workspace is in use".into(),
            })?
            .replace_response(adjusted)?;
        let optimized = crate::optimizer::optimize_theta_lmm_control(
            Arc::clone(&self.lmm),
            self.prepared.init_theta.clone(),
            reml,
            control,
        )
        .map_err(|e| LmeError::NonConvergence {
            message: e.to_string(),
        })?;
        if control.require_convergence && !optimized.converged {
            return Err(LmeError::NonConvergence {
                message: "outer optimization did not converge".into(),
            });
        }
        crate::assemble_lme_fit(&self.lmm, &self.prepared.matrices, optimized, reml, &y)
    }
}

impl crate::GlmerPrepared {
    /// Fit an optional new response with numerical controls.
    pub fn fit(&self, y: Option<Array1<f64>>, control: &FitControl) -> Result<LmeFit> {
        crate::fit_prepared_glmer_with_control(self, y, control)
    }
}
