use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use polars::prelude::DataFrame;

use crate::kr_modcomp::KenwardRogerModcompData;
use crate::kr_vcov_adj;
use crate::{LmeError, LmeFit};

/// Result of Kenward-Roger approximation
#[derive(Debug, Clone)]
pub struct KenwardRogerResult {
    /// Kenward-Roger denominator degrees of freedom for each fixed effect.
    pub dfs: Array1<f64>,
    /// Two-sided p-values derived from generalized t-distributions using computed `dfs`.
    pub p_values: Array1<f64>,
    /// Matrices for multi-DoF `KRmodcomp` F-tests (`anova(..., KenwardRoger)`).
    pub(crate) modcomp: KenwardRogerModcompData,
}

pub(crate) fn invert_kr_matrix(matrix: &Array2<f64>, context: &str) -> crate::Result<Array2<f64>> {
    let inverse = matrix.inv().map_err(|error| LmeError::LinearAlgebra {
        message: format!("Kenward-Roger {context} inversion failed: {error}"),
    })?;
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(LmeError::LinearAlgebra {
            message: format!(
                "Kenward-Roger {context} inversion produced non-finite values; data may be ill-conditioned"
            ),
        });
    }
    Ok(inverse)
}

/// Derives conservative fixed effects F-tests by accounting for the small-sample
/// bias introduced through the estimation of variance components in Linear Mixed Models.
pub fn compute_kenward_roger(fit: &LmeFit, data: &DataFrame) -> crate::Result<KenwardRogerResult> {
    if fit.family.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "Kenward-Roger approximation is only available for linear mixed models (LMMs), not GLMMs.".to_string(),
        });
    }

    // Marginal inference and arbitrary contrasts must share the same structural
    // covariance adjustment and expected information, including for unbalanced data.
    let modcomp = kr_vcov_adj::kenward_roger_modcomp_data(fit, data)?;
    let p = fit.coefficients.len();
    let mut dfs = Array1::zeros(p);
    let mut p_values = Array1::zeros(p);
    for i in 0..p {
        let mut contrast = Array2::zeros((1, p));
        contrast[[0, i]] = 1.0;
        let result =
            crate::kr_modcomp::kr_modcomp_test(&modcomp, &contrast, &fit.coefficients, None)?;
        dfs[i] = result.den_df;
        p_values[i] = result.p_value;
    }

    Ok(KenwardRogerResult {
        dfs,
        p_values,
        modcomp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singular_kr_matrix_returns_structured_linear_algebra_error() {
        let singular = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let error = invert_kr_matrix(&singular, "test matrix").unwrap_err();

        match error {
            LmeError::LinearAlgebra { message } => {
                assert!(message.contains("Kenward-Roger test matrix inversion failed"));
            }
            other => panic!("expected LinearAlgebra error, got {other:?}"),
        }
    }
}
