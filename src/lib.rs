pub mod math;
pub mod optimizer;
pub mod formula;
pub mod model_matrix;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, QRInto};
use polars::prelude::DataFrame;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, LmeError>;

#[derive(Debug, Error)]
pub enum LmeError {
    #[error("dimension mismatch: y has length {y_len}, X has {x_rows} rows")]
    DimensionMismatch { y_len: usize, x_rows: usize },
    #[error("design matrix has fewer rows ({rows}) than columns ({cols}); system is underdetermined")]
    Underdetermined { rows: usize, cols: usize },
    #[error("linear algebra failure: {message}")]
    LinearAlgebra { message: String },
    #[error("formula is empty")]
    EmptyFormula,
    #[error("not implemented: {feature}")]
    NotImplemented { feature: String },
}

#[derive(Debug, Clone)]
pub struct LmeFit {
    pub coefficients: Array1<f64>,
    pub residuals: Array1<f64>,
    pub fitted: Array1<f64>,
    pub ranef: Option<DataFrame>,
    pub var_corr: Option<DataFrame>,
    pub sigma2: Option<f64>,
    pub log_likelihood: Option<f64>,
}

/// Fit a linear model `y = X * beta + e` using a QR decomposition.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use lme_rs::lm;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let y = array![1.0, 2.0, 3.0];
///     let x = array![
///         [1.0, 1.0],
///         [1.0, 2.0],
///         [1.0, 3.0],
///     ];
///
///     let fit = lm(&y, &x)?;
///     assert_eq!(fit.coefficients.len(), 2);
///     assert_eq!(fit.residuals.len(), 3);
///     Ok(())
/// }
/// ```
pub fn lm(y: &Array1<f64>, x: &Array2<f64>) -> Result<LmeFit> {
    if y.len() != x.nrows() {
        return Err(LmeError::DimensionMismatch {
            y_len: y.len(),
            x_rows: x.nrows(),
        });
    }
    if x.nrows() < x.ncols() {
        return Err(LmeError::Underdetermined {
            rows: x.nrows(),
            cols: x.ncols(),
        });
    }

    // Solve beta = R^{-1} Q^T y for full-column-rank X = Q R.
    let (q, r) = x
        .clone()
        .qr_into()
        .map_err(|e| LmeError::LinearAlgebra { message: e.to_string() })?;
    let qty = q.t().dot(y);
    let r_inv = r
        .inv()
        .map_err(|e| LmeError::LinearAlgebra { message: e.to_string() })?;
    let coefficients = r_inv.dot(&qty);

    let fitted = x.dot(&coefficients);
    let residuals = y - &fitted;
    let sigma2 = if y.len() > x.ncols() {
        Some(residuals.dot(&residuals) / ((y.len() - x.ncols()) as f64))
    } else {
        None
    };

    Ok(LmeFit {
        coefficients,
        residuals,
        fitted,
        ranef: None,
        var_corr: None,
        sigma2,
        log_likelihood: None,
    })
}

/// Fit a linear mixed-effects model.
///
/// # Examples
///
/// ```
/// use polars::prelude::*;
/// use lme_rs::lmer;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // let df = DataFrame::new(vec![])?;
///     // let result = lmer("Reaction ~ 1 + (1|Subject)", &df)?;
///     Ok(())
/// }
/// ```
pub fn lmer(formula_str: &str, data: &DataFrame) -> Result<LmeFit> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }

    // 1. Parse Wilkinson formula into AST
    let ast = formula::parse(formula_str)?;

    // 2. Build design matrices X, Zt, y from DataFrame and AST
    let matrices = model_matrix::build_design_matrices(&ast, data)?;

    // 3. Optimize theta
    let best_theta = optimizer::optimize_theta_1d(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        1e-5,
        10.0,
    )
    .map_err(|e| LmeError::NotImplemented {
        feature: format!("Optimizer failed: {}", e),
    })?;

    // Optionally: Re-evaluate to get coefficients
    // For now we just return a stub containing the optimized variance component
    Ok(LmeFit {
        coefficients: Array1::zeros(matrices.x.ncols()),
        residuals: Array1::zeros(matrices.y.len()),
        fitted: Array1::zeros(matrices.y.len()),
        ranef: None,
        var_corr: None,
        sigma2: Some(best_theta), // Hack: returning best_theta here for testing
        log_likelihood: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lm_recovers_simple_line() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ];

        let fit = lm(&y, &x).expect("lm should fit a full-rank design matrix");
        assert!((fit.coefficients[0] - 0.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 1.0).abs() < 1e-10);
    }
}
