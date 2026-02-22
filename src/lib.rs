pub mod math;
pub mod optimizer;
pub mod formula;
pub mod model_matrix;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, QRInto};
use polars::prelude::DataFrame;
use thiserror::Error;
use std::fmt;

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
    pub theta: Option<Array1<f64>>,
    pub sigma2: Option<f64>,
    pub reml: Option<f64>,
    pub log_likelihood: Option<f64>,
    pub b: Option<Array1<f64>>,
    pub u: Option<Array1<f64>>,
    pub beta_se: Option<Array1<f64>>,
    pub beta_t: Option<Array1<f64>>,
    // Metadata for Summary Display
    pub formula: Option<String>,
    pub fixed_names: Option<Vec<String>>,
    pub re_blocks: Option<Vec<model_matrix::ReBlock>>,
    pub num_obs: usize,
}

impl fmt::Display for LmeFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(formula) = &self.formula {
            writeln!(f, "Linear mixed model fit by REML ['lmerMod']")?;
            writeln!(f, "Formula: {}", formula)?;
        }
        
        if let Some(reml) = self.reml {
            writeln!(f, "REML criterion at convergence: {:.4}", reml)?;
        }
        
        writeln!(f, "Scaled residuals:")?;
        if let Some(sigma2) = self.sigma2 {
            let sigma = sigma2.sqrt();
            let mut scaled_res: Vec<f64> = self.residuals.iter().map(|&r| r / sigma).collect();
            scaled_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = scaled_res.len();
            if n > 0 {
                let min = scaled_res[0];
                let q1 = scaled_res[n / 4];
                let median = scaled_res[n / 2];
                let q3 = scaled_res[3 * n / 4];
                let max = scaled_res[n - 1];
                writeln!(f, "    Min      1Q  Median      3Q     Max ")?;
                writeln!(f, "{:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4}", min, q1, median, q3, max)?;
            }
        }

        writeln!(f, "\nRandom effects:")?;
        writeln!(f, " Groups   Name        Variance Std.Dev.")?;
        
        if let (Some(theta), Some(re_blocks), Some(sigma2)) = (&self.theta, &self.re_blocks, self.sigma2) {
            let mut theta_idx = 0;
            let mut obs_groups = Vec::new();
            
            for block in re_blocks {
                let th = &theta.as_slice().unwrap()[theta_idx..theta_idx + block.theta_len];
                theta_idx += block.theta_len;
                
                let mut lambda = ndarray::Array2::<f64>::zeros((block.k, block.k));
                let mut idx = 0;
                for j in 0..block.k {
                    for i in j..block.k {
                        lambda[[i, j]] = th[idx];
                        idx += 1;
                    }
                }
                let cov = lambda.dot(&lambda.t()) * sigma2;
                
                for i in 0..block.k {
                    let var = cov[[i, i]];
                    let std_dev = var.sqrt();
                    let group = if i == 0 { &block.group_name } else { "" };
                    let name = &block.effect_names[i];
                    writeln!(f, " {:<8} {:<11} {:<8.4} {:<8.4}", group, name, var, std_dev)?;
                }
                obs_groups.push(format!("{}, {}", block.group_name, block.m));
            }
            writeln!(f, " Residual             {:<8.4} {:<8.4}", sigma2, sigma2.sqrt())?;
            writeln!(f, "Number of obs: {}, groups: {}", self.num_obs, obs_groups.join("; "))?;
        }

        writeln!(f, "\nFixed effects:")?;
        writeln!(f, "            Estimate Std. Error t value")?;
        
        if let (Some(fixed_names), Some(beta_se), Some(beta_t)) = (&self.fixed_names, &self.beta_se, &self.beta_t) {
            for i in 0..self.coefficients.len() {
                let name = if i < fixed_names.len() { &fixed_names[i] } else { "" };
                let est = self.coefficients[i];
                let se = beta_se[i];
                let t = beta_t[i];
                writeln!(f, "{:<11} {:>8.4} {:>10.4} {:>7.2}", name, est, se, t)?;
            }
        }
        
        Ok(())
    }
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
        theta: None,
        sigma2,
        reml: None,
        log_likelihood: None,
        b: None,
        u: None,
        beta_se: None,
        beta_t: None,
        formula: None,
        fixed_names: None,
        re_blocks: None,
        num_obs: y.len(),
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
pub fn lmer(formula_str: &str, data: &DataFrame, reml: bool) -> Result<LmeFit> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }

    // 1. Parse Wilkinson formula into AST
    let ast = formula::parse(formula_str)?;

    // 2. Build design matrices X, Zt, y from DataFrame and AST
    let matrices = model_matrix::build_design_matrices(&ast, data)?;

    // 3. Setup initial theta vector (length depends on the random effects structure)
    let total_theta_len: usize = matrices.re_blocks.iter().map(|b| b.theta_len).sum();
    let init_theta = Array1::from_vec(vec![1.0; total_theta_len]);
    
    println!("Zt shape: {}x{}, nnz: {}, theta_len: {}", 
        matrices.zt.rows(), matrices.zt.cols(), matrices.zt.nnz(), total_theta_len);
    for block in &matrices.re_blocks {
        println!("  block: m={}, k={}, theta={}", block.m, block.k, block.theta_len);
    }

    // 4. Optimize theta using Nelder-Mead
    let best_theta = optimizer::optimize_theta_nd(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
        init_theta,
        reml,
    )
    .map_err(|e| LmeError::NotImplemented {
        feature: format!("Optimizer failed: {}", e),
    })?;

    // 5. Re-evaluate to get coefficients
    let lmm = math::LmmData::new(matrices.x.clone(), matrices.zt.clone(), matrices.y.clone(), matrices.re_blocks.clone());
    let best_th_slice = best_theta.as_slice().unwrap();
    let coefs = lmm.evaluate(best_th_slice, reml);
    let reml_eval = lmm.log_reml_deviance(best_th_slice, reml);

    Ok(LmeFit {
        coefficients: coefs.beta,
        residuals: Array1::zeros(matrices.y.len()),
        fitted: Array1::zeros(matrices.y.len()),
        ranef: None,
        var_corr: None,
        theta: Some(best_theta),
        sigma2: Some(coefs.sigma2),
        reml: Some(reml_eval),
        log_likelihood: None,
        b: Some(coefs.b),
        u: Some(coefs.u),
        beta_se: Some(coefs.beta_se),
        beta_t: Some(coefs.beta_t),
        formula: Some(matrices.formula),
        fixed_names: Some(matrices.fixed_names),
        re_blocks: Some(matrices.re_blocks),
        num_obs: matrices.y.len(),
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
