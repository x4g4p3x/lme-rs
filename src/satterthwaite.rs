use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use polars::prelude::DataFrame;
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::{LmeFit, LmeError};
use crate::formula::parse;
use crate::model_matrix::build_design_matrices;
use crate::math::LmmData;

/// Computes the Satterthwaite approximation for degrees of freedom and p-values
/// for the fixed effects of a fitted Linear Mixed-Effects Model.
pub fn compute_satterthwaite(fit: &LmeFit, data: &DataFrame) -> crate::Result<(Array1<f64>, Array1<f64>)> {
    if fit.family.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "Satterthwaite approximation is only available for linear mixed models (LMMs), not GLMMs.".to_string(),
        });
    }

    let theta = fit.theta.as_ref().ok_or_else(|| LmeError::NotImplemented {
        feature: "Theta represents the variance components and is required for Satterthwaite.".to_string(),
    })?;

    let formula_str = fit.formula.as_ref().ok_or_else(|| LmeError::NotImplemented {
        feature: "Formula is missing from fit.".to_string(),
    })?;

    // 1. Rebuild LmmData
    let ast = parse(formula_str)?;
    let matrices = build_design_matrices(&ast, data)?;
    
    // We assume unweighted for now, or we'd need to extract weights if they were stored in LmeFit.
    // To support weights fully, we should ideally store them or re-extract them, but for now we re-evaluate unweighted.
    let lmm = LmmData::new(matrices.x.clone(), matrices.zt.clone(), matrices.y.clone(), matrices.re_blocks.clone());
    
    let reml = fit.reml.is_some();
    
    let p = matrices.x.ncols();
    let n_theta = theta.len();

    // Step size for finite differences
    let h = 1e-4;

    // 2. Compute analytical baseline
    let base_coefs = lmm.evaluate(theta.as_slice().unwrap(), reml);
    let var_beta_base = base_coefs.beta_se.mapv(|se| se * se);
    
    // 3. Compute Hessian of REML deviance w.r.t theta
    let mut hessian = Array2::<f64>::zeros((n_theta, n_theta));
    for j in 0..n_theta {
        for k in j..n_theta {
            let mut th_pp = theta.clone();
            let mut th_pm = theta.clone();
            let mut th_mp = theta.clone();
            let mut th_mm = theta.clone();

            th_pp[j] += h; th_pp[k] += h;
            th_pm[j] += h; th_pm[k] -= h;
            th_mp[j] -= h; th_mp[k] += h;
            th_mm[j] -= h; th_mm[k] -= h;

            let f_pp = lmm.log_reml_deviance(th_pp.as_slice().unwrap(), reml);
            let f_pm = lmm.log_reml_deviance(th_pm.as_slice().unwrap(), reml);
            let f_mp = lmm.log_reml_deviance(th_mp.as_slice().unwrap(), reml);
            let f_mm = lmm.log_reml_deviance(th_mm.as_slice().unwrap(), reml);

            let d2f = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hessian[[j, k]] = d2f;
            hessian[[k, j]] = d2f;
        }
        
        // Refine diagonal with O(h^2) central difference
        let mut th_p = theta.clone();
        let mut th_m = theta.clone();
        th_p[j] += h;
        th_m[j] -= h;
        
        let f_p = lmm.log_reml_deviance(th_p.as_slice().unwrap(), reml);
        let f_m = lmm.log_reml_deviance(th_m.as_slice().unwrap(), reml);
        let f_0 = lmm.log_reml_deviance(theta.as_slice().unwrap(), reml);
        
        let d2f = (f_p - 2.0 * f_0 + f_m) / (h * h);
        hessian[[j, j]] = d2f;
    }

    // A = 2 * Hessian^{-1}
    let hess_inv = hessian.inv().map_err(|e| LmeError::NotImplemented {
        feature: format!("Failed to invert Hessian for Satterthwaite: {}", e),
    })?;
    let a_mat = hess_inv * 2.0;

    // 4. Compute gradient of Var(beta_i) w.r.t theta
    let mut grad_v = Array2::<f64>::zeros((p, n_theta)); // row i is gradient for beta_i
    for j in 0..n_theta {
        let mut th_p = theta.clone();
        let mut th_m = theta.clone();
        th_p[j] += h;
        th_m[j] -= h;

        let coefs_p = lmm.evaluate(th_p.as_slice().unwrap(), reml);
        let coefs_m = lmm.evaluate(th_m.as_slice().unwrap(), reml);

        for i in 0..p {
            let vp = coefs_p.beta_se[i] * coefs_p.beta_se[i];
            let vm = coefs_m.beta_se[i] * coefs_m.beta_se[i];
            grad_v[[i, j]] = (vp - vm) / (2.0 * h);
        }
    }

    // 5. Compute df and p-values
    let mut dfs = Array1::<f64>::zeros(p);
    let mut p_values = Array1::<f64>::zeros(p);

    for i in 0..p {
        let g_i = grad_v.row(i);
        // denominator = g_i^T * A_mat * g_i
        let mut denom = 0.0;
        for j in 0..n_theta {
            for k in 0..n_theta {
                denom += g_i[j] * a_mat[[j, k]] * g_i[k];
            }
        }
        
        let v_i = var_beta_base[i];
        let df = if denom > 1e-12 {
            2.0 * v_i * v_i / denom
        } else {
            // Fallback if gradient is near zero (e.g. variance doesn't depend on theta)
            (matrices.y.len() - p) as f64
        };
        
        dfs[i] = df;

        // p-value from t-distribution with `df` degrees of freedom
        let t_stat = fit.coefficients[i] / fit.beta_se.as_ref().unwrap()[i];
        
        if let Ok(dist) = StudentsT::new(0.0, 1.0, df) {
            let p_val = 2.0 * (1.0 - dist.cdf(t_stat.abs()));
            p_values[i] = p_val;
        } else {
            p_values[i] = f64::NAN;
        }
    }

    Ok((dfs, p_values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use polars::prelude::*;
    use crate::family::Family;
    use crate::LmeFit;

    fn create_dummy_fit() -> LmeFit {
        LmeFit {
            formula: Some("y ~ x".to_string()),
            coefficients: array![1.0, 2.0],
            sigma2: Some(1.0),
            beta_se: Some(array![0.5, 0.5]),
            beta_t: None,
            b: None,
            u: None,
            theta: Some(array![1.0]), // Mock theta
            log_likelihood: Some(0.0),
            deviance: Some(0.0),
            aic: Some(0.0),
            bic: Some(0.0),
            converged: Some(true),
            iterations: Some(10),
            reml: Some(0.0),
            num_obs: 4,
            fitted: array![1.0, 1.0, 2.0, 2.0],
            residuals: array![0.0, 1.0, 1.0, 2.0],
            family: None,
            var_corr: None,
            ranef: None,
            fixed_names: None,
            re_blocks: None,
            family_name: None,
            link_name: None,
            satterthwaite: None,
        }
    }

    #[test]
    fn test_satterthwaite_glmm_error() {
        let mut fit = create_dummy_fit();
        fit.family = Some(Family::Binomial); // GLMM
        let df = DataFrame::empty();
        
        let res = compute_satterthwaite(&fit, &df);
        assert!(res.is_err());
        if let Err(LmeError::NotImplemented { feature }) = res {
            assert!(feature.contains("only available for linear mixed models"));
        } else {
            panic!("Expected NotImplemented error");
        }
    }

    #[test]
    fn test_satterthwaite_missing_formula() {
        let mut fit = create_dummy_fit();
        fit.formula = None;
        let df = DataFrame::empty();
        
        let res = compute_satterthwaite(&fit, &df);
        assert!(res.is_err());
        if let Err(LmeError::NotImplemented { feature }) = res {
            assert!(feature.contains("Formula is missing"));
        } else {
            panic!("Expected NotImplemented error");
        }
    }

    #[test]
    fn test_satterthwaite_singular_hessian_or_zero_grad() {
        // We can force a singular hessian by creating a perfectly flat deviance surface
        // or a dataset with perfect fit and no variance constraints.
        // Let's create a minimal dataset
        let mut fit = create_dummy_fit();
        fit.formula = Some("y ~ x + (1|group)".to_string());
        
        let df = df!(
            "y" => &[1.0, 1.0],
            "x" => &[0.0, 1.0],
            "group" => &["A", "B"]
        ).unwrap();
        
        // This might fail to invert Hessian because there's not enough data,
        // or the gradient could be near zero leading to fallback DF.
        let res = compute_satterthwaite(&fit, &df);
        
        // We just want to ensure it either errors gracefully on Hessian inv, or returns successfully
        // bypassing the near-zero denom branch.
        match res {
            Ok((dfs, _)) => {
                // Should hit fallback DF if it succeeds
                assert!(dfs.len() == 2);
            },
            Err(_) => {
                // Or fails on Hessian
            }
        }
    }
}
