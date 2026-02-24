use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use polars::prelude::DataFrame;
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::{LmeFit, LmeError};
use crate::formula::parse;
use crate::model_matrix::build_design_matrices;
use crate::math::LmmData;

/// Result of Kenward-Roger approximation
#[derive(Debug, Clone)]
pub struct KenwardRogerResult {
    pub dfs: Array1<f64>,
    pub p_values: Array1<f64>,
}

pub fn compute_kenward_roger(fit: &LmeFit, data: &DataFrame) -> crate::Result<KenwardRogerResult> {
    if fit.family.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "Kenward-Roger approximation is only available for linear mixed models (LMMs), not GLMMs.".to_string(),
        });
    }

    let theta = fit.theta.as_ref().ok_or_else(|| LmeError::NotImplemented {
        feature: "Theta represents the variance components and is required for Kenward-Roger.".to_string(),
    })?;

    let formula_str = fit.formula.as_ref().ok_or_else(|| LmeError::NotImplemented {
        feature: "Formula is missing from fit.".to_string(),
    })?;

    // We will build the exact implementation based on finite differences of Phi = (X^T V^{-1} X)^{-1}
    
    // Placeholder implementation identical to Satterthwaite for now so that it compiles
    // 1. Rebuild LmmData
    let ast = parse(formula_str)?;
    let matrices = build_design_matrices(&ast, data)?;
    
    let lmm = LmmData::new(matrices.x.clone(), matrices.zt.clone(), matrices.y.clone(), matrices.re_blocks.clone());
    let reml = fit.reml.is_some();
    
    let p = matrices.x.ncols();
    let n_theta = theta.len();

    // Step size for finite differences
    let h = 1e-4;

    // 2. Compute analytical baseline
    let base_coefs = lmm.evaluate(theta.as_slice().unwrap(), reml);
    let base_sigma2 = base_coefs.sigma2;
    let n = matrices.y.len();
    let reml_df = if reml { (n - p) as f64 } else { n as f64 };
    let twopi = std::f64::consts::PI * 2.0;
    
    // Helper to evaluate unprofiled deviance
    let unprofiled_deviance = |th: &[f64], sig2: f64| -> f64 {
        let coefs = lmm.evaluate(th, reml);
        let d_prof = coefs.reml_crit;
        let s2_hat = coefs.sigma2;
        let r2 = s2_hat * reml_df;
        let base_term = d_prof - reml_df * (twopi * s2_hat).ln() - reml_df;
        reml_df * (twopi * sig2).ln() + base_term + r2 / sig2
    };

    let n_rho = n_theta + 1; // rho = (theta, sigma2)
    let mut rho = Array1::<f64>::zeros(n_rho);
    for i in 0..n_theta { rho[i] = theta[i]; }
    rho[n_theta] = base_sigma2;

    // 3. Compute Hessian of REML deviance w.r.t rho
    let mut hessian = Array2::<f64>::zeros((n_rho, n_rho));
    for j in 0..n_rho {
        for k in j..n_rho {
            let mut r_pp = rho.clone();
            let mut r_pm = rho.clone();
            let mut r_mp = rho.clone();
            let mut r_mm = rho.clone();

            // Step size h might be problematic for sigma2 if it's large, but 1e-4 is small
            let hj = if j == n_theta { h * rho[n_theta].max(1e-4) } else { h };
            let hk = if k == n_theta { h * rho[n_theta].max(1e-4) } else { h };

            r_pp[j] += hj; r_pp[k] += hk;
            r_pm[j] += hj; r_pm[k] -= hk;
            r_mp[j] -= hj; r_mp[k] += hk;
            r_mm[j] -= hj; r_mm[k] -= hk;

            let f_pp = unprofiled_deviance(&r_pp.as_slice().unwrap()[0..n_theta], r_pp[n_theta]);
            let f_pm = unprofiled_deviance(&r_pm.as_slice().unwrap()[0..n_theta], r_pm[n_theta]);
            let f_mp = unprofiled_deviance(&r_mp.as_slice().unwrap()[0..n_theta], r_mp[n_theta]);
            let f_mm = unprofiled_deviance(&r_mm.as_slice().unwrap()[0..n_theta], r_mm[n_theta]);

            let d2f = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hj * hk);
            hessian[[j, k]] = d2f;
            hessian[[k, j]] = d2f;
        }
        
        let mut r_p = rho.clone();
        let mut r_m = rho.clone();
        let hj = if j == n_theta { h * rho[n_theta].max(1e-4) } else { h };
        r_p[j] += hj;
        r_m[j] -= hj;
        
        let f_p = unprofiled_deviance(&r_p.as_slice().unwrap()[0..n_theta], r_p[n_theta]);
        let f_m = unprofiled_deviance(&r_m.as_slice().unwrap()[0..n_theta], r_m[n_theta]);
        let f_0 = unprofiled_deviance(&rho.as_slice().unwrap()[0..n_theta], rho[n_theta]);
        
        let d2f = (f_p - 2.0 * f_0 + f_m) / (hj * hj);
        hessian[[j, j]] = d2f;
    }

    let hess_inv = hessian.inv().map_err(|e| LmeError::NotImplemented {
        feature: format!("Failed to invert Hessian for Kenward-Roger: {}", e),
    })?;
    
    // W is the inverse of the expected information matrix. Here we use 2 * Hessian^{-1}
    let w_mat = hess_inv * 2.0;

    // 4. Compute Phi = (X^T V^{-1} X)^{-1}
    let inv_lx = base_coefs.l_x.inv().map_err(|e| LmeError::NotImplemented {
        feature: format!("Failed to invert L_x: {}", e),
    })?;
    let phi_unscaled = inv_lx.t().dot(&inv_lx);
    let phi = &phi_unscaled * base_sigma2;

    // Helper for Phi_unscaled
    let get_phi_unscaled = |th: &[f64]| -> Array2<f64> {
        let c = lmm.evaluate(th, reml);
        let ilx = c.l_x.inv().unwrap();
        ilx.t().dot(&ilx)
    };

    // 5. Compute exact derivatives of Phi w.r.t rho components
    let mut phi_derivs = Vec::with_capacity(n_rho);
    for j in 0..n_rho {
        if j < n_theta {
            let mut th_p = theta.clone();
            let mut th_m = theta.clone();
            th_p[j] += h;
            th_m[j] -= h;
            
            let phi_u_p = get_phi_unscaled(th_p.as_slice().unwrap());
            let phi_u_m = get_phi_unscaled(th_m.as_slice().unwrap());
            let d_phi = (&phi_u_p - &phi_u_m) / (2.0 * h) * base_sigma2;
            phi_derivs.push(d_phi);
        } else {
            // Derivative w.r.t sigma2 is just Phi_unscaled
            phi_derivs.push(phi_unscaled.clone());
        }
    }

    // 6. Compute Q_{ij} = d^2 Phi / (dRho_i dRho_j)
    let mut q_mats = Vec::with_capacity(n_rho * n_rho);
    for i in 0..n_rho {
        for j in 0..n_rho {
            if i < n_theta && j < n_theta {
                let mut th_pp = theta.clone();
                let mut th_pm = theta.clone();
                let mut th_mp = theta.clone();
                let mut th_mm = theta.clone();

                th_pp[i] += h; th_pp[j] += h;
                th_pm[i] += h; th_pm[j] -= h;
                th_mp[i] -= h; th_mp[j] += h;
                th_mm[i] -= h; th_mm[j] -= h;

                let pu_pp = get_phi_unscaled(th_pp.as_slice().unwrap());
                let pu_pm = get_phi_unscaled(th_pm.as_slice().unwrap());
                let pu_mp = get_phi_unscaled(th_mp.as_slice().unwrap());
                let pu_mm = get_phi_unscaled(th_mm.as_slice().unwrap());

                let q_ij = (&pu_pp - &pu_pm - &pu_mp + &pu_mm) / (4.0 * h * h) * base_sigma2;
                q_mats.push(q_ij);
            } else if i < n_theta && j == n_theta {
                let mut th_p = theta.clone();
                let mut th_m = theta.clone();
                th_p[i] += h;
                th_m[i] -= h;
                let pu_p = get_phi_unscaled(th_p.as_slice().unwrap());
                let pu_m = get_phi_unscaled(th_m.as_slice().unwrap());
                q_mats.push((&pu_p - &pu_m) / (2.0 * h));
            } else if i == n_theta && j < n_theta {
                let mut th_p = theta.clone();
                let mut th_m = theta.clone();
                th_p[j] += h;
                th_m[j] -= h;
                let pu_p = get_phi_unscaled(th_p.as_slice().unwrap());
                let pu_m = get_phi_unscaled(th_m.as_slice().unwrap());
                q_mats.push((&pu_p - &pu_m) / (2.0 * h));
            } else {
                q_mats.push(Array2::<f64>::zeros((p, p)));
            }
        }
    }

    // 7. Compute \hat{\Phi}_A = \Phi + 2 \Phi \left( \sum_{i,j} W_{ij} (Q_{ij} - P_i \Phi P_j) \right) \Phi
    // Actually, following Kenward & Roger (1997), the adjusted matrix is:
    // \hat{\Phi}_A = \Phi + 2 \Phi { \sum W_{ij} (Q_{ij} - P_i \Phi^{-1} P_j) } \Phi
    // Note: The original paper uses P_i \Phi P_j instead of \Phi^{-1} depending on parametrization, 
    // but the established simplified form (via pbkrtest) uses \Lambda = \sum W_{ij} P_i \Phi^{-1} P_j.
    // Let's implement the core K-R approximation: \hat{\Phi}_A = \Phi + \sum W_{ij} (P_i \Phi^{-1} P_j) 
    // Wait, the covariance of fixed effects \Phi = (X^T V^{-1} X)^{-1}.
    // Their approximation is \hat{\Phi}_A = \Phi + 2 \Lambda
    // Where \Lambda = \Phi [ \sum W_{ij} (Q_{ij} - P_i \Phi P_j) ] \Phi.
    // Let's build \Lambda
    let inv_phi = phi.inv().unwrap();
    
    let mut bracket = Array2::<f64>::zeros((p, p));
    for i in 0..n_rho {
        for j in 0..n_rho {
            let w_ij = w_mat[[i, j]];
            let q_ij = &q_mats[i * n_rho + j];
            let p_i = &phi_derivs[i];
            let p_j = &phi_derivs[j];
            
            // P_i \Phi^{-1} P_j
            let pipj = p_i.dot(&inv_phi).dot(p_j);
            
            // bracket += W_{ij} * (D_i \Phi^{-1} D_j - Q_{ij}^{mine})
            let term = &pipj - q_ij;
            bracket = bracket + term * w_ij;
        }
    }

    let lamb = bracket;
    let phi_a = &phi + &(lamb * 2.0);

    // 8. Compute K-R degrees of freedom
    let mut dfs = Array1::<f64>::zeros(p);
    let mut p_values = Array1::<f64>::zeros(p);

    for i in 0..p {
        // Contrast vector for beta_i
        let mut c = Array1::<f64>::zeros(p);
        c[i] = 1.0;

        let c_phi_c = c.dot(&phi).dot(&c);
        let c_phi_a_c = c.dot(&phi_a).dot(&c);
        
        let mut denom = 0.0;
        for j in 0..n_rho {
            for k in 0..n_rho {
                let p_j_c = c.dot(&phi_derivs[j]).dot(&c);
                let p_k_c = c.dot(&phi_derivs[k]).dot(&c);
                denom += w_mat[[j, k]] * p_j_c * p_k_c;
            }
        }
        
        let mut df = if denom > 1e-12 {
            2.0 * (c_phi_c * c_phi_c) / denom
        } else {
            (matrices.y.len() - p) as f64
        };
        
        if df <= 0.0 || df.is_nan() {
            df = (matrices.y.len() - p) as f64;
        } else if df > 3000.0 {
            df = 3000.0;
        }
        
        dfs[i] = df;

        // F-statistic or t-statistic
        // The variance is taken from the adjusted Phi_A
        let se_adj = if c_phi_a_c > 0.0 {
            c_phi_a_c.sqrt()
        } else {
            f64::NAN
        };
        let t_stat = fit.coefficients[i] / se_adj;
        
        if t_stat.is_nan() || t_stat.is_infinite() {
            p_values[i] = f64::NAN;
        } else if let Ok(dist) = StudentsT::new(0.0, 1.0, df) {
            let p_val = 2.0 * (1.0 - dist.cdf(t_stat.abs()));
            p_values[i] = p_val;
        } else {
            p_values[i] = f64::NAN;
        }
    }

    Ok(KenwardRogerResult { dfs, p_values })
}
