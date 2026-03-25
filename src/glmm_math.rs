//! GLMM deviance computation using the Penalized Iteratively Reweighted Least Squares (PIRLS) algorithm.
//!
//! This module is the GLMM counterpart of `math.rs` for LMMs.
//! For a given theta (variance parameters), it runs the PIRLS inner loop to find
//! the conditional modes of the random effects and fixed effects, then computes
//! the Laplace-approximated deviance, or adaptive Gauss–Hermite quadrature for a single random-effect
//! block when `n_agq > 1` (scalar `k = 1` per group, or product quadrature for vector `k > 1`).
//! With **multiple** random-effect terms, joint AGQ over the full `u` is used when `q` is small.

use crate::family::GlmFamily;
use crate::model_matrix::ReBlock;
use ndarray::{Array1, Array2};
use ndarray_linalg::UPLO;
use ndarray_linalg::{Cholesky, Inverse, Solve};
use sprs::{CsMat, TriMat};
use std::collections::HashSet;

mod gh_rule_tables {
    #![allow(clippy::excessive_precision)]
    // lme4 / fastGHQuad-style Gauss–Hermite rules: z = sqrt(2) * x (physicist roots x), weights sum to 1.
    pub const GH_Z3: [f64; 3] = [-1.7320508075688772, 0.0, 1.7320508075688772];
    pub const GH_W3: [f64; 3] = [0.16666666666666669, 0.6666666666666665, 0.16666666666666669];
    pub const GH_Z5: [f64; 5] = [
        -2.8569700138728056,
        -1.3556261799742659,
        0.0,
        1.3556261799742659,
        2.8569700138728056,
    ];
    pub const GH_W5: [f64; 5] = [
        0.011257411327720693,
        0.2220759220056126,
        0.5333333333333333,
        0.2220759220056126,
        0.011257411327720693,
    ];
    pub const GH_Z7: [f64; 7] = [
        -3.7504397177257425,
        -2.3667594107345416,
        -1.1544053947399682,
        0.0,
        1.1544053947399682,
        2.3667594107345416,
        3.7504397177257425,
    ];
    pub const GH_W7: [f64; 7] = [
        0.0005482688559722182,
        0.03075712396758651,
        0.24012317860501273,
        0.45714285714285713,
        0.24012317860501273,
        0.03075712396758651,
        0.0005482688559722182,
    ];
    pub const GH_Z9: [f64; 9] = [
        -4.512745863399783,
        -3.2054290028564703,
        -2.07684797867783,
        -1.0232556637891326,
        0.0,
        1.0232556637891326,
        2.07684797867783,
        3.2054290028564703,
        4.512745863399783,
    ];
    pub const GH_W9: [f64; 9] = [
        2.2345844007746576e-5,
        0.0027891413212317653,
        0.04991640676521791,
        0.2440975028949394,
        0.4063492063492064,
        0.2440975028949394,
        0.04991640676521791,
        0.0027891413212317653,
        2.2345844007746576e-5,
    ];
    pub const GH_Z11: [f64; 11] = [
        -5.188001224374871,
        -3.9361666071299766,
        -2.8651231606436456,
        -1.876035020154846,
        -0.9288689973810641,
        0.0,
        0.9288689973810641,
        1.876035020154846,
        2.8651231606436456,
        3.9361666071299766,
        5.188001224374871,
    ];
    pub const GH_W11: [f64; 11] = [
        8.121849790214923e-7,
        0.00019567193027122338,
        0.006720285235537264,
        0.06613874607105782,
        0.24224029987396992,
        0.3694083694083694,
        0.24224029987396992,
        0.06613874607105782,
        0.006720285235537264,
        0.00019567193027122338,
        8.121849790214923e-7,
    ];
}

fn resolve_gh_order(n_agq: usize) -> Option<usize> {
    if n_agq < 2 {
        return None;
    }
    const SUPPORTED: [usize; 5] = [3, 5, 7, 9, 11];
    for &s in &SUPPORTED {
        if s >= n_agq {
            return Some(s);
        }
    }
    Some(11)
}

fn gh_rule(order: usize) -> Option<(&'static [f64], &'static [f64])> {
    use gh_rule_tables::*;
    match order {
        3 => Some((&GH_Z3[..], &GH_W3[..])),
        5 => Some((&GH_Z5[..], &GH_W5[..])),
        7 => Some((&GH_Z7[..], &GH_W7[..])),
        9 => Some((&GH_Z9[..], &GH_W9[..])),
        11 => Some((&GH_Z11[..], &GH_W11[..])),
        _ => None,
    }
}

fn csr_diag_entry(a: &CsMat<f64>, i: usize) -> Option<f64> {
    let row = a.outer_view(i)?;
    for (col, &val) in row.iter() {
        if col == i {
            return Some(val);
        }
    }
    None
}

fn csr_get(a: &CsMat<f64>, i: usize, j: usize) -> f64 {
    let Some(row) = a.outer_view(i) else {
        return 0.0;
    };
    for (col, &val) in row.iter() {
        if col == j {
            return val;
        }
    }
    0.0
}

fn csr_dense_block(a: &CsMat<f64>, start: usize, k: usize) -> Array2<f64> {
    let mut mat = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            mat[[i, j]] = csr_get(a, start + i, start + j);
        }
    }
    mat
}

/// Maximum spherical RE dimension for **joint** multivariate AGQ (multiple `re_blocks`, full `u`).
const AGQ_JOINT_MAX_Q: usize = 8;

/// Picks a 1D quadrature order; for `k > 1`, reduces `order` so `order^k` stays bounded.
fn resolve_gh_order_product(n_agq: usize, k: usize) -> Option<usize> {
    let base = resolve_gh_order(n_agq)?;
    if k == 1 {
        return Some(base);
    }
    const MAX_POINTS: usize = 400;
    let mut ord = base;
    loop {
        if ord.pow(k as u32) <= MAX_POINTS {
            return Some(ord);
        }
        ord = match ord {
            11 => 9,
            9 => 7,
            7 => 5,
            5 => 3,
            3 => return None,
            _ => return None,
        };
    }
}

/// Product-grid size cap for joint integration over all `q` random effects at once.
fn resolve_gh_order_joint(n_agq: usize, q: usize) -> Option<usize> {
    if q == 0 || q > AGQ_JOINT_MAX_Q {
        return None;
    }
    let base = resolve_gh_order(n_agq)?;
    // Allow order=3 with q=8 (3^8=6561) when n_agq is small; cap keeps memory bounded.
    const MAX_POINTS: usize = 7000;
    let mut ord = base;
    loop {
        if ord.pow(q as u32) <= MAX_POINTS {
            return Some(ord);
        }
        ord = match ord {
            11 => 9,
            9 => 7,
            7 => 5,
            5 => 3,
            3 => return None,
            _ => return None,
        };
    }
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|x| (x - m).exp()).sum();
    m + s.ln()
}

/// Encapsulates the design matrices and family for a GLMM evaluation.
pub struct GlmmData {
    /// Dense fixed-effects design matrix ($X$).
    pub x: Array2<f64>,
    /// Sparse transposed random-effects design matrix ($Z^T$).
    pub zt: CsMat<f64>,
    /// Dependent variable vector ($y$).
    pub y: Array1<f64>,
    /// Collection of random effect dimensional tracking blocks.
    pub re_blocks: Vec<ReBlock>,
    /// Family distribution specification detailing variance and link properties.
    pub family: Box<dyn GlmFamily>,
    // Cached structural matrices
    /// Cross product of the transposed design matrix ($Z^T Z$).
    pub zt_z: CsMat<f64>,
    // Precomputed mapping for Z^t W Z update: (col_in_Zt, zt_z_data_idx, value_product)
    /// Track mapped positions to rapidly update dense/sparse intermediate combinations without recreating arrays.
    pub zt_w_z_map: Vec<(usize, usize, f64)>,
}

/// Result of PIRLS convergence for a single theta evaluation.
#[derive(Debug, Clone)]
pub struct GlmmCoefficients {
    /// Laplace-approximated deviance (cost for optimizer).
    pub deviance: f64,
    /// Fixed effects coefficients.
    pub beta: Array1<f64>,
    /// Conditional modes of random effects (b = Λu).
    pub b: Array1<f64>,
    /// Spherical random effects.
    pub u: Array1<f64>,
    /// Fitted values on the response scale (μ = g⁻¹(η)).
    pub fitted: Array1<f64>,
    /// Linear predictor η = Xβ + Zb.
    pub eta: Array1<f64>,
    /// Working residuals on the response scale.
    pub residuals: Array1<f64>,
    /// Standard errors for fixed effects.
    pub beta_se: Array1<f64>,
    /// z-values for fixed effects (beta / se).
    pub beta_z: Array1<f64>,
    /// Unscaled variance-covariance matrix of fixed effects.
    pub v_beta_unscaled: Array2<f64>,
}

impl GlmmData {
    /// Construct a fresh GLMM data block capturing design matrices and specific generative distributions,
    /// pre-building the index maps used repeatedly during inner PIRLS loops for $Z^T W Z$ weight updates.
    pub fn new(
        x: Array2<f64>,
        zt: CsMat<f64>,
        y: Array1<f64>,
        re_blocks: Vec<ReBlock>,
        family: Box<dyn GlmFamily>,
        _n_agq: usize,
    ) -> Self {
        let zt_z = &zt * &zt.transpose_view();

        let z_csc = zt.to_csc();
        let mut zt_w_z_map = Vec::new();
        let n = zt.cols();
        for k in 0..n {
            if let Some(col_k) = z_csc.outer_view(k) {
                let indices = col_k.indices();
                let data = col_k.data();
                for i in 0..indices.len() {
                    for j in 0..indices.len() {
                        let row_a = indices[i];
                        let row_b = indices[j];
                        let val = data[i] * data[j];

                        // find data index in zt_z for (row_a, row_b)
                        let row_view = zt_z.outer_view(row_a).unwrap();
                        let mut data_idx = None;

                        let base_ptr = zt_z.data().as_ptr() as usize;
                        let row_ptr = row_view.data().as_ptr() as usize;
                        let start_idx = (row_ptr - base_ptr) / std::mem::size_of::<f64>();

                        for (offset, &col) in row_view.indices().iter().enumerate() {
                            if col == row_b {
                                data_idx = Some(start_idx + offset);
                                break;
                            }
                        }
                        if let Some(idx) = data_idx {
                            zt_w_z_map.push((k, idx, val));
                        }
                    }
                }
            }
        }

        GlmmData {
            x,
            zt,
            y,
            re_blocks,
            family,
            zt_z,
            zt_w_z_map,
        }
    }

    /// Compute Laplace or AGQ approximated deviance.
    pub fn laplace_deviance(
        &mut self,
        theta: &[f64],
        offset: Option<&Array1<f64>>,
        n_agq: usize,
    ) -> f64 {
        match self.pirls(theta, offset, n_agq) {
            Some(coefs) => coefs.deviance,
            None => f64::MAX, // Return large value for invalid regions
        }
    }

    /// Compute coefficients and final structural parameters at the MLE / REML theta.
    pub fn evaluate(
        &mut self,
        theta: &[f64],
        offset: Option<&Array1<f64>>,
        n_agq: usize,
    ) -> Option<GlmmCoefficients> {
        self.pirls(theta, offset, n_agq)
    }

    /// Run the full PIRLS algorithm for a given theta.
    /// Returns `None` if PIRLS fails to converge or hits numerical issues.
    pub fn pirls(
        &mut self,
        theta: &[f64],
        offset: Option<&Array1<f64>>,
        n_agq: usize,
    ) -> Option<GlmmCoefficients> {
        let n = self.y.len();
        let q = self.zt.rows();
        let p = self.x.ncols();
        let link = self.family.link();

        // Build Lambda from theta (same as LMM)
        let lambda = self.build_lambda(theta, q);

        // Initialize: mu from family, eta from link
        let mut mu = self.family.initialize_mu(&self.y);
        let mut eta = link.link_fun(&mu);
        if let Some(off) = offset {
            eta += off;
        }

        // Unit weights (no prior weights for now)
        let wt = Array1::ones(n);

        let max_iter = if link.name() == "inverse" { 1000 } else { 100 };
        let tol = 1e-8;
        let mut old_pwrss = f64::MAX;

        // Initialize u to zero
        let mut u = Array1::<f64>::zeros(q);

        let mut beta = Array1::<f64>::zeros(p);

        for _iter in 0..max_iter {
            // 1. Compute working weights and working response
            let mu_eta_val = link.mu_eta(&eta);
            let var_mu = self.family.variance(&mu);

            // Working weights: w = (dmu/deta)^2 / V(mu)
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                let me = mu_eta_val[i];
                let v = var_mu[i].max(f64::EPSILON);
                w[i] = (me * me / v).max(f64::EPSILON);
            }

            // Working response: z = eta + (y - mu) / (dmu/deta)
            let mut z = Array1::<f64>::zeros(n);
            for i in 0..n {
                let me_raw = mu_eta_val[i];
                let me = if me_raw.abs() < f64::EPSILON {
                    if me_raw.is_sign_negative() {
                        -f64::EPSILON
                    } else {
                        f64::EPSILON
                    }
                } else {
                    me_raw
                };
                z[i] = eta[i] + (self.y[i] - mu[i]) / me;
            }

            // 2. Solve the penalized weighted least squares system
            // A = Λ'Z'WZΛ + I
            let lam_t = lambda.transpose_view();

            // Build W*Z and related products
            // We need: Λ'Z'WZΛ which requires weighting Z by sqrt(W)
            // Compute ZΛ products column-by-column through sparse ops

            // Build Z^T W Z in-place using the precomputed map
            let mut zt_w_z = self.zt_z.clone();
            for v in zt_w_z.data_mut() {
                *v = 0.0;
            }
            for &(k, data_idx, val) in &self.zt_w_z_map {
                zt_w_z.data_mut()[data_idx] += val * w[k];
            }
            let a_part = &lam_t * &(&zt_w_z * &lambda);

            let mut eye_tri = TriMat::new((q, q));
            for i in 0..q {
                eye_tri.add_triplet(i, i, 1.0);
            }
            let eye: CsMat<f64> = eye_tri.to_csr();
            let a = &a_part + &eye;

            // LDLT decomposition of A
            use sprs::SymmetryCheck;
            use sprs_ldl::Ldl;
            let ldl = match Ldl::new()
                .check_symmetry(SymmetryCheck::DontCheckSymmetry)
                .numeric(a.view())
            {
                Ok(l) => l,
                Err(e) => {
                    log::debug!("LDLT decomposition failed: {:?}", e);
                    return None;
                }
            };

            // Compute weighted Zt * (z - X*beta_old) for the u-update
            // But we solve for both u and beta simultaneously
            // RHS_u = Λ'Z'W(z)
            let wz = &w * &z; // element-wise w*z
            let mut zt_wz = Array1::<f64>::zeros(q);
            for (val, (row, col)) in self.zt.iter() {
                zt_wz[row] += val * wz[col];
            }
            let mut v_y = Array1::<f64>::zeros(q);
            for (val, (row, col)) in lam_t.iter() {
                v_y[row] += val * zt_wz[col];
            }

            let w_y_vec: Vec<f64> = ldl.solve(v_y.to_vec());
            let w_y = Array1::from_vec(w_y_vec);

            // RHS for beta: X'W*z and X'W*Z*Λ terms
            let mut w_diag_x = Array2::<f64>::zeros((n, p));
            for j in 0..p {
                for i in 0..n {
                    w_diag_x[[i, j]] = self.x[[i, j]] * w[i];
                }
            }
            let xt_w_x = w_diag_x.t().dot(&self.x);
            let xt_wz_vec = w_diag_x.t().dot(&z);

            // Compute RZX = L⁻¹ Λ'Z'WX per column (same approach as LMM in math.rs)
            let mut w_cols = Vec::with_capacity(p);
            let mut v_cols = Vec::with_capacity(p);
            for j in 0..p {
                // Extract weighted X column: W * x_col
                let mut wx_col = Array1::<f64>::zeros(n);
                for i in 0..n {
                    wx_col[i] = self.x[[i, j]] * w[i];
                }
                // Z'W x_col
                let mut zt_wx_j = Array1::<f64>::zeros(q);
                for (val, (row, col)) in self.zt.iter() {
                    zt_wx_j[row] += val * wx_col[col];
                }
                // Λ' Z'W x_col
                let mut v_j = Array1::<f64>::zeros(q);
                for (val, (row, col)) in lam_t.iter() {
                    v_j[row] += val * zt_wx_j[col];
                }

                let w_j_vec: Vec<f64> = ldl.solve(v_j.to_vec());
                let w_j = Array1::from_vec(w_j_vec);
                v_cols.push(v_j);
                w_cols.push(w_j);
            }

            // RZX'RZX
            let mut rzx_t_rzx = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    rzx_t_rzx[[i, j]] = v_cols[i].dot(&w_cols[j]);
                }
            }

            // RZX'cu
            let mut rzx_t_cu = Array1::<f64>::zeros(p);
            for i in 0..p {
                rzx_t_cu[i] = v_cols[i].dot(&w_y);
            }

            // Downdated X'WX
            let a_x = &xt_w_x - &rzx_t_rzx;
            let l_x = match a_x.cholesky(UPLO::Lower) {
                Ok(l) => l,
                Err(e) => {
                    log::debug!("Cholesky of X'WX downdate failed: {:?}", e);
                    log::debug!("theta: {:?}", theta);
                    log::debug!("beta: {:?}", beta);
                    log::debug!("xt_w_x diag: {:?}", xt_w_x.diag());
                    log::debug!("a_x diag: {:?}", a_x.diag());
                    return None;
                }
            };

            // Solve for beta
            let rhs_beta = &xt_wz_vec - &rzx_t_cu;
            let c_beta = match l_x.solve(&rhs_beta) {
                Ok(c) => c,
                Err(e) => {
                    log::debug!("Solve for c_beta failed: {:?}", e);
                    return None;
                }
            };
            beta = match l_x.t().solve(&c_beta) {
                Ok(b) => b,
                Err(e) => {
                    log::debug!("Solve for beta failed: {:?}", e);
                    return None;
                }
            };

            // Solve for u
            for i in 0..q {
                let mut w_beta_i = 0.0;
                for j in 0..p {
                    w_beta_i += w_cols[j][i] * beta[j];
                }
                u[i] = w_y[i] - w_beta_i;
            }

            // Update eta, mu
            let mut b = Array1::<f64>::zeros(q);
            for (val, (row, col)) in lambda.iter() {
                b[row] += val * u[col];
            }
            let x_beta = self.x.dot(&beta);

            // Compute Z*b
            let mut z_b_vec = vec![0.0f64; n];
            for (j, row_vec) in self.zt.outer_iterator().enumerate() {
                for (i, &val) in row_vec.iter() {
                    z_b_vec[i] += val * b[j];
                }
            }
            let z_b = Array1::from_vec(z_b_vec);

            let eta_prev = eta.clone();
            let mut eta_raw = &x_beta + &z_b;
            if let Some(off) = offset {
                eta_raw += off;
            }

            // Dampen PIRLS updates only when the proposed linear predictor steps
            // outside the link's numerically stable region, which is especially
            // important for Gamma with the inverse link.
            let mut accepted = None;
            let delta_eta = &eta_raw - &eta_prev;
            let needs_positive_eta = link.name() == "inverse";
            let mut step = 1.0;
            let max_backoff = if needs_positive_eta { 100 } else { 25 };
            for _ in 0..max_backoff {
                let eta_candidate = &eta_prev + &(delta_eta.clone() * step);

                let eta_valid = eta_candidate
                    .iter()
                    .all(|e| e.is_finite() && (!needs_positive_eta || *e > f64::EPSILON));
                if !eta_valid {
                    step *= 0.5;
                    continue;
                }

                let mu_candidate = link.link_inv(&eta_candidate);
                if !mu_candidate.iter().all(|m| m.is_finite()) {
                    step *= 0.5;
                    continue;
                }

                let mut pwrss_candidate = u.dot(&u);
                for i in 0..n {
                    pwrss_candidate += w[i] * (z[i] - eta_candidate[i]).powi(2);
                }
                if pwrss_candidate.is_finite() {
                    accepted = Some((eta_candidate, mu_candidate, pwrss_candidate));
                    break;
                }

                step *= 0.5;
            }

            let Some((eta_next, mu_next, pwrss)) = accepted else {
                log::debug!("PIRLS step-halving failed to find a valid update");
                return None;
            };

            eta = eta_next;
            mu = mu_next;

            log::debug!(
                "PIRLS Iter {}: old_pwrss = {}, new_pwrss = {}",
                _iter,
                old_pwrss,
                pwrss
            );

            if (old_pwrss - pwrss).abs() / (pwrss + 0.1) < tol {
                // Compute Laplace deviance
                let dev_resid = self.family.dev_resid(&self.y, &mu, &wt);
                let sum_dev_resid: f64 = dev_resid.sum();

                // log|A| from LDLT
                let mut log_det_a = 0.0;
                for &d in ldl.d() {
                    log_det_a += d.abs().ln();
                }

                // Laplace-approximated conditional deviance for the optimizer.
                let mut deviance = sum_dev_resid + log_det_a + u.dot(&u);

                if n_agq > 1 {
                    if self.re_blocks.len() == 1 {
                        if let Some(d_agq) =
                            self.agq_deviance(n_agq, &a, &u, &beta, &lambda, offset, &mu)
                        {
                            deviance = d_agq;
                        }
                    } else if u.len() <= AGQ_JOINT_MAX_Q {
                        if let Some(d_agq) =
                            self.agq_deviance_joint(n_agq, &a, &u, &beta, &lambda, offset, &mu)
                        {
                            deviance = d_agq;
                        } else {
                            log::debug!(
                                "Joint AGQ failed (singular Hessian or grid too large); using Laplace."
                            );
                        }
                    } else {
                        log::warn!(
                            "nAGQ > 1 with multiple random-effect terms requires total RE dimension q <= {} (got {}). Falling back to Laplace (nAGQ = 1).",
                            AGQ_JOINT_MAX_Q,
                            u.len()
                        );
                    }
                }

                // Standard errors for fixed effects
                let mut beta_se = Array1::<f64>::zeros(p);
                let mut beta_z = Array1::<f64>::zeros(p);
                let mut v_beta_unscaled = Array2::<f64>::zeros((p, p));
                if let Ok(inv_lx) = l_x.inv() {
                    v_beta_unscaled = inv_lx.t().dot(&inv_lx);
                    for i in 0..p {
                        // For GLMMs without dispersion, sigma2=1
                        let var_i = v_beta_unscaled[[i, i]];
                        beta_se[i] = var_i.sqrt();
                        if beta_se[i] > 0.0 {
                            beta_z[i] = beta[i] / beta_se[i];
                        }
                    }
                }

                let residuals = &self.y - &mu;

                return Some(GlmmCoefficients {
                    deviance,
                    beta,
                    b,
                    u,
                    fitted: mu,
                    eta,
                    residuals,
                    beta_se,
                    beta_z,
                    v_beta_unscaled,
                });
            }

            old_pwrss = pwrss;
        }

        // If we reach here, PIRLS did not converge — still return last result with a warning
        log::warn!("PIRLS did not converge within {} iterations", max_iter);

        let mut b = Array1::<f64>::zeros(q);
        for (val, (row, col)) in lambda.iter() {
            b[row] += val * u[col];
        }
        Some(GlmmCoefficients {
            deviance: f64::MAX,
            beta,
            b,
            u,
            eta,
            residuals: &self.y - &mu,
            fitted: mu,
            beta_se: Array1::zeros(p),
            beta_z: Array1::zeros(p),
            v_beta_unscaled: Array2::zeros((p, p)),
        })
    }

    /// Adaptive Gauss–Hermite quadrature for a **single** random-effect block.
    ///
    /// - `k = 1`: same convention as `lme4::GHrule` — `u = u_hat + σ z` with `σ = 1 / sqrt(A_{gg})`.
    /// - `k > 1`: tensor-product rules on `N(0, I_k)` with `u_quad = u_hat_block + L z`, where
    ///   `Σ = A_block^{-1}` and `L` is the lower Cholesky factor of `Σ` (block of `A` at the mode).
    ///
    /// Returns `sum_dev(μ_hat) + Σ_g Q_g`. `None` triggers Laplace fallback (singular block, etc.).
    #[allow(clippy::too_many_arguments)]
    fn agq_deviance(
        &self,
        n_agq: usize,
        a: &CsMat<f64>,
        u_hat: &Array1<f64>,
        beta: &Array1<f64>,
        lambda: &CsMat<f64>,
        offset: Option<&Array1<f64>>,
        mu_hat: &Array1<f64>,
    ) -> Option<f64> {
        let block = self.re_blocks.first()?;
        let k = block.k;
        let m = block.m;
        if u_hat.len() != m * k {
            return None;
        }

        let order = resolve_gh_order_product(n_agq, k)?;
        let (z, w) = gh_rule(order)?;
        let n_nodes = z.len();
        let n = self.y.len();
        let wt = Array1::ones(n);
        let dev_hat = self.family.dev_resid(&self.y, mu_hat, &wt);
        let sum_dev = dev_hat.sum();

        let mut total_q = 0.0;

        if k == 1 {
            for g in 0..m {
                let a_gg = csr_diag_entry(a, g)?;
                if a_gg <= f64::EPSILON || !a_gg.is_finite() {
                    return None;
                }
                let sigma = a_gg.sqrt().recip();
                let u_hat_g = u_hat[g];

                let obs_idx: Vec<usize> = self
                    .zt
                    .outer_view(g)
                    .map(|row| row.iter().map(|(col, _)| col).collect())
                    .unwrap_or_default();

                let mut u_trial = u_hat.clone();
                let mut log_terms = Vec::with_capacity(n_nodes);
                for (ki, &z_k) in z.iter().enumerate() {
                    let u_quad = u_hat_g + sigma * z_k;
                    u_trial[g] = u_quad;

                    let eta = self.linear_predictor_from_u(&u_trial, beta, lambda, offset);
                    let mu = self.family.link().link_inv(&eta);
                    let dev_new = self.family.dev_resid(&self.y, &mu, &wt);

                    let mut dev_diff = 0.0;
                    for &i in &obs_idx {
                        dev_diff += dev_new[i] - dev_hat[i];
                    }
                    let u_pen = u_quad * u_quad - u_hat_g * u_hat_g;
                    let delta = -0.5 * dev_diff - 0.5 * u_pen;

                    log_terms.push(w[ki].ln() + delta);
                }

                let log_inner = log_sum_exp(&log_terms);
                if !log_inner.is_finite() {
                    return None;
                }
                total_q += -2.0 * log_inner;
            }
        } else {
            let ncomb = n_nodes.pow(k as u32);
            for g in 0..m {
                let start = g * k;
                let a_block = csr_dense_block(a, start, k);
                let sigma = a_block.inv().ok()?;
                let l_sigma = sigma.cholesky(UPLO::Lower).ok()?;

                let mut obs_set = HashSet::new();
                for r in start..start + k {
                    if let Some(row) = self.zt.outer_view(r) {
                        for (col, _) in row.iter() {
                            obs_set.insert(col);
                        }
                    }
                }
                let obs_idx: Vec<usize> = obs_set.into_iter().collect();

                let mut u_trial = u_hat.clone();
                let mut z_vec = Array1::<f64>::zeros(k);
                let mut u_delta = vec![0.0_f64; k];
                let mut log_terms = Vec::with_capacity(ncomb);
                for t in 0..ncomb {
                    let mut rem = t;
                    let mut log_w = 0.0_f64;
                    for dim in 0..k {
                        let idx = rem % n_nodes;
                        rem /= n_nodes;
                        z_vec[dim] = z[idx];
                        log_w += w[idx].ln();
                    }

                    for i in 0..k {
                        let mut s = 0.0_f64;
                        for j in 0..k {
                            s += l_sigma[[i, j]] * z_vec[j];
                        }
                        u_delta[i] = s;
                    }
                    for i in 0..k {
                        u_trial[start + i] = u_hat[start + i] + u_delta[i];
                    }

                    let eta = self.linear_predictor_from_u(&u_trial, beta, lambda, offset);
                    let mu = self.family.link().link_inv(&eta);
                    let dev_new = self.family.dev_resid(&self.y, &mu, &wt);

                    let mut dev_diff = 0.0;
                    for &i in &obs_idx {
                        dev_diff += dev_new[i] - dev_hat[i];
                    }
                    let mut u_pen = 0.0_f64;
                    for i in start..start + k {
                        u_pen += u_trial[i] * u_trial[i] - u_hat[i] * u_hat[i];
                    }
                    let delta = -0.5 * dev_diff - 0.5 * u_pen;

                    log_terms.push(log_w + delta);
                }

                let log_inner = log_sum_exp(&log_terms);
                if !log_inner.is_finite() {
                    return None;
                }
                total_q += -2.0 * log_inner;
            }
        }

        Some(sum_dev + total_q)
    }

    /// Joint multivariate AGQ over the **entire** spherical vector `u` (used when `re_blocks.len() > 1` and `q` is small).
    ///
    /// This approximates `∫ φ(u) exp(ℓ(u) - ℓ*) du` with a single tensor-product rule and
    /// `u_quad = u_hat + L z`, `Σ = A^{-1}`, `L L^T = Σ` for the full `q`×`q` Hessian block.
    #[allow(clippy::too_many_arguments)]
    fn agq_deviance_joint(
        &self,
        n_agq: usize,
        a: &CsMat<f64>,
        u_hat: &Array1<f64>,
        beta: &Array1<f64>,
        lambda: &CsMat<f64>,
        offset: Option<&Array1<f64>>,
        mu_hat: &Array1<f64>,
    ) -> Option<f64> {
        let q = u_hat.len();
        if q > AGQ_JOINT_MAX_Q {
            return None;
        }

        let order = resolve_gh_order_joint(n_agq, q)?;
        let (z, w) = gh_rule(order)?;
        let n_nodes = z.len();
        let ncomb = n_nodes.pow(q as u32);
        let n = self.y.len();
        let wt = Array1::ones(n);
        let dev_hat = self.family.dev_resid(&self.y, mu_hat, &wt);
        let sum_dev = dev_hat.sum();

        let a_dense = csr_dense_block(a, 0, q);
        let sigma = a_dense.inv().ok()?;
        let l_sigma = sigma.cholesky(UPLO::Lower).ok()?;

        let u_hat_sq_sum: f64 = u_hat.iter().map(|x| x * x).sum();
        let mut z_vec = vec![0.0_f64; q];
        let mut u_delta = vec![0.0_f64; q];
        let mut u_trial = u_hat.clone();
        let mut log_terms = Vec::with_capacity(ncomb);
        for t in 0..ncomb {
            let mut rem = t;
            let mut log_w = 0.0_f64;
            for z_slot in z_vec.iter_mut() {
                let idx = rem % n_nodes;
                rem /= n_nodes;
                *z_slot = z[idx];
                log_w += w[idx].ln();
            }

            for i in 0..q {
                let mut s = 0.0_f64;
                for j in 0..q {
                    s += l_sigma[[i, j]] * z_vec[j];
                }
                u_delta[i] = s;
            }

            for i in 0..q {
                u_trial[i] = u_hat[i] + u_delta[i];
            }

            let eta = self.linear_predictor_from_u(&u_trial, beta, lambda, offset);
            let mu = self.family.link().link_inv(&eta);
            let dev_new = self.family.dev_resid(&self.y, &mu, &wt);

            let mut dev_diff = 0.0;
            for i in 0..n {
                dev_diff += dev_new[i] - dev_hat[i];
            }
            let u_trial_sq_sum: f64 = u_trial.iter().map(|x| x * x).sum();
            let u_pen = u_trial_sq_sum - u_hat_sq_sum;
            let delta = -0.5 * dev_diff - 0.5 * u_pen;

            log_terms.push(log_w + delta);
        }

        let log_inner = log_sum_exp(&log_terms);
        if !log_inner.is_finite() {
            return None;
        }
        let total_q = -2.0 * log_inner;
        Some(sum_dev + total_q)
    }

    fn linear_predictor_from_u(
        &self,
        u: &Array1<f64>,
        beta: &Array1<f64>,
        lambda: &CsMat<f64>,
        offset: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        let q = self.zt.rows();
        let mut b = Array1::<f64>::zeros(q);
        for (val, (row, col)) in lambda.iter() {
            b[row] += val * u[col];
        }
        let n = self.y.len();
        let mut z_b = vec![0.0f64; n];
        for (j, row_vec) in self.zt.outer_iterator().enumerate() {
            for (i, &val) in row_vec.iter() {
                z_b[i] += val * b[j];
            }
        }
        let mut eta = self.x.dot(beta) + Array1::from_vec(z_b);
        if let Some(off) = offset {
            eta += off;
        }
        eta
    }

    /// Build the sparse Lambda matrix from theta (identical to LMM logic).
    fn build_lambda(&self, theta: &[f64], q: usize) -> CsMat<f64> {
        let mut lam_tri = TriMat::new((q, q));
        let mut row_offset = 0;
        let mut theta_offset = 0;

        for block in &self.re_blocks {
            let m = block.m;
            let k = block.k;

            for group in 0..m {
                let offset = row_offset + group * k;
                let mut idx = 0;
                for j in 0..k {
                    for i in j..k {
                        lam_tri.add_triplet(offset + i, offset + j, theta[theta_offset + idx]);
                        idx += 1;
                    }
                }
            }
            row_offset += m * k;
            theta_offset += block.theta_len;
        }

        lam_tri.to_csr()
    }
}

/// Multiply a sparse CsMat (q×n) by a dense Array2 (n×p) to produce a dense CsMat or Array2.
/// We return a CsMat for compatibility with the sparse pipeline, but this is actually a dense result.
fn _sparse_times_dense(sp: &CsMat<f64>, dense: &ndarray::ArrayView2<f64>) -> CsMat<f64> {
    let q = sp.rows();
    let p = dense.ncols();
    let mut tri = TriMat::new((q, p));

    for (j, row_vec) in sp.outer_iterator().enumerate() {
        for col in 0..p {
            let mut val = 0.0;
            for (i, &v) in row_vec.iter() {
                val += v * dense[[i, col]];
            }
            if val.abs() > 1e-15 {
                tri.add_triplet(j, col, val);
            }
        }
    }

    tri.to_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::BinomialFamily;
    use ndarray::{array, Array2};
    use sprs::TriMat;

    #[test]
    fn test_sparse_times_dense() {
        // Create 2x3 sparse
        let mut sp_tri = TriMat::new((2, 3));
        sp_tri.add_triplet(0, 0, 1.0);
        sp_tri.add_triplet(1, 2, 2.0);
        let sp = sp_tri.to_csr();

        // Create 3x2 dense
        let dense = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let result = _sparse_times_dense(&sp, &dense.view());
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        let mat = result.to_dense();
        assert_eq!(mat[[0, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 2.0);
        assert_eq!(mat[[1, 0]], 10.0);
        assert_eq!(mat[[1, 1]], 12.0);
    }

    #[test]
    fn test_log_sum_exp_basic() {
        let xs = [0.0_f64, 1.0, 2.0];
        let lse = log_sum_exp(&xs);
        let expected = (1.0 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((lse - expected).abs() < 1e-12);
    }

    #[test]
    fn test_agq_scalar_re_deviance_finite() {
        let x = Array2::<f64>::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .unwrap();
        let mut zt_tri = TriMat::new((2, 4));
        zt_tri.add_triplet(0, 0, 1.0);
        zt_tri.add_triplet(0, 1, 1.0);
        zt_tri.add_triplet(1, 2, 1.0);
        zt_tri.add_triplet(1, 3, 1.0);
        let zt = zt_tri.to_csr();

        let y = array![0.0, 1.0, 0.0, 1.0];

        let re_blocks = vec![ReBlock {
            m: 2,
            k: 1,
            theta_len: 1,
            group_name: "G".to_string(),
            effect_names: vec!["(Intercept)".to_string()],
            group_map: std::collections::HashMap::new(),
        }];

        let fam = Box::new(BinomialFamily::new());
        let mut glmm_lap =
            GlmmData::new(x.clone(), zt.clone(), y.clone(), re_blocks.clone(), fam, 1);
        let fam2 = Box::new(BinomialFamily::new());
        let mut glmm_agq = GlmmData::new(x, zt, y, re_blocks, fam2, 7);

        let theta = [0.8];
        let d_lap = glmm_lap.laplace_deviance(&theta, None, 1);
        let d_agq = glmm_agq.laplace_deviance(&theta, None, 7);
        assert!(d_lap.is_finite() && d_lap < 1e10);
        assert!(d_agq.is_finite() && d_agq < 1e10);
    }

    #[test]
    fn test_agq_vector_k2_deviance_finite() {
        // m=2 groups, k=2 (intercept + slope), n=4.
        let x = Array2::<f64>::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .unwrap();
        let mut zt_tri = TriMat::new((4, 4));
        zt_tri.add_triplet(0, 0, 1.0);
        zt_tri.add_triplet(0, 1, 1.0);
        zt_tri.add_triplet(1, 0, 0.0);
        zt_tri.add_triplet(1, 1, 1.0);
        zt_tri.add_triplet(2, 2, 1.0);
        zt_tri.add_triplet(2, 3, 1.0);
        zt_tri.add_triplet(3, 2, 0.0);
        zt_tri.add_triplet(3, 3, 1.0);
        let zt = zt_tri.to_csr();

        let y = array![0.0, 1.0, 0.0, 1.0];

        let re_blocks = vec![ReBlock {
            m: 2,
            k: 2,
            theta_len: 3,
            group_name: "G".to_string(),
            effect_names: vec!["(Intercept)".to_string(), "x".to_string()],
            group_map: std::collections::HashMap::new(),
        }];

        let fam = Box::new(BinomialFamily::new());
        let mut glmm = GlmmData::new(x, zt, y, re_blocks, fam, 5);
        let theta = [0.8, 0.0, 0.8];
        let d = glmm.laplace_deviance(&theta, None, 5);
        assert!(d.is_finite() && d < 1e10);
    }

    #[test]
    fn test_agq_joint_two_scalar_blocks_finite() {
        // Two `(1|·)` blocks, disjoint observations: q = 4, identity Z.
        let x = Array2::<f64>::ones((4, 1));
        let mut zt_tri = TriMat::new((4, 4));
        zt_tri.add_triplet(0, 0, 1.0);
        zt_tri.add_triplet(1, 1, 1.0);
        zt_tri.add_triplet(2, 2, 1.0);
        zt_tri.add_triplet(3, 3, 1.0);
        let zt = zt_tri.to_csr();

        let y = array![0.0, 1.0, 0.0, 1.0];

        let re_blocks = vec![
            ReBlock {
                m: 2,
                k: 1,
                theta_len: 1,
                group_name: "A".to_string(),
                effect_names: vec!["(Intercept)".to_string()],
                group_map: std::collections::HashMap::new(),
            },
            ReBlock {
                m: 2,
                k: 1,
                theta_len: 1,
                group_name: "B".to_string(),
                effect_names: vec!["(Intercept)".to_string()],
                group_map: std::collections::HashMap::new(),
            },
        ];

        let fam = Box::new(BinomialFamily::new());
        let mut glmm = GlmmData::new(x, zt, y, re_blocks, fam, 5);
        let theta = [0.8, 0.8];
        let d = glmm.laplace_deviance(&theta, None, 5);
        assert!(d.is_finite() && d < 1e10);
    }

    #[test]
    fn test_pirls_divergence() {
        // Provide nonsensical inputs to force LDLT or Cholesky or PIRLS convergence failures
        let x = Array2::<f64>::ones((2, 2));
        let mut zt_tri = TriMat::new((2, 2));
        zt_tri.add_triplet(0, 0, 1.0);
        zt_tri.add_triplet(1, 1, 1.0);
        let zt = zt_tri.to_csr();

        let y = array![0.0, 1.0];

        // Single RE block
        let re_blocks = vec![ReBlock {
            m: 2,
            k: 1,
            theta_len: 1,
            group_name: "G".to_string(),
            effect_names: vec!["(Intercept)".to_string()],
            group_map: std::collections::HashMap::new(),
        }];

        let fam = Box::new(BinomialFamily::new());
        let mut glmm = GlmmData::new(x, zt, y, re_blocks, fam, 1);

        // Feed in a NaN theta or super large theta to break LDLT or Cholesky
        let dev = glmm.laplace_deviance(&[f64::NAN], None, 1);
        assert_eq!(dev, f64::MAX);

        // Feed extremely large vectors to cause divergence/max iters
        let offset = Array1::from_vec(vec![10.0, -10.0]);
        let dev2 = glmm.laplace_deviance(&[1e100], Some(&offset), 1);
        assert_eq!(dev2, f64::MAX);

        // Also call evaluate to trigger `evaluate` branch wrapper directly
        let eval_res = glmm.evaluate(&[1e100], Some(&offset), 1);
        assert!(eval_res.is_none(), "PIRLS failure should return None");
    }
}
