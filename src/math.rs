use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Inverse, Solve};
use ndarray_linalg::UPLO;
use sprs::{CsMat, TriMat};

/// Represents the resolved analytical outputs from evaluating a fully optimized variance structure.
#[derive(Debug, Clone)]
pub struct ModelCoefficients {
    pub reml_crit: f64,
    pub sigma2: f64,
    pub beta: Array1<f64>,
    pub b: Array1<f64>,
    pub u: Array1<f64>,
    pub beta_se: Array1<f64>,
    pub beta_t: Array1<f64>,
    pub fitted: Array1<f64>,
    pub residuals: Array1<f64>,
}

/// Encapsulates the core dense/sparse design matrices for Linear Mixed-Effects modeling evaluations.
pub struct LmmData {
    pub x: Array2<f64>,
    pub zt: CsMat<f64>,
    pub y: Array1<f64>,
    pub re_blocks: Vec<crate::model_matrix::ReBlock>,
    
    // Cached structural matrices that are independent of theta
    pub zt_z: CsMat<f64>,
    pub xt_x: Array2<f64>,
    pub xt_y: Array1<f64>,
}

impl LmmData {
    pub fn new(x: Array2<f64>, zt: CsMat<f64>, y: Array1<f64>, re_blocks: Vec<crate::model_matrix::ReBlock>) -> Self {
        let zt_z = &zt * &zt.transpose_view();
        let xt_x = x.t().dot(&x);
        let xt_y = x.t().dot(&y);
        
        LmmData { x, zt, y, re_blocks, zt_z, xt_x, xt_y }
    }

    pub fn log_reml_deviance(&self, theta: &[f64], reml: bool) -> f64 {
        self.evaluate(theta, reml).reml_crit
    }

    pub fn evaluate(&self, theta: &[f64], reml: bool) -> ModelCoefficients {
        let n = self.y.len() as f64;
        let p = self.x.ncols() as f64;
        let q = self.zt.rows();

        let zt = &self.zt;
        let x = &self.x;
        let y = &self.y;
        
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
        let lambda: CsMat<f64> = lam_tri.to_csr();

        // A = Lambda^T Z^T Z Lambda + I
        let lam_t = lambda.transpose_view();
        let a_part1 = &lam_t * &self.zt_z;
        let a_part2 = &a_part1 * &lambda;

        let mut eye_tri = TriMat::new((q, q));
        for i in 0..q {
            eye_tri.add_triplet(i, i, 1.0);
        }
        let eye: CsMat<f64> = eye_tri.to_csr();
        let a = &a_part2 + &eye;

        // LDLT of A
        use sprs_ldl::Ldl;
        use sprs::SymmetryCheck;
        let ldl = Ldl::new()
            .check_symmetry(SymmetryCheck::DontCheckSymmetry)
            .numeric(a.view())
            .expect("LDLT of A failed");

        // V_y = Lambda^T Z^T y
        let zt_y = zt * y;
        let v_y = &lam_t * &zt_y;
        
        // Use Vec for ldl.solve since sprs_ldl natively requires DenseVector (Vec implements it simply)
        let w_y_vec: Vec<f64> = ldl.solve(v_y.to_vec());
        let w_y = Array1::from_vec(w_y_vec);

        // V = Lambda^T Z^T X
        let p_usize = p as usize;
        let mut v_cols = Vec::with_capacity(p_usize);
        let mut w_cols = Vec::with_capacity(p_usize);
        
        for j in 0..p_usize {
            let x_col = x.column(j).to_owned();
            let zt_x_j = zt * &x_col;
            let v_j = &lam_t * &zt_x_j;
            
            let w_j_vec: Vec<f64> = ldl.solve(v_j.to_vec());
            let w_j = Array1::from_vec(w_j_vec);
            
            v_cols.push(v_j);
            w_cols.push(w_j);
        }

        let mut rzx_t_rzx = Array2::<f64>::zeros((p_usize, p_usize));
        for i in 0..p_usize {
            for j in 0..p_usize {
                let dot = v_cols[i].dot(&w_cols[j]);
                rzx_t_rzx[[i, j]] = dot;
            }
        }

        let mut rzx_t_cu = Array1::<f64>::zeros(p_usize);
        for i in 0..p_usize {
            let dot = v_cols[i].dot(&w_y);
            rzx_t_cu[i] = dot;
        }

        let a_x = &self.xt_x - &rzx_t_rzx;
        let l_x = a_x.cholesky(UPLO::Lower).expect("Cholesky of A_x failed");

        let rhs_beta = &self.xt_y - &rzx_t_cu;
        
        // Solve A_x * beta = rhs_beta using L_x
        // c_beta = L_x^{-1} rhs_beta
        let c_beta = l_x.solve(&rhs_beta).expect("Solve for c_beta failed");
        // beta = L_x^{-T} c_beta
        let beta = l_x.t().solve(&c_beta).expect("Solve for beta failed");

        let y_norm2 = y.dot(y);
        let cu_norm2 = v_y.dot(&w_y);
        let c_beta_norm2: f64 = beta.dot(&rhs_beta);
        
        let r2 = y_norm2 - cu_norm2 - c_beta_norm2;

        let reml_df = if reml { n - p } else { n };
        let sigma2 = r2 / reml_df;

        let mut log_det_a = 0.0;
        for &d in ldl.d() {
            log_det_a += d.ln();
        }
        
        let mut log_det_l_x = 0.0;
        for i in 0..l_x.nrows() {
            log_det_l_x += l_x[[i, i]].ln();
        }

        let twopi = std::f64::consts::PI * 2.0;
        let mut deviance = reml_df * (twopi * sigma2).ln()
            + log_det_a
            + reml_df;
            
        if reml {
            deviance += 2.0 * log_det_l_x;
        }
        
        let reml_crit = deviance;

        let mut u = Array1::<f64>::zeros(q);
        for i in 0..q {
            let mut w_beta_i = 0.0;
            for j in 0..p_usize {
                w_beta_i += w_cols[j][i] * beta[j];
            }
            u[i] = w_y[i] - w_beta_i;
        }
        
        let b = &lambda * &u;

        // 15. Standard Errors for Fixed Effects
        // Variance-Covariance Matrix V(beta) = sigma2 * A_x^{-1}
        // Since A_x = L_x L_x^T, A_x^{-1} = L_x^{-T} L_x^{-1}
        let mut beta_se = Array1::<f64>::zeros(p_usize);
        let mut beta_t = Array1::<f64>::zeros(p_usize);
        
        let inv_lx = l_x.inv().expect("Inverse of L_x failed");
        let v_beta_unscaled = inv_lx.t().dot(&inv_lx);
        
        for i in 0..p_usize {
            let var_i = sigma2 * v_beta_unscaled[[i, i]];
            beta_se[i] = var_i.sqrt();
            beta_t[i] = beta[i] / beta_se[i];
        }

        // 16. Compute Fitted Values and Residuals
        // fitted = X*beta + Z*b where Z = zt^T (zt is q×n, so Z is n×q)
        let x_beta = x.dot(&beta);
        // Compute Z*b by iterating over zt (CSR, q×n): Z*b = zt^T * b
        // For each row j of zt (random effect j), and each entry zt[j, i], contribute b[j] * zt[j, i] to result[i]
        let n_obs = self.y.len();
        let mut z_b_vec = vec![0.0f64; n_obs];
        for (j, row_vec) in zt.outer_iterator().enumerate() {
            for (i, &val) in row_vec.iter() {
                z_b_vec[i] += val * b[j];
            }
        }
        let z_b = Array1::from_vec(z_b_vec);
        let fitted = &x_beta + &z_b;
        let residuals = y - &fitted;

        ModelCoefficients {
            reml_crit,
            sigma2,
            beta,
            b,
            u,
            beta_se,
            beta_t,
            fitted,
            residuals,
        }
    }
}
