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
///
/// Supports optional prior observation weights via `weights`. When provided, all
/// cross-products are pre-computed as weighted versions (X'WX, X'Wy, Z'WZ, Z'Wy).
pub struct LmmData {
    pub x: Array2<f64>,
    pub zt: CsMat<f64>,
    pub y: Array1<f64>,
    pub re_blocks: Vec<crate::model_matrix::ReBlock>,
    /// Optional prior observation weights (length n).
    pub weights: Option<Array1<f64>>,
    
    // Cached structural matrices that are independent of theta
    // When weights are present, these are the weighted versions.
    pub zt_z: CsMat<f64>,
    pub xt_x: Array2<f64>,
    pub xt_y: Array1<f64>,
}

impl LmmData {
    pub fn new(x: Array2<f64>, zt: CsMat<f64>, y: Array1<f64>, re_blocks: Vec<crate::model_matrix::ReBlock>) -> Self {
        Self::new_weighted(x, zt, y, re_blocks, None)
    }

    /// Create LmmData with optional observation weights.
    ///
    /// When `weights` is Some, cross-products are computed as X'WX, X'Wy, Z'WZ, Z'Wy
    /// where W = diag(weights). This is equivalent to scaling rows by sqrt(w).
    pub fn new_weighted(
        x: Array2<f64>,
        zt: CsMat<f64>,
        y: Array1<f64>,
        re_blocks: Vec<crate::model_matrix::ReBlock>,
        weights: Option<Array1<f64>>,
    ) -> Self {
        match &weights {
            Some(w) => {
                // Weighted cross-products: scale X and y by sqrt(w)
                let sqrt_w = w.mapv(|wi| wi.sqrt());
                let n = x.nrows();
                let p = x.ncols();
                
                // X_w = diag(sqrt_w) * X
                let mut x_w = Array2::<f64>::zeros((n, p));
                for i in 0..n {
                    for j in 0..p {
                        x_w[[i, j]] = x[[i, j]] * sqrt_w[i];
                    }
                }
                let y_w = &y * &sqrt_w;
                
                // Z_w^T = Z^T * diag(sqrt_w), which means scaling columns of Zt
                let zt_w = weight_sparse_cols(&zt, &sqrt_w);
                
                let zt_z = &zt_w * &zt_w.transpose_view();
                let xt_x = x_w.t().dot(&x_w);
                let xt_y = x_w.t().dot(&y_w);
                
                LmmData { x, zt, y, re_blocks, weights, zt_z, xt_x, xt_y }
            }
            None => {
                let zt_z = &zt * &zt.transpose_view();
                let xt_x = x.t().dot(&x);
                let xt_y = x.t().dot(&y);
                
                LmmData { x, zt, y, re_blocks, weights, zt_z, xt_x, xt_y }
            }
        }
    }

    pub fn log_reml_deviance(&self, theta: &[f64], reml: bool) -> f64 {
        self.evaluate(theta, reml).reml_crit
    }

    pub fn evaluate(&self, theta: &[f64], reml: bool) -> ModelCoefficients {
        let n = self.y.len() as f64;
        let p = self.x.ncols() as f64;
        let q = self.zt.rows();

        // When weights are present, we work with the weighted versions of Z^T
        let (zt_eff, x_eff, y_eff) = match &self.weights {
            Some(w) => {
                let sqrt_w = w.mapv(|wi| wi.sqrt());
                let n_obs = self.x.nrows();
                let p_cols = self.x.ncols();
                let mut x_w = Array2::<f64>::zeros((n_obs, p_cols));
                for i in 0..n_obs {
                    for j in 0..p_cols {
                        x_w[[i, j]] = self.x[[i, j]] * sqrt_w[i];
                    }
                }
                let y_w = &self.y * &sqrt_w;
                let zt_w = weight_sparse_cols(&self.zt, &sqrt_w);
                (zt_w, x_w, y_w)
            }
            None => (self.zt.clone(), self.x.clone(), self.y.clone()),
        };

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

        // A = Lambda^T Z^T W Z Lambda + I (using pre-weighted Zt)
        let lam_t = lambda.transpose_view();
        let zt_z_local = &zt_eff * &zt_eff.transpose_view();
        let a_part1 = &lam_t * &zt_z_local;
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

        // V_y = Lambda^T Z_w^T y_w
        let zt_y = &zt_eff * &y_eff;
        let v_y = &lam_t * &zt_y;
        
        let w_y_vec: Vec<f64> = ldl.solve(v_y.to_vec());
        let w_y = Array1::from_vec(w_y_vec);

        // V = Lambda^T Z_w^T X_w
        let p_usize = p as usize;
        let mut v_cols = Vec::with_capacity(p_usize);
        let mut w_cols = Vec::with_capacity(p_usize);
        
        for j in 0..p_usize {
            let x_col = x_eff.column(j).to_owned();
            let zt_x_j = &zt_eff * &x_col;
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
        
        let c_beta = l_x.solve(&rhs_beta).expect("Solve for c_beta failed");
        let beta = l_x.t().solve(&c_beta).expect("Solve for beta failed");

        let y_norm2 = y_eff.dot(&y_eff);
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

        // Standard Errors for Fixed Effects
        let mut beta_se = Array1::<f64>::zeros(p_usize);
        let mut beta_t = Array1::<f64>::zeros(p_usize);
        
        let inv_lx = l_x.inv().expect("Inverse of L_x failed");
        let v_beta_unscaled = inv_lx.t().dot(&inv_lx);
        
        for i in 0..p_usize {
            let var_i = sigma2 * v_beta_unscaled[[i, i]];
            beta_se[i] = var_i.sqrt();
            beta_t[i] = beta[i] / beta_se[i];
        }

        // Fitted Values and Residuals (on original unweighted scale)
        let x_beta = self.x.dot(&beta);
        let n_obs = self.y.len();
        let mut z_b_vec = vec![0.0f64; n_obs];
        for (j, row_vec) in self.zt.outer_iterator().enumerate() {
            for (i, &val) in row_vec.iter() {
                z_b_vec[i] += val * b[j];
            }
        }
        let z_b = Array1::from_vec(z_b_vec);
        let fitted = &x_beta + &z_b;
        let residuals = &self.y - &fitted;

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

/// Multiply each column i of a sparse CSR matrix (q×n) by scale[i].
fn weight_sparse_cols(sp: &CsMat<f64>, scale: &Array1<f64>) -> CsMat<f64> {
    let (rows, cols) = sp.shape();
    let mut tri = TriMat::new((rows, cols));
    for (row_idx, row_vec) in sp.outer_iterator().enumerate() {
        for (col_idx, &val) in row_vec.iter() {
            tri.add_triplet(row_idx, col_idx, val * scale[col_idx]);
        }
    }
    tri.to_csr()
}

