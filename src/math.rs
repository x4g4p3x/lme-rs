use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::UPLO;
use ndarray_linalg::{Cholesky, Inverse, Solve};
use sprs::{CsMat, TriMat};

/// Represents the resolved analytical outputs from evaluating a fully optimized variance structure.
#[derive(Debug, Clone)]
pub struct ModelCoefficients {
    /// The REML criterion at convergence (-2 log restricted-likelihood).
    pub reml_crit: f64,
    /// The estimated residual variance (σ²).
    pub sigma2: f64,
    /// The estimated fixed-effect coefficients (β).
    pub beta: Array1<f64>,
    /// The conditional modes of the random effects (b).
    pub b: Array1<f64>,
    /// The spherical random effects (u).
    pub u: Array1<f64>,
    /// Standard errors of the fixed-effect coefficients.
    pub beta_se: Array1<f64>,
    /// t-statistics for the fixed-effect coefficients.
    pub beta_t: Array1<f64>,
    /// The fitted conditional values (Xβ + Zb).
    pub fitted: Array1<f64>,
    /// The unscaled conditional residuals (y - Xβ - Zb).
    pub residuals: Array1<f64>,
    /// The Cholesky factor of the unscaled fixed-effects variance matrix (L_x).
    pub l_x: Array2<f64>,
    /// The unscaled variance-covariance matrix of the fixed effects.
    pub v_beta_unscaled: Array2<f64>,
}

/// Encapsulates the core dense/sparse design matrices for Linear Mixed-Effects modeling evaluations.
///
/// Supports optional prior observation weights via `weights`. When provided, all
/// cross-products are pre-computed as weighted versions (X'WX, X'Wy, Z'WZ, Z'Wy).
pub struct LmmData {
    /// Dense fixed-effects design matrix ($X$).
    pub x: Array2<f64>,
    /// Sparse transposed random-effects design matrix ($Z^T$).
    pub zt: CsMat<f64>,
    /// Dependent variable vector ($y$).
    pub y: Array1<f64>,
    /// Collection of random effect dimensional tracking blocks.
    pub re_blocks: Vec<crate::model_matrix::ReBlock>,
    /// Optional prior observation weights (length n).
    pub weights: Option<Array1<f64>>,

    // Cached effective matrices that are independent of theta
    // When weights are present, these are the weighted versions.
    /// Effective Dense fixed-effects design matrix ($X$), optionally scaled by observation weights.
    pub x_eff: Array2<f64>,
    /// Effective Sparse transposed random-effects design matrix ($Z^T$), optionally scaled by observation weights.
    pub zt_eff: CsMat<f64>,
    /// Effective Dependent variable vector ($y$), optionally scaled by observation weights.
    pub y_eff: Array1<f64>,
    /// Cross product of the transposed design matrix ($Z^T Z$). This is `zt_eff * zt_eff^T`.
    pub zt_z: CsMat<f64>,
    /// Cross product of the fixed-effects design matrix ($X^T X$). This is `x_eff^T * x_eff`.
    pub xt_x: Array2<f64>,
    /// Cross product of the fixed-effects design matrix and the dependent variable ($X^T y$). This is `x_eff^T * y_eff`.
    pub xt_y: Array1<f64>,
    /// Precomputed $Z^T X$ (q × p), independent of θ.
    zt_x: Array2<f64>,
    /// Precomputed $Z^T y$ (length q), independent of θ.
    zt_y: Array1<f64>,
    /// Sparse identity matrix (q × q) for the random-effects solve.
    eye_q: CsMat<f64>,
    /// True when every RE block is intercept-only (k = 1); enables diagonal-Λ fast path.
    intercept_only_re: bool,
}

impl LmmData {
    /// Create `LmmData` containing unweighted structural design matrices.
    pub fn new(
        x: Array2<f64>,
        zt: CsMat<f64>,
        y: Array1<f64>,
        re_blocks: Vec<crate::model_matrix::ReBlock>,
    ) -> Self {
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
                let q = zt_w.rows();
                let (zt_x, zt_y) = precompute_zt_products(&zt_w, &x_w, &y_w);
                let eye_q = identity_sparse(q);
                let intercept_only_re = re_blocks.iter().all(|b| b.k == 1);

                LmmData {
                    x,
                    zt,
                    y,
                    re_blocks,
                    weights,
                    x_eff: x_w,
                    zt_eff: zt_w,
                    y_eff: y_w,
                    zt_z,
                    xt_x,
                    xt_y,
                    zt_x,
                    zt_y,
                    eye_q,
                    intercept_only_re,
                }
            }
            None => {
                let zt_z = &zt * &zt.transpose_view();
                let xt_x = x.t().dot(&x);
                let xt_y = x.t().dot(&y);
                let q = zt.rows();
                let (zt_x, zt_y) = precompute_zt_products(&zt, &x, &y);
                let eye_q = identity_sparse(q);
                let intercept_only_re = re_blocks.iter().all(|b| b.k == 1);

                LmmData {
                    x_eff: x.clone(),
                    zt_eff: zt.clone(),
                    y_eff: y.clone(),
                    x,
                    zt,
                    y,
                    re_blocks,
                    weights,
                    zt_z,
                    xt_x,
                    xt_y,
                    zt_x,
                    zt_y,
                    eye_q,
                    intercept_only_re,
                }
            }
        }
    }

    /// Evaluate the profiled REML/ML deviance for a fixed θ (optimizer hot path).
    pub fn log_reml_deviance(&self, theta: &[f64], reml: bool) -> f64 {
        self.profile_deviance(theta, reml)
    }

    /// Computes the complete profiled analytical output for a specific parameter vector `theta`,
    /// including coefficient values (fixed β, random $u$/$b$), deviance limits and scaling components.
    pub fn evaluate(&self, theta: &[f64], reml: bool) -> ModelCoefficients {
        let solved = self.solve_profile(theta, reml);

        let p_usize = self.x.ncols();
        let mut beta_se = Array1::<f64>::zeros(p_usize);
        let mut beta_t = Array1::<f64>::zeros(p_usize);

        let inv_lx = solved.l_x.inv().expect("Inverse of L_x failed");
        let v_beta_unscaled = inv_lx.t().dot(&inv_lx);

        for i in 0..p_usize {
            let var_i = solved.sigma2 * v_beta_unscaled[[i, i]];
            beta_se[i] = var_i.sqrt();
            beta_t[i] = solved.beta[i] / beta_se[i];
        }

        // Fitted Values and Residuals (on original unweighted scale)
        let x_beta = self.x.dot(&solved.beta);
        let n_obs = self.y.len();
        let mut z_b_vec = vec![0.0f64; n_obs];
        for (j, row_vec) in self.zt.outer_iterator().enumerate() {
            for (i, &val) in row_vec.iter() {
                z_b_vec[i] += val * solved.b[j];
            }
        }
        let z_b = Array1::from_vec(z_b_vec);
        let fitted = &x_beta + &z_b;
        let residuals = &self.y - &fitted;

        ModelCoefficients {
            reml_crit: solved.reml_crit,
            sigma2: solved.sigma2,
            beta: solved.beta,
            b: solved.b,
            u: solved.u,
            beta_se,
            beta_t,
            fitted,
            residuals,
            l_x: solved.l_x,
            v_beta_unscaled,
        }
    }

    /// Profiled REML/ML deviance only — skips SEs, fitted values, and matrix inverses.
    fn profile_deviance(&self, theta: &[f64], reml: bool) -> f64 {
        self.solve_profile(theta, reml).reml_crit
    }

    fn solve_profile(&self, theta: &[f64], reml: bool) -> ProfileSolution {
        if self.intercept_only_re {
            self.solve_profile_diagonal(theta, reml)
        } else {
            self.solve_profile_general(theta, reml)
        }
    }

    fn solve_profile_diagonal(&self, theta: &[f64], reml: bool) -> ProfileSolution {
        let n = self.y.len() as f64;
        let p = self.x.ncols() as f64;
        let q = self.zt.rows();
        let p_usize = p as usize;

        let d = theta_diagonal(theta, q, &self.re_blocks);

        let a = build_a_diagonal_scaled(&self.zt_z, &d, &self.eye_q);

        use sprs::SymmetryCheck;
        use sprs_ldl::Ldl;
        let ldl = Ldl::new()
            .check_symmetry(SymmetryCheck::DontCheckSymmetry)
            .numeric(a.view())
            .expect("LDLT of A failed");

        let v_y = &self.zt_y * &d;
        let w_y_vec: Vec<f64> = ldl.solve(v_y.to_vec());
        let w_y = Array1::from_vec(w_y_vec);

        let mut v_cols = Vec::with_capacity(p_usize);
        let mut w_cols = Vec::with_capacity(p_usize);

        for j in 0..p_usize {
            let v_j = &self.zt_x.column(j) * &d;
            let w_j_vec: Vec<f64> = ldl.solve(v_j.to_vec());
            let w_j = Array1::from_vec(w_j_vec);
            v_cols.push(v_j);
            w_cols.push(w_j);
        }

        let mut log_det_a = 0.0;
        for &diag in ldl.d() {
            log_det_a += diag.ln();
        }

        solve_profile_finish(
            self,
            ProfileFinishInput {
                reml,
                n,
                p,
                p_usize,
                q,
                log_det_a,
                v_y: &v_y,
                w_y: &w_y,
                v_cols: &v_cols,
                w_cols: &w_cols,
            },
            |u| &d * u,
        )
    }

    fn solve_profile_general(&self, theta: &[f64], reml: bool) -> ProfileSolution {
        let n = self.y.len() as f64;
        let p = self.x.ncols() as f64;
        let q = self.zt.rows();
        let p_usize = p as usize;

        let lambda = build_lambda(theta, q, &self.re_blocks);

        // A = Lambda^T Z^T W Z Lambda + I (using pre-weighted Zt)
        let lam_t = lambda.transpose_view();
        let a_part1 = &lam_t * &self.zt_z;
        let a_part2 = &a_part1 * &lambda;
        let a = &a_part2 + &self.eye_q;

        // LDLT of A
        use sprs::SymmetryCheck;
        use sprs_ldl::Ldl;
        let ldl = Ldl::new()
            .check_symmetry(SymmetryCheck::DontCheckSymmetry)
            .numeric(a.view())
            .expect("LDLT of A failed");

        let v_y = sparse_transpose_matvec(&lambda, self.zt_y.view());
        let w_y_vec: Vec<f64> = ldl.solve(v_y.to_vec());
        let w_y = Array1::from_vec(w_y_vec);

        let mut v_cols = Vec::with_capacity(p_usize);
        let mut w_cols = Vec::with_capacity(p_usize);

        for j in 0..p_usize {
            let v_j = sparse_transpose_matvec(&lambda, self.zt_x.column(j));
            let w_j_vec: Vec<f64> = ldl.solve(v_j.to_vec());
            let w_j = Array1::from_vec(w_j_vec);
            v_cols.push(v_j);
            w_cols.push(w_j);
        }

        let mut log_det_a = 0.0;
        for &diag in ldl.d() {
            log_det_a += diag.ln();
        }

        solve_profile_finish(
            self,
            ProfileFinishInput {
                reml,
                n,
                p,
                p_usize,
                q,
                log_det_a,
                v_y: &v_y,
                w_y: &w_y,
                v_cols: &v_cols,
                w_cols: &w_cols,
            },
            |u| apply_lambda(&lambda, u),
        )
    }
}

struct ProfileFinishInput<'a> {
    reml: bool,
    n: f64,
    p: f64,
    p_usize: usize,
    q: usize,
    log_det_a: f64,
    v_y: &'a Array1<f64>,
    w_y: &'a Array1<f64>,
    v_cols: &'a [Array1<f64>],
    w_cols: &'a [Array1<f64>],
}

fn solve_profile_finish(
    lmm: &LmmData,
    input: ProfileFinishInput<'_>,
    apply_lambda_to_u: impl Fn(&Array1<f64>) -> Array1<f64>,
) -> ProfileSolution {
    let ProfileFinishInput {
        reml,
        n,
        p,
        p_usize,
        q,
        log_det_a,
        v_y,
        w_y,
        v_cols,
        w_cols,
    } = input;
    let mut rzx_t_rzx = Array2::<f64>::zeros((p_usize, p_usize));
    for i in 0..p_usize {
        for j in 0..p_usize {
            rzx_t_rzx[[i, j]] = v_cols[i].dot(&w_cols[j]);
        }
    }

    let mut rzx_t_cu = Array1::<f64>::zeros(p_usize);
    for i in 0..p_usize {
        rzx_t_cu[i] = v_cols[i].dot(w_y);
    }

    let a_x = &lmm.xt_x - &rzx_t_rzx;
    let l_x = a_x.cholesky(UPLO::Lower).expect("Cholesky of A_x failed");

    let rhs_beta = &lmm.xt_y - &rzx_t_cu;

    let c_beta = l_x.solve(&rhs_beta).expect("Solve for c_beta failed");
    let beta = l_x.t().solve(&c_beta).expect("Solve for beta failed");

    let y_norm2: f64 = lmm.y_eff.iter().map(|&x| x * x).sum();

    let mut cu_norm2 = 0.0;
    for i in 0..v_y.len() {
        cu_norm2 += v_y[i] * w_y[i];
    }

    let mut c_beta_norm2 = 0.0;
    for i in 0..beta.len() {
        c_beta_norm2 += beta[i] * rhs_beta[i];
    }

    let r2 = y_norm2 - cu_norm2 - c_beta_norm2;

    let reml_df = if reml { n - p } else { n };
    let sigma2 = r2 / reml_df;

    let mut log_det_l_x = 0.0;
    for i in 0..l_x.nrows() {
        log_det_l_x += l_x[[i, i]].ln();
    }

    let twopi = std::f64::consts::PI * 2.0;
    let mut deviance = reml_df * (twopi * sigma2).ln() + log_det_a + reml_df;

    if reml {
        deviance += 2.0 * log_det_l_x;
    }

    let mut u = Array1::<f64>::zeros(q);
    for i in 0..q {
        let mut w_beta_i = 0.0;
        for j in 0..p_usize {
            w_beta_i += w_cols[j][i] * beta[j];
        }
        u[i] = w_y[i] - w_beta_i;
    }
    let b = apply_lambda_to_u(&u);

    ProfileSolution {
        reml_crit: deviance,
        sigma2,
        beta,
        b,
        u,
        l_x,
    }
}

fn apply_lambda(lambda: &CsMat<f64>, u: &Array1<f64>) -> Array1<f64> {
    let mut b = Array1::<f64>::zeros(lambda.rows());
    for (val, (row, col)) in lambda.iter() {
        b[row] += val * u[col];
    }
    b
}

fn theta_diagonal(
    theta: &[f64],
    q: usize,
    re_blocks: &[crate::model_matrix::ReBlock],
) -> Array1<f64> {
    let mut d = Array1::<f64>::zeros(q);
    let mut row_offset = 0;
    let mut theta_offset = 0;
    for block in re_blocks {
        let t = theta[theta_offset];
        for _ in 0..block.m {
            d[row_offset] = t;
            row_offset += 1;
        }
        theta_offset += block.theta_len;
    }
    d
}

fn build_a_diagonal_scaled(zt_z: &CsMat<f64>, d: &Array1<f64>, eye_q: &CsMat<f64>) -> CsMat<f64> {
    let q = zt_z.rows();
    let mut tri = TriMat::new((q, q));
    for (val, (i, j)) in zt_z.iter() {
        tri.add_triplet(i, j, val * d[i] * d[j]);
    }
    let scaled = tri.to_csr();
    &scaled + eye_q
}

struct ProfileSolution {
    reml_crit: f64,
    sigma2: f64,
    beta: Array1<f64>,
    b: Array1<f64>,
    u: Array1<f64>,
    l_x: Array2<f64>,
}

fn build_lambda(theta: &[f64], q: usize, re_blocks: &[crate::model_matrix::ReBlock]) -> CsMat<f64> {
    let mut lam_tri = TriMat::new((q, q));

    let mut row_offset = 0;
    let mut theta_offset = 0;

    for block in re_blocks {
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

fn precompute_zt_products(
    zt: &CsMat<f64>,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let q = zt.rows();
    let p = x.ncols();
    let mut zt_x = Array2::<f64>::zeros((q, p));
    let mut zt_y = Array1::<f64>::zeros(q);

    for (val, (row, col)) in zt.iter() {
        zt_y[row] += val * y[col];
        for j in 0..p {
            zt_x[[row, j]] += val * x[[col, j]];
        }
    }

    (zt_x, zt_y)
}

fn identity_sparse(n: usize) -> CsMat<f64> {
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, 1.0);
    }
    tri.to_csr()
}

/// Multiply Λᵀ by a dense vector (Λ stored in CSR).
fn sparse_transpose_matvec(mat: &CsMat<f64>, x: ArrayView1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(mat.cols());
    for (val, (row, col)) in mat.iter() {
        out[col] += val * x[row];
    }
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_matrix::ReBlock;
    use ndarray::array;
    use sprs::TriMat;

    #[test]
    fn test_weighted_lmm_evaluation() {
        let x = array![[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];

        let mut zt_tri = TriMat::new((3, 3));
        for i in 0..3 {
            zt_tri.add_triplet(i, i, 1.0);
        }
        let zt = zt_tri.to_csr();

        let y = array![2.0, 4.0, 6.0];

        let re_blocks = vec![ReBlock {
            m: 3,
            k: 1,
            theta_len: 1,
            group_name: "G".to_string(),
            effect_names: vec!["(Intercept)".to_string()],
            group_map: std::collections::HashMap::new(),
        }];

        let weights = array![0.5, 1.0, 2.0];

        let lmm_weighted = LmmData::new_weighted(
            x.clone(),
            zt.clone(),
            y.clone(),
            re_blocks.clone(),
            Some(weights),
        );
        let theta = vec![1.0];
        let coefs = lmm_weighted.evaluate(&theta, true);

        assert_eq!(coefs.beta.len(), 2);
        assert_eq!(coefs.b.len(), 3);
        assert!(coefs.sigma2 > -1e-10);
    }
}
