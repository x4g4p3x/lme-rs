use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::UPLO;
use ndarray_linalg::{Cholesky, Inverse, Solve};
use sprs::{CsMat, TriMat};
use std::collections::HashMap;
use std::sync::Mutex;

#[path = "intercept_blocked.rs"]
mod intercept_blocked;

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
    /// Reused symbolic/numeric LDLT for intercept-only A = diag(θ) Z^T Z diag(θ) + I.
    intercept_ldl: Option<Mutex<InterceptLdlCache>>,
    /// Block-diagonal ΛᵀZᵀZΛ fast path for one grouping factor with k > 1 (random slopes).
    single_factor_slopes: Option<Mutex<SingleFactorSlopesCache>>,
    /// Cached ‖y_eff‖² for profile deviance (independent of θ).
    y_norm2: f64,
}

impl LmmData {
    /// True when every random-effects block is intercept-only (`k = 1`).
    pub fn intercept_only_re(&self) -> bool {
        self.intercept_only_re
    }

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

                let zt_z = if re_blocks.iter().all(|b| b.k == 1)
                    && should_build_zt_z_intercept_fast(&zt_w)
                {
                    build_zt_z_intercept_from_zt(&zt_w)
                } else {
                    (&zt_w * &zt_w.transpose_view()).to_csr()
                };
                let xt_x = x_w.t().dot(&x_w);
                let xt_y = x_w.t().dot(&y_w);
                let q = zt_w.rows();
                let (zt_x, zt_y) = precompute_zt_products(&zt_w, &x_w, &y_w);
                let eye_q = identity_sparse(q);
                let intercept_only_re = re_blocks.iter().all(|b| b.k == 1);
                let y_norm2: f64 = y_w.iter().map(|&xi| xi * xi).sum();

                finish_lmm_data(LmmData {
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
                    intercept_ldl: None,
                    single_factor_slopes: None,
                    y_norm2,
                })
            }
            None => {
                let intercept_only_re = re_blocks.iter().all(|b| b.k == 1);
                let zt_z = if intercept_only_re && should_build_zt_z_intercept_fast(&zt) {
                    build_zt_z_intercept_from_re_blocks(&zt, &re_blocks)
                        .unwrap_or_else(|| build_zt_z_intercept_from_zt(&zt))
                } else {
                    (&zt * &zt.transpose_view()).to_csr()
                };
                let xt_x = x.t().dot(&x);
                let xt_y = x.t().dot(&y);
                let q = zt.rows();
                let (zt_x, zt_y) = precompute_zt_products(&zt, &x, &y);
                let eye_q = identity_sparse(q);
                let y_norm2: f64 = y.iter().map(|&xi| xi * xi).sum();

                finish_lmm_data(LmmData {
                    x,
                    zt,
                    y,
                    re_blocks,
                    weights,
                    // Unweighted: cross-products use x/zt/y directly; no duplicate storage.
                    x_eff: Array2::zeros((0, 0)),
                    zt_eff: CsMat::zero((0, 0)),
                    y_eff: Array1::zeros(0),
                    zt_z,
                    xt_x,
                    xt_y,
                    zt_x,
                    zt_y,
                    eye_q,
                    intercept_only_re,
                    intercept_ldl: None,
                    single_factor_slopes: None,
                    y_norm2,
                })
            }
        }
    }

    /// True when a single-factor random-slopes block solver is active (`k > 1`, one RE term).
    pub fn single_factor_slopes_re(&self) -> bool {
        self.single_factor_slopes.is_some()
    }

    /// True when intercept-only blocked Cholesky is available for this model.
    pub fn blocked_kernel_available(&self) -> bool {
        self.intercept_ldl.as_ref().is_some_and(|cache| {
            cache
                .lock()
                .expect("intercept LDL lock poisoned")
                .blocked_gate
        })
    }

    /// Perf-diag label for blocked kernel availability.
    pub fn blocked_kernel_detail(&self) -> &'static str {
        if self.blocked_kernel_available() {
            "blocked_active"
        } else {
            intercept_blocked::blocked_unavailable_reason(self)
        }
    }

    /// Evaluate the profiled REML/ML deviance for a fixed θ (optimizer hot path).
    pub fn log_reml_deviance(&self, theta: &[f64], reml: bool) -> f64 {
        crate::perf_diag::inc_deviance_eval();
        crate::perf_diag::scope(crate::perf_diag::Phase::DevianceEval, || {
            self.profile_deviance(theta, reml)
        })
    }
}

fn finish_lmm_data(mut data: LmmData) -> LmmData {
    if data.intercept_only_re {
        let cache = InterceptLdlCache::new_from_lmm(&data).expect("intercept LDL setup failed");
        data.intercept_ldl = Some(Mutex::new(cache));
    } else if let Some(cache) = SingleFactorSlopesCache::try_new_from_lmm(&data) {
        data.single_factor_slopes = Some(Mutex::new(cache));
    }
    data
}

impl LmmData {
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
        if self.intercept_only_re {
            self.profile_deviance_diagonal(theta, reml)
        } else if let Some(cache_mutex) = &self.single_factor_slopes {
            let mut cache = cache_mutex
                .lock()
                .expect("single-factor slopes lock poisoned");
            cache.profile_deviance(self, theta, reml)
        } else {
            self.solve_profile_general(theta, reml).reml_crit
        }
    }

    fn solve_profile(&self, theta: &[f64], reml: bool) -> ProfileSolution {
        if self.intercept_only_re {
            self.solve_profile_diagonal(theta, reml)
        } else if let Some(cache_mutex) = &self.single_factor_slopes {
            let mut cache = cache_mutex
                .lock()
                .expect("single-factor slopes lock poisoned");
            cache.solve_profile(self, theta, reml)
        } else {
            self.solve_profile_general(theta, reml)
        }
    }

    fn solve_profile_diagonal(&self, theta: &[f64], reml: bool) -> ProfileSolution {
        if let Some(cache_mutex) = &self.intercept_ldl {
            if let Ok(mut cache) = cache_mutex.lock() {
                if let Ok(solved) = cache.solve_profile(self, theta, reml) {
                    return solved;
                }
            }
        }
        self.solve_profile_diagonal_fresh(theta, reml)
    }

    fn solve_profile_diagonal_fresh(&self, theta: &[f64], reml: bool) -> ProfileSolution {
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

        let w_col_slices: Vec<&[f64]> = w_cols.iter().map(|c| c.as_slice().unwrap()).collect();
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
                w_y: w_y.as_slice().unwrap(),
                v_cols: &v_cols,
                w_cols: &w_col_slices,
            },
            |u| &d * u,
        )
        .expect("profile finish failed")
    }

    /// Optimizer hot path: deviance without building u, b, or v_cols.
    fn profile_deviance_diagonal(&self, theta: &[f64], reml: bool) -> f64 {
        self.profile_deviance_diagonal_fast(theta, reml)
    }

    fn profile_deviance_diagonal_fast(&self, theta: &[f64], reml: bool) -> f64 {
        let mut cache = self
            .intercept_ldl
            .as_ref()
            .expect("intercept LDL cache missing")
            .lock()
            .expect("intercept LDL lock poisoned");
        cache.profile_deviance(self, theta, reml)
    }

    pub(crate) fn solve_profile_general(&self, theta: &[f64], reml: bool) -> ProfileSolution {
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

        let w_col_slices: Vec<&[f64]> = w_cols.iter().map(|c| c.as_slice().unwrap()).collect();
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
                w_y: w_y.as_slice().unwrap(),
                v_cols: &v_cols,
                w_cols: &w_col_slices,
            },
            |u| apply_lambda(&lambda, u),
        )
        .expect("profile finish failed")
    }
}

pub(crate) struct ProfileFinishInput<'a> {
    reml: bool,
    n: f64,
    p: f64,
    p_usize: usize,
    q: usize,
    log_det_a: f64,
    v_y: &'a Array1<f64>,
    w_y: &'a [f64],
    v_cols: &'a [Array1<f64>],
    w_cols: &'a [&'a [f64]],
}

struct ProfileDevianceBlocksInput<'a> {
    reml: bool,
    n: f64,
    p: f64,
    p_usize: usize,
    log_det_a: f64,
    theta: &'a [f64],
    row_block: &'a [usize],
    w_y: &'a [f64],
    w_cols: &'a [Vec<f64>],
}

fn compute_profile_deviance_blocks(lmm: &LmmData, input: ProfileDevianceBlocksInput<'_>) -> f64 {
    let ProfileDevianceBlocksInput {
        reml,
        n,
        p,
        p_usize,
        log_det_a,
        theta,
        row_block,
        w_y,
        w_cols,
    } = input;
    let mut rzx_t_rzx = Array2::<f64>::zeros((p_usize, p_usize));
    for i in 0..p_usize {
        for j in 0..p_usize {
            rzx_t_rzx[[i, j]] =
                dot_scaled_block_col(lmm.zt_x.column(i), row_block, theta, w_cols[j].as_slice());
        }
    }

    let mut rzx_t_cu = Array1::<f64>::zeros(p_usize);
    for i in 0..p_usize {
        rzx_t_cu[i] = dot_scaled_block_col(lmm.zt_x.column(i), row_block, theta, w_y);
    }

    let a_x = &lmm.xt_x - &rzx_t_rzx;
    let l_x = match a_x.cholesky(UPLO::Lower) {
        Ok(f) => f,
        Err(_) => return f64::MAX,
    };

    let rhs_beta = &lmm.xt_y - &rzx_t_cu;
    let c_beta = l_x.solve(&rhs_beta).expect("Solve for c_beta failed");
    let beta = l_x.t().solve(&c_beta).expect("Solve for beta failed");

    let y_norm2 = lmm.y_norm2;
    let cu_norm2 = dot_scaled_block_col(lmm.zt_y.view(), row_block, theta, w_y);

    let mut c_beta_norm2 = 0.0;
    for i in 0..beta.len() {
        c_beta_norm2 += beta[i] * rhs_beta[i];
    }

    let r2 = y_norm2 - cu_norm2 - c_beta_norm2;
    let reml_df = if reml { n - p } else { n };
    let sigma2 = r2 / reml_df;
    if sigma2 <= 0.0 {
        return f64::MAX;
    }

    let twopi = std::f64::consts::PI * 2.0;
    let mut deviance = reml_df * (twopi * sigma2).ln() + log_det_a + reml_df;

    if reml {
        let mut log_det_l_x = 0.0;
        for i in 0..l_x.nrows() {
            log_det_l_x += l_x[[i, i]].ln();
        }
        deviance += 2.0 * log_det_l_x;
    }

    deviance
}

#[inline]
fn dot_scaled_block_col(
    zt_col: ArrayView1<f64>,
    row_block: &[usize],
    theta: &[f64],
    w: &[f64],
) -> f64 {
    let mut s = 0.0;
    for k in 0..row_block.len() {
        s += zt_col[k] * theta[row_block[k]] * w[k];
    }
    s
}

fn build_row_blocks(re_blocks: &[crate::model_matrix::ReBlock]) -> Vec<usize> {
    let mut row_block = Vec::new();
    for (block_idx, block) in re_blocks.iter().enumerate() {
        let n_rows = block.m * block.k;
        row_block.extend(std::iter::repeat_n(block_idx, n_rows));
    }
    row_block
}

pub(crate) fn solve_profile_finish(
    lmm: &LmmData,
    input: ProfileFinishInput<'_>,
    apply_lambda_to_u: impl Fn(&Array1<f64>) -> Array1<f64>,
) -> Result<ProfileSolution, ()> {
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
            rzx_t_rzx[[i, j]] = v_cols[i].dot(&ArrayView1::from(w_cols[j]));
        }
    }

    let mut rzx_t_cu = Array1::<f64>::zeros(p_usize);
    for i in 0..p_usize {
        rzx_t_cu[i] = v_cols[i].dot(&ArrayView1::from(w_y));
    }

    let a_x = &lmm.xt_x - &rzx_t_rzx;
    let l_x = a_x.cholesky(UPLO::Lower).map_err(|_| ())?;

    let rhs_beta = &lmm.xt_y - &rzx_t_cu;

    let c_beta = l_x.solve(&rhs_beta).map_err(|_| ())?;
    let beta = l_x.t().solve(&c_beta).map_err(|_| ())?;

    let y_norm2 = lmm.y_norm2;

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

    Ok(ProfileSolution {
        reml_crit: deviance,
        sigma2,
        beta,
        b,
        u,
        l_x,
    })
}

fn apply_lambda(lambda: &CsMat<f64>, u: &Array1<f64>) -> Array1<f64> {
    let mut b = Array1::<f64>::zeros(lambda.rows());
    for (val, (row, col)) in lambda.iter() {
        b[row] += val * u[col];
    }
    b
}

pub(crate) fn theta_diagonal(
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

pub(crate) struct ProfileSolution {
    reml_crit: f64,
    sigma2: f64,
    beta: Array1<f64>,
    b: Array1<f64>,
    u: Array1<f64>,
    l_x: Array2<f64>,
}

/// Block-diagonal ΛᵀZᵀZΛ solver for one grouping factor with correlated random effects (`k > 1`).
///
/// When `ZᵀZ` is block-diagonal across groups (single RE term), each `k × k` block is factored
/// independently instead of a full `q × q` sparse LDL rebuild on every θ evaluation.
struct SingleFactorSlopesCache {
    k: usize,
    m: usize,
    q: usize,
    p_usize: usize,
    s_blocks: Vec<Array2<f64>>,
    lambda: CsMat<f64>,
    w_y_buf: Vec<f64>,
    v_y_buf: Vec<f64>,
    w_col_bufs: Vec<Vec<f64>>,
    v_col_bufs: Vec<Vec<f64>>,
    a_block: Array2<f64>,
    sl_scratch: Array2<f64>,
    a_indptr: Vec<usize>,
    a_indices: Vec<usize>,
    a_values: Vec<f64>,
    full_ldl: Option<sprs_ldl::LdlNumeric<f64, usize>>,
}

impl SingleFactorSlopesCache {
    fn try_new_from_lmm(lmm: &LmmData) -> Option<Self> {
        if lmm.re_blocks.len() != 1 {
            return None;
        }
        let block = &lmm.re_blocks[0];
        let k = block.k;
        if k <= 1 {
            return None;
        }
        let m = block.m;
        let q = m * k;
        if lmm.zt_z.rows() != q || !zt_z_is_block_diagonal(&lmm.zt_z, k, m) {
            return None;
        }

        let mut s_blocks = vec![Array2::<f64>::zeros((k, k)); m];
        for (val, (i, j)) in lmm.zt_z.iter() {
            let g = i / k;
            debug_assert_eq!(g, j / k, "block-diagonal gate must hold");
            s_blocks[g][[i % k, j % k]] += *val;
        }
        for s in &mut s_blocks {
            for i in 0..k {
                for j in (i + 1)..k {
                    let sym = 0.5 * (s[[i, j]] + s[[j, i]]);
                    s[[i, j]] = sym;
                    s[[j, i]] = sym;
                }
            }
        }

        let p_usize = lmm.x.ncols();
        let mut w_col_bufs = Vec::with_capacity(p_usize);
        let mut v_col_bufs = Vec::with_capacity(p_usize);
        for _ in 0..p_usize {
            w_col_bufs.push(vec![0.0; q]);
            v_col_bufs.push(vec![0.0; q]);
        }

        let mut a_indptr = Vec::with_capacity(q + 1);
        let mut a_indices = Vec::with_capacity(m * k * k);
        a_indptr.push(0);
        for r in 0..q {
            let o = (r / k) * k;
            for j in 0..k {
                a_indices.push(o + j);
            }
            a_indptr.push(a_indices.len());
        }
        let a_values = vec![0.0; m * k * k];

        Some(Self {
            k,
            m,
            q,
            p_usize,
            s_blocks,
            lambda: build_lambda(&vec![1.0; block.theta_len], q, &lmm.re_blocks),
            w_y_buf: vec![0.0; q],
            v_y_buf: vec![0.0; q],
            w_col_bufs,
            v_col_bufs,
            a_block: Array2::zeros((k, k)),
            sl_scratch: Array2::zeros((k, k)),
            a_indptr,
            a_indices,
            a_values,
            full_ldl: None,
        })
    }

    fn profile_deviance(&mut self, lmm: &LmmData, theta: &[f64], reml: bool) -> f64 {
        match self.factor_and_collect(lmm, theta) {
            Ok(log_det_a) => self.deviance_from_workspaces(lmm, reml, log_det_a),
            Err(_) => f64::MAX,
        }
    }

    fn deviance_from_workspaces(&self, lmm: &LmmData, reml: bool, log_det_a: f64) -> f64 {
        let n = lmm.y.len() as f64;
        let p = lmm.x.ncols() as f64;
        let p_usize = self.p_usize;

        let mut rzx_t_rzx = Array2::<f64>::zeros((p_usize, p_usize));
        for i in 0..p_usize {
            for j in 0..p_usize {
                rzx_t_rzx[[i, j]] = self.v_col_bufs[i]
                    .iter()
                    .zip(self.w_col_bufs[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }

        let mut rzx_t_cu = Array1::<f64>::zeros(p_usize);
        for i in 0..p_usize {
            rzx_t_cu[i] = self.v_col_bufs[i]
                .iter()
                .zip(self.w_y_buf.iter())
                .map(|(a, b)| a * b)
                .sum();
        }

        let a_x = &lmm.xt_x - &rzx_t_rzx;
        let l_x = match a_x.cholesky(UPLO::Lower) {
            Ok(f) => f,
            Err(_) => return f64::MAX,
        };

        let rhs_beta = &lmm.xt_y - &rzx_t_cu;
        let c_beta = match l_x.solve(&rhs_beta) {
            Ok(v) => v,
            Err(_) => return f64::MAX,
        };
        let beta = match l_x.t().solve(&c_beta) {
            Ok(v) => v,
            Err(_) => return f64::MAX,
        };

        let mut cu_norm2 = 0.0;
        for (vy, wy) in self.v_y_buf.iter().zip(self.w_y_buf.iter()) {
            cu_norm2 += vy * wy;
        }

        let mut c_beta_norm2 = 0.0;
        for i in 0..beta.len() {
            c_beta_norm2 += beta[i] * rhs_beta[i];
        }

        let r2 = lmm.y_norm2 - cu_norm2 - c_beta_norm2;
        let reml_df = if reml { n - p } else { n };
        let sigma2 = r2 / reml_df;
        if sigma2 <= 0.0 {
            return f64::MAX;
        }

        let twopi = std::f64::consts::PI * 2.0;
        let mut deviance = reml_df * (twopi * sigma2).ln() + log_det_a + reml_df;

        if reml {
            let mut log_det_l_x = 0.0;
            for i in 0..l_x.nrows() {
                log_det_l_x += l_x[[i, i]].ln();
            }
            deviance += 2.0 * log_det_l_x;
        }

        deviance
    }

    fn solve_profile(&mut self, lmm: &LmmData, theta: &[f64], reml: bool) -> ProfileSolution {
        let log_det_a = self
            .factor_and_collect(lmm, theta)
            .expect("single-factor slopes factorization failed");
        let n = lmm.y.len() as f64;
        let p = lmm.x.ncols() as f64;
        let p_usize = self.p_usize;
        let q = self.m * self.k;
        self.lambda = build_lambda(theta, q, &lmm.re_blocks);
        let v_y = Array1::from_vec(self.v_y_buf.clone());
        let v_cols: Vec<Array1<f64>> = self
            .v_col_bufs
            .iter()
            .map(|v| Array1::from_vec(v.clone()))
            .collect();
        let w_col_slices: Vec<&[f64]> = self.w_col_bufs.iter().map(|v| v.as_slice()).collect();
        solve_profile_finish(
            lmm,
            ProfileFinishInput {
                reml,
                n,
                p,
                p_usize,
                q,
                log_det_a,
                v_y: &v_y,
                w_y: &self.w_y_buf,
                v_cols: &v_cols,
                w_cols: &w_col_slices,
            },
            |u| apply_lambda(&self.lambda, u),
        )
        .expect("single-factor slopes profile finish failed")
    }

    fn factor_and_collect(&mut self, lmm: &LmmData, theta: &[f64]) -> Result<f64, ()> {
        let k = self.k;
        let q = self.q;
        let l = build_lambda_block(theta, k);
        for g in 0..self.m {
            build_a_block(
                &l,
                &self.s_blocks[g],
                &mut self.sl_scratch,
                &mut self.a_block,
            );
            for i in 0..k {
                let base = self.a_indptr[g * k + i];
                for j in 0..k {
                    self.a_values[base + j] = self.a_block[[i, j]];
                }
            }
        }

        use sprs::{CsMatView, SymmetryCheck};
        let a = CsMatView::new((q, q), &self.a_indptr, &self.a_indices, &self.a_values);
        match &mut self.full_ldl {
            Some(ldl) => ldl.update(a).map_err(|_| ())?,
            slot @ None => {
                *slot = Some(
                    sprs_ldl::Ldl::new()
                        .check_symmetry(SymmetryCheck::DontCheckSymmetry)
                        .numeric(a)
                        .map_err(|_| ())?,
                );
            }
        }
        let ldl = self.full_ldl.as_ref().expect("full LDL");
        let mut log_det_a = 0.0;
        for &diag in ldl.d() {
            log_det_a += diag.ln();
        }

        self.lambda = build_lambda(theta, q, &lmm.re_blocks);
        let v_y = sparse_transpose_matvec(&self.lambda, lmm.zt_y.view());
        self.v_y_buf.copy_from_slice(v_y.as_slice().unwrap());
        let w_y = ldl.solve(v_y.to_vec());
        self.w_y_buf.copy_from_slice(&w_y);

        for j in 0..self.p_usize {
            let v_j = sparse_transpose_matvec(&self.lambda, lmm.zt_x.column(j));
            self.v_col_bufs[j].copy_from_slice(v_j.as_slice().unwrap());
            let w_j = ldl.solve(v_j.to_vec());
            self.w_col_bufs[j].copy_from_slice(&w_j);
        }

        Ok(log_det_a)
    }
}

fn zt_z_is_block_diagonal(zt_z: &CsMat<f64>, k: usize, m: usize) -> bool {
    let q = m * k;
    if zt_z.rows() != q || zt_z.cols() != q {
        return false;
    }
    for (val, (i, j)) in zt_z.iter() {
        if i / k != j / k && val.abs() > 1e-15 {
            return false;
        }
    }
    true
}

fn build_lambda_block(theta: &[f64], k: usize) -> Array2<f64> {
    let mut l = Array2::<f64>::zeros((k, k));
    let mut idx = 0;
    for j in 0..k {
        for i in j..k {
            l[[i, j]] = theta[idx];
            idx += 1;
        }
    }
    l
}

fn build_a_block(l: &Array2<f64>, s: &Array2<f64>, sl: &mut Array2<f64>, a: &mut Array2<f64>) {
    let k = l.nrows();
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for p in 0..k {
                sum += s[[i, p]] * l[[p, j]];
            }
            sl[[i, j]] = sum;
        }
    }
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for p in 0..k {
                sum += l[[p, i]] * sl[[p, j]];
            }
            a[[i, j]] = sum;
        }
        a[[i, i]] += 1.0;
    }
}

/// Use dense Cholesky when q is modest (crossed 20k → q=350); sparse LDL for larger q.
const INTERCEPT_DENSE_MAX_Q: usize = 0;

/// Cached LDLT for intercept-only models: numeric values updated each θ, symbolic part fixed.
struct InterceptSparseLdl {
    q: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    a_values: Vec<f64>,
    zt_z_coeff: Vec<f64>,
    nz_rows: Vec<usize>,
    nz_cols: Vec<usize>,
    /// `row_block[nz_rows[k]]` — avoids repeated indirect indexing in `factor_blocks`.
    nz_theta_i: Vec<usize>,
    /// `row_block[nz_cols[k]]`
    nz_theta_j: Vec<usize>,
    ldl: sprs_ldl::LdlNumeric<f64, usize>,
    solve_buf: Vec<f64>,
}

impl InterceptSparseLdl {
    fn new(
        zt_z: &CsMat<f64>,
        q: usize,
        row_block: &[usize],
    ) -> Result<Self, sprs::errors::LinalgError> {
        let (indptr, indices, a_values, nz_rows, nz_cols, zt_z_coeff) =
            build_intercept_a_template(zt_z, q);
        let mut nz_theta_i = Vec::with_capacity(nz_rows.len());
        let mut nz_theta_j = Vec::with_capacity(nz_cols.len());
        for k in 0..nz_rows.len() {
            nz_theta_i.push(row_block[nz_rows[k]]);
            nz_theta_j.push(row_block[nz_cols[k]]);
        }
        let a = CsMat::new((q, q), indptr.clone(), indices.clone(), a_values.clone());
        use sprs::SymmetryCheck;
        let ldl = sprs_ldl::Ldl::new()
            .check_symmetry(SymmetryCheck::DontCheckSymmetry)
            .numeric(a.view())?;
        Ok(Self {
            q,
            indptr,
            indices,
            a_values,
            zt_z_coeff,
            nz_rows,
            nz_cols,
            nz_theta_i,
            nz_theta_j,
            ldl,
            solve_buf: vec![0.0; q],
        })
    }

    fn factor_blocks(
        &mut self,
        theta: &[f64],
        _row_block: &[usize],
    ) -> Result<(), sprs::errors::LinalgError> {
        for k in 0..self.a_values.len() {
            let ti = self.nz_theta_i[k];
            let tj = self.nz_theta_j[k];
            let mut val = self.zt_z_coeff[k] * theta[ti] * theta[tj];
            if self.nz_rows[k] == self.nz_cols[k] {
                val += 1.0;
            }
            self.a_values[k] = val;
        }
        self.update_ldl()
    }

    fn update_ldl(&mut self) -> Result<(), sprs::errors::LinalgError> {
        use sprs::CsMatView;
        let a = CsMatView::new(
            (self.q, self.q),
            &self.indptr,
            &self.indices,
            &self.a_values,
        );
        self.ldl.update(a)
    }

    fn solve_into(&mut self, rhs: &[f64], out: &mut [f64]) {
        debug_assert_eq!(rhs.len(), self.q);
        debug_assert_eq!(out.len(), self.q);
        self.solve_buf.copy_from_slice(rhs);
        let solved = self.ldl.solve(self.solve_buf.as_slice());
        out.copy_from_slice(solved.as_slice());
    }

    fn log_det_a(&self) -> f64 {
        self.ldl.d().iter().map(|x| x.ln()).sum()
    }
}

/// Dense Cholesky path for small q (disabled while INTERCEPT_DENSE_MAX_Q = 0).
#[allow(dead_code)]
struct InterceptDenseChol {
    q: usize,
    p: usize,
    nz_rows: Vec<usize>,
    nz_cols: Vec<usize>,
    zt_z_coeff: Vec<f64>,
    row_block: Vec<usize>,
    a: Array2<f64>,
    chol_l: Array2<f64>,
    w_y: Vec<f64>,
    w_cols: Vec<Vec<f64>>,
    rhs_work: Array1<f64>,
    theta_scale: Vec<f64>,
}

impl InterceptDenseChol {
    fn new(zt_z: &CsMat<f64>, q: usize, p: usize, row_block: &[usize]) -> Self {
        let mut tri = TriMat::new((q, q));
        let mut diag_in_zt = vec![false; q];
        for (val, (i, j)) in zt_z.iter() {
            tri.add_triplet(i, j, *val);
            if i == j {
                diag_in_zt[i] = true;
            }
        }
        for (i, &seen) in diag_in_zt.iter().enumerate().take(q) {
            if !seen {
                tri.add_triplet(i, i, 0.0);
            }
        }
        let zt_template: CsMat<f64> = tri.to_csr();
        let mut nz_rows = Vec::with_capacity(zt_template.nnz());
        let mut nz_cols = Vec::with_capacity(zt_template.nnz());
        let mut zt_z_coeff = Vec::with_capacity(zt_template.nnz());
        for (val, (i, j)) in zt_template.iter() {
            nz_rows.push(i);
            nz_cols.push(j);
            zt_z_coeff.push(*val);
        }

        let mut w_cols = Vec::with_capacity(p);
        for _ in 0..p {
            w_cols.push(vec![0.0; q]);
        }
        Self {
            q,
            p,
            nz_rows,
            nz_cols,
            zt_z_coeff,
            row_block: row_block.to_vec(),
            a: Array2::zeros((q, q)),
            chol_l: Array2::zeros((q, q)),
            w_y: vec![0.0; q],
            w_cols,
            rhs_work: Array1::zeros(q),
            theta_scale: vec![0.0; q],
        }
    }

    fn rebuild_a(&mut self, scale: &[f64]) {
        self.a.fill(0.0);
        for k in 0..self.nz_rows.len() {
            let i = self.nz_rows[k];
            let j = self.nz_cols[k];
            self.a[[i, j]] += self.zt_z_coeff[k] * scale[i] * scale[j];
        }
        for i in 0..self.q {
            self.a[[i, i]] += 1.0;
        }
    }

    fn build_a_from_d(&mut self, d: &Array1<f64>) {
        self.rebuild_a(d.as_slice().unwrap());
    }

    fn build_a(&mut self, theta: &[f64]) {
        for i in 0..self.q {
            self.theta_scale[i] = theta[self.row_block[i]];
        }
        let scale = self.theta_scale.clone();
        self.rebuild_a(&scale);
    }

    fn cholesky_factor(&mut self) -> Result<f64, sprs::errors::LinalgError> {
        let factor = self.a.cholesky(UPLO::Lower).map_err(|_| {
            sprs::errors::LinalgError::SingularMatrix(sprs::errors::SingularMatrixInfo {
                index: 0,
                reason: "intercept dense Cholesky failed",
            })
        })?;
        let mut log_det_a = 0.0;
        for i in 0..self.q {
            log_det_a += factor[[i, i]].ln();
        }
        self.chol_l = factor;
        Ok(2.0 * log_det_a)
    }

    fn factor_from_d(&mut self, d: &Array1<f64>) -> Result<(), sprs::errors::LinalgError> {
        self.build_a_from_d(d);
        self.cholesky_factor()?;
        Ok(())
    }

    fn factor_and_solve(&mut self, lmm: &LmmData, theta: &[f64]) -> Result<f64, ()> {
        self.build_a(theta);
        let log_det_a = self.cholesky_factor().map_err(|_| ())?;

        self.scale_into_work(lmm.zt_y.view(), theta);
        chol_solve_vec(&self.chol_l, &self.rhs_work, &mut self.w_y);

        for j in 0..self.p {
            self.scale_into_work(lmm.zt_x.column(j), theta);
            chol_solve_vec(&self.chol_l, &self.rhs_work, &mut self.w_cols[j]);
        }

        Ok(log_det_a)
    }

    fn scale_into_work(&mut self, zt_vec: ArrayView1<f64>, theta: &[f64]) {
        let rb = &self.row_block;
        for i in 0..self.q {
            self.rhs_work[i] = zt_vec[i] * theta[rb[i]];
        }
    }

    #[allow(dead_code)]
    fn chol_solve_work_into(&self, out: &mut [f64]) {
        chol_solve_vec(&self.chol_l, &self.rhs_work, out);
    }

    fn profile_deviance(&mut self, lmm: &LmmData, theta: &[f64], reml: bool) -> f64 {
        let log_det_a = match self.factor_and_solve(lmm, theta) {
            Ok(d) => d,
            Err(_) => return f64::MAX,
        };
        let n = lmm.y.len() as f64;
        let p = lmm.x.ncols() as f64;
        if self.p == 1 {
            profile_deviance_p1(
                lmm,
                reml,
                n,
                p,
                log_det_a,
                theta,
                &self.row_block,
                &self.w_y,
                &self.w_cols,
            )
        } else if self.p == 2 {
            profile_deviance_p2(
                lmm,
                reml,
                n,
                p,
                log_det_a,
                theta,
                &self.row_block,
                &self.w_y,
                &self.w_cols,
            )
        } else {
            profile_deviance_general_p(
                lmm,
                reml,
                n,
                p,
                self.p,
                log_det_a,
                theta,
                &self.row_block,
                &self.w_y,
                &self.w_cols,
            )
        }
    }
}

fn chol_solve_vec(chol_l: &Array2<f64>, rhs: &Array1<f64>, out: &mut [f64]) {
    let sol = chol_l.solve(rhs).expect("dense Cholesky solve failed");
    out.copy_from_slice(sol.as_slice().unwrap());
}

pub(crate) fn scale_block_rhs_buf(
    scaled_rhs: &mut [f64],
    zt_vec: ArrayView1<f64>,
    theta: &[f64],
    row_block: &[usize],
) {
    for i in 0..row_block.len() {
        scaled_rhs[i] = zt_vec[i] * theta[row_block[i]];
    }
}

/// Intercept-only LDL / Cholesky cache with reusable workspaces.
struct InterceptLdlCache {
    blocked: Option<intercept_blocked::InterceptBlockedChol>,
    blocked_gate: bool,
    dense: Option<InterceptDenseChol>,
    sparse: Option<InterceptSparseLdl>,
    p: usize,
    row_block: Vec<usize>,
    solve_out: Vec<f64>,
    scaled_rhs: Vec<f64>,
    w_y_buf: Vec<f64>,
    w_col_bufs: Vec<Vec<f64>>,
}

impl InterceptLdlCache {
    fn new_from_lmm(lmm: &LmmData) -> Result<Self, sprs::errors::LinalgError> {
        let q = lmm.zt_z.rows();
        let row_block = build_row_blocks(&lmm.re_blocks);
        let p = lmm.x.ncols();
        let blocked_gate = intercept_blocked::blocked_gate_failure(lmm).is_none();
        let sparse = if !blocked_gate {
            Some(InterceptSparseLdl::new(&lmm.zt_z, q, &row_block)?)
        } else {
            None
        };
        #[allow(clippy::absurd_extreme_comparisons)]
        let dense = match INTERCEPT_DENSE_MAX_Q {
            0 => None,
            max_q if q <= max_q => Some(InterceptDenseChol::new(&lmm.zt_z, q, p, &row_block)),
            _ => None,
        };
        let mut w_col_bufs = Vec::with_capacity(p);
        for _ in 0..p {
            w_col_bufs.push(vec![0.0; q]);
        }
        if lmm.intercept_only_re() {
            if blocked_gate {
                crate::perf_diag::set_kernel_detail("blocked_active");
            } else {
                let detail = intercept_blocked::blocked_unavailable_reason(lmm);
                crate::perf_diag::set_kernel_detail(detail);
            }
        }
        Ok(Self {
            blocked: None,
            blocked_gate,
            dense,
            sparse,
            p,
            row_block,
            solve_out: vec![0.0; q],
            scaled_rhs: vec![0.0; q],
            w_y_buf: vec![0.0; q],
            w_col_bufs,
        })
    }

    fn ensure_blocked(&mut self, lmm: &LmmData) {
        if self.blocked_gate && self.blocked.is_none() {
            self.blocked = intercept_blocked::InterceptBlockedChol::try_new(lmm);
        }
    }

    fn ensure_sparse(&mut self, lmm: &LmmData) -> Result<(), sprs::errors::LinalgError> {
        if self.sparse.is_none() {
            let q = lmm.zt_z.rows();
            self.sparse = Some(InterceptSparseLdl::new(&lmm.zt_z, q, &self.row_block)?);
            let q = self.row_block.len();
            if self.solve_out.len() != q {
                self.solve_out.resize(q, 0.0);
                self.scaled_rhs.resize(q, 0.0);
                self.w_y_buf.resize(q, 0.0);
            }
        }
        Ok(())
    }

    fn solve_profile(
        &mut self,
        lmm: &LmmData,
        theta: &[f64],
        reml: bool,
    ) -> Result<ProfileSolution, sprs::errors::LinalgError> {
        self.ensure_blocked(lmm);
        if let Some(blocked) = &mut self.blocked {
            crate::perf_diag::set_kernel("blocked");
            if let Ok(solved) = blocked.solve_profile_blocked(
                lmm,
                theta,
                reml,
                &self.row_block,
                &mut self.w_y_buf,
                &mut self.w_col_bufs,
            ) {
                return Ok(solved);
            }
        }
        self.ensure_sparse(lmm)?;
        let sparse = self
            .sparse
            .as_mut()
            .expect("sparse intercept solver missing");
        sparse.factor_blocks(theta, &self.row_block)?;
        let log_det_a = sparse.log_det_a();

        let q = lmm.zt.rows();
        let p_usize = lmm.x.ncols();
        let n = lmm.y.len() as f64;
        let p = p_usize as f64;
        let d = theta_diagonal(theta, q, &lmm.re_blocks);

        scale_block_rhs_buf(
            &mut self.scaled_rhs,
            lmm.zt_y.view(),
            theta,
            &self.row_block,
        );
        sparse.solve_into(&self.scaled_rhs, &mut self.solve_out);
        self.w_y_buf.copy_from_slice(&self.solve_out);

        let v_y = &lmm.zt_y * &d;
        let mut v_cols = Vec::with_capacity(p_usize);
        for j in 0..p_usize {
            let v_j = &lmm.zt_x.column(j) * &d;
            scale_block_rhs_buf(
                &mut self.scaled_rhs,
                lmm.zt_x.column(j),
                theta,
                &self.row_block,
            );
            sparse.solve_into(&self.scaled_rhs, &mut self.solve_out);
            self.w_col_bufs[j].copy_from_slice(&self.solve_out);
            v_cols.push(v_j);
        }

        let w_col_slices: Vec<&[f64]> = self.w_col_bufs.iter().map(|v| v.as_slice()).collect();
        solve_profile_finish(
            lmm,
            ProfileFinishInput {
                reml,
                n,
                p,
                p_usize,
                q,
                log_det_a,
                v_y: &v_y,
                w_y: &self.w_y_buf,
                v_cols: &v_cols,
                w_cols: &w_col_slices,
            },
            |u| &d * u,
        )
        .map_err(|_| {
            sprs::errors::LinalgError::SingularMatrix(sprs::errors::SingularMatrixInfo {
                index: 0,
                reason: "intercept profile finish failed",
            })
        })
    }

    fn profile_deviance(&mut self, lmm: &LmmData, theta: &[f64], reml: bool) -> f64 {
        self.ensure_blocked(lmm);
        if let Some(blocked) = &mut self.blocked {
            crate::perf_diag::set_kernel("blocked");
            return blocked.profile_deviance(lmm, theta, reml);
        }
        if let Some(dense) = &mut self.dense {
            crate::perf_diag::set_kernel("dense_chol");
            return crate::perf_diag::scope(crate::perf_diag::Phase::DevianceDenseChol, || {
                dense.profile_deviance(lmm, theta, reml)
            });
        }
        crate::perf_diag::set_kernel("sparse_ldl");
        crate::perf_diag::scope(crate::perf_diag::Phase::DevianceSparseLdl, || {
            let InterceptLdlCache {
                sparse,
                p,
                row_block,
                solve_out,
                scaled_rhs,
                w_y_buf,
                w_col_bufs,
                dense: _,
                blocked: _,
                blocked_gate: _,
            } = self;
            let sparse = sparse.as_mut().expect("sparse intercept solver missing");
            sparse
                .factor_blocks(theta, row_block)
                .expect("LDLT update of A failed");
            let log_det_a = sparse.log_det_a();
            let p_usize = lmm.x.ncols();
            scale_block_rhs_buf(scaled_rhs, lmm.zt_y.view(), theta, row_block);
            sparse.solve_into(scaled_rhs, solve_out);
            w_y_buf.copy_from_slice(solve_out);
            for (j, col_buf) in w_col_bufs.iter_mut().enumerate().take(p_usize) {
                scale_block_rhs_buf(scaled_rhs, lmm.zt_x.column(j), theta, row_block);
                sparse.solve_into(scaled_rhs, solve_out);
                col_buf.copy_from_slice(solve_out);
            }
            let n = lmm.y.len() as f64;
            let p_f = lmm.x.ncols() as f64;
            if *p == 1 && p_usize == 1 {
                profile_deviance_p1(
                    lmm, reml, n, p_f, log_det_a, theta, row_block, w_y_buf, w_col_bufs,
                )
            } else if *p == 2 && p_usize == 2 {
                profile_deviance_p2(
                    lmm, reml, n, p_f, log_det_a, theta, row_block, w_y_buf, w_col_bufs,
                )
            } else {
                compute_profile_deviance_blocks(
                    lmm,
                    ProfileDevianceBlocksInput {
                        reml,
                        n,
                        p: p_f,
                        p_usize,
                        log_det_a,
                        theta,
                        row_block,
                        w_y: w_y_buf,
                        w_cols: w_col_bufs,
                    },
                )
            }
        })
    }

    #[allow(dead_code)]
    fn scale_block_rhs(&mut self, zt_vec: ArrayView1<f64>, theta: &[f64]) {
        scale_block_rhs_buf(&mut self.scaled_rhs, zt_vec, theta, &self.row_block);
    }

    #[allow(dead_code)]
    fn factor(&mut self, d: &Array1<f64>) -> Result<(), sprs::errors::LinalgError> {
        if let Some(dense) = &mut self.dense {
            dense.factor_from_d(d)
        } else {
            Err(sprs::errors::LinalgError::SingularMatrix(
                sprs::errors::SingularMatrixInfo {
                    index: 0,
                    reason: "sparse intercept cache has no d-vector factor path",
                },
            ))
        }
    }

    #[allow(dead_code)]
    fn factor_blocks(&mut self, theta: &[f64]) -> Result<(), sprs::errors::LinalgError> {
        if self.dense.is_some() {
            Ok(())
        } else {
            self.sparse
                .as_mut()
                .unwrap()
                .factor_blocks(theta, &self.row_block)
        }
    }

    #[allow(dead_code)]
    fn log_det_a(&self) -> f64 {
        if let Some(dense) = &self.dense {
            let mut log_det = 0.0;
            for i in 0..dense.q {
                log_det += dense.chol_l[[i, i]].ln();
            }
            2.0 * log_det
        } else {
            self.sparse.as_ref().unwrap().log_det_a()
        }
    }

    #[allow(dead_code)]
    fn solve_vec(&mut self, rhs: ArrayView1<f64>) -> Array1<f64> {
        if let Some(dense) = &mut self.dense {
            for i in 0..dense.q {
                dense.rhs_work[i] = rhs[i];
            }
            let sol = dense
                .chol_l
                .solve(&dense.rhs_work)
                .expect("dense solve failed");
            Array1::from_vec(sol.to_vec())
        } else {
            self.sparse
                .as_mut()
                .unwrap()
                .solve_into(rhs.as_slice().unwrap(), &mut self.solve_out);
            Array1::from_vec(self.solve_out.clone())
        }
    }

    #[allow(dead_code)]
    fn solve_scaled_block_vec(&mut self, zt_vec: ArrayView1<f64>, theta: &[f64]) -> Vec<f64> {
        self.scale_block_rhs(zt_vec, theta);
        if let Some(dense) = &self.dense {
            let sol = dense
                .chol_l
                .solve(&Array1::from_vec(self.scaled_rhs.clone()))
                .expect("dense solve failed");
            sol.to_vec()
        } else {
            self.sparse
                .as_mut()
                .unwrap()
                .solve_into(&self.scaled_rhs, &mut self.solve_out);
            self.solve_out.clone()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn profile_deviance_p1(
    lmm: &LmmData,
    reml: bool,
    n: f64,
    p: f64,
    log_det_a: f64,
    theta: &[f64],
    row_block: &[usize],
    w_y: &[f64],
    w_cols: &[Vec<f64>],
) -> f64 {
    let w0 = w_cols[0].as_slice();
    let zt_x = &lmm.zt_x;

    let mut r00 = 0.0;
    let mut c0 = 0.0;
    for k in 0..row_block.len() {
        let t = theta[row_block[k]];
        let zk0 = zt_x[[k, 0]] * t;
        r00 += zk0 * w0[k];
        c0 += zk0 * w_y[k];
    }

    let a_x = lmm.xt_x[[0, 0]] - r00;
    if a_x <= 0.0 {
        return f64::MAX;
    }
    let l00 = a_x.sqrt();
    let rhs0 = lmm.xt_y[0] - c0;
    let beta0 = rhs0 / a_x;

    let mut cu_norm2 = 0.0;
    for k in 0..row_block.len() {
        cu_norm2 += lmm.zt_y[k] * theta[row_block[k]] * w_y[k];
    }

    let c_beta_norm2 = beta0 * rhs0;
    let r2 = lmm.y_norm2 - cu_norm2 - c_beta_norm2;
    let reml_df = if reml { n - p } else { n };
    let sigma2 = r2 / reml_df;
    if sigma2 <= 0.0 {
        return f64::MAX;
    }

    let twopi = std::f64::consts::PI * 2.0;
    let mut deviance = reml_df * (twopi * sigma2).ln() + log_det_a + reml_df;
    if reml {
        deviance += 2.0 * l00.ln();
    }
    deviance
}

/// Hand-unrolled 2×2 SPD solve (avoids LAPACK overhead on the optimizer hot path).
#[inline]
fn try_solve_spd_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<([f64; 2], f64)> {
    if a[0][0] <= 0.0 {
        return None;
    }
    let l00 = a[0][0].sqrt();
    let l10 = a[1][0] / l00;
    let l11_sq = a[1][1] - l10 * l10;
    if l11_sq <= 0.0 {
        return None;
    }
    let l11 = l11_sq.sqrt();
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let x1 = y1 / l11;
    let x0 = (y0 - l10 * x1) / l00;
    let log_det = l00.ln() + l11.ln();
    Some(([x0, x1], log_det))
}

#[allow(clippy::too_many_arguments)]
fn profile_deviance_p2(
    lmm: &LmmData,
    reml: bool,
    n: f64,
    p: f64,
    log_det_a: f64,
    theta: &[f64],
    row_block: &[usize],
    w_y: &[f64],
    w_cols: &[Vec<f64>],
) -> f64 {
    let w0 = w_cols[0].as_slice();
    let w1 = w_cols[1].as_slice();
    let zt_x = &lmm.zt_x;

    let mut r00 = 0.0;
    let mut r01 = 0.0;
    let mut r11 = 0.0;
    let mut c0 = 0.0;
    let mut c1 = 0.0;
    for k in 0..row_block.len() {
        let t = theta[row_block[k]];
        let zk0 = zt_x[[k, 0]] * t;
        let zk1 = zt_x[[k, 1]] * t;
        r00 += zk0 * w0[k];
        r01 += zk0 * w1[k];
        r11 += zk1 * w1[k];
        c0 += zk0 * w_y[k];
        c1 += zk1 * w_y[k];
    }

    let mut a_x = [[lmm.xt_x[[0, 0]] - r00, lmm.xt_x[[0, 1]] - r01], [0.0, 0.0]];
    a_x[1][0] = lmm.xt_x[[1, 0]] - r01;
    a_x[1][1] = lmm.xt_x[[1, 1]] - r11;

    let rhs = [lmm.xt_y[0] - c0, lmm.xt_y[1] - c1];
    let Some((beta, log_det_l_x)) = try_solve_spd_2x2(a_x, rhs) else {
        return f64::MAX;
    };

    let mut cu_norm2 = 0.0;
    for k in 0..row_block.len() {
        cu_norm2 += lmm.zt_y[k] * theta[row_block[k]] * w_y[k];
    }

    let c_beta_norm2 = beta[0] * rhs[0] + beta[1] * rhs[1];
    let r2 = lmm.y_norm2 - cu_norm2 - c_beta_norm2;
    let reml_df = if reml { n - p } else { n };
    let sigma2 = r2 / reml_df;
    if sigma2 <= 0.0 {
        return f64::MAX;
    }

    let twopi = std::f64::consts::PI * 2.0;
    let mut deviance = reml_df * (twopi * sigma2).ln() + log_det_a + reml_df;
    if reml {
        deviance += 2.0 * log_det_l_x;
    }
    deviance
}

#[allow(clippy::too_many_arguments)]
fn profile_deviance_general_p(
    lmm: &LmmData,
    reml: bool,
    n: f64,
    p: f64,
    p_usize: usize,
    log_det_a: f64,
    theta: &[f64],
    row_block: &[usize],
    w_y: &[f64],
    w_cols: &[Vec<f64>],
) -> f64 {
    compute_profile_deviance_blocks(
        lmm,
        ProfileDevianceBlocksInput {
            reml,
            n,
            p,
            p_usize,
            log_det_a,
            theta,
            row_block,
            w_y,
            w_cols,
        },
    )
}

type InterceptATemplate = (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
);

/// For intercept-only RE with many levels, each observation touches few Z columns;
/// accumulate ZᵀZ per observation instead of a full sparse multiply.
fn should_build_zt_z_intercept_fast(zt: &CsMat<f64>) -> bool {
    // Obs-major bucketing allocates O(n) scratch; only pay that for larger q where
    // sparse ZᵀZ dominates prepare (nested/crossed), not tiny random-intercept models.
    zt.rows() >= 256
}

fn build_zt_z_intercept_from_zt(zt: &CsMat<f64>) -> CsMat<f64> {
    let q = zt.rows();
    let n = zt.cols();
    if let Some(diagonal) = build_zt_z_single_membership(zt) {
        return diagonal;
    }
    let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(2); n];
    for (re_idx, row_vec) in zt.outer_iterator().enumerate() {
        for (obs_col, &v) in row_vec.iter() {
            if v != 0.0 {
                buckets[obs_col].push((re_idx, v));
            }
        }
    }
    let mut acc: HashMap<(usize, usize), f64> = HashMap::new();
    for bucket in buckets {
        let k = bucket.len();
        for i in 0..k {
            for j in i..k {
                let (ri, vi) = bucket[i];
                let (rj, vj) = bucket[j];
                let contrib = vi * vj;
                if contrib == 0.0 {
                    continue;
                }
                let (r, c) = if ri >= rj { (ri, rj) } else { (rj, ri) };
                *acc.entry((r, c)).or_insert(0.0) += contrib;
            }
        }
    }
    let mut tri = TriMat::new((q, q));
    for ((r, c), v) in acc {
        if v != 0.0 {
            tri.add_triplet(r, c, v);
            if r != c {
                tri.add_triplet(c, r, v);
            }
        }
    }
    tri.to_csr()
}

/// Direct ZᵀZ construction for up to two unweighted intercept-only factors.
///
/// The fair crossed/nested layouts have exactly one unit-valued row from each
/// factor per observation. Reconstructing that known layout avoids allocating
/// one sparse bucket per observation before accumulating Gram entries.
fn build_zt_z_intercept_from_re_blocks(
    zt: &CsMat<f64>,
    re_blocks: &[crate::model_matrix::ReBlock],
) -> Option<CsMat<f64>> {
    if !(1..=2).contains(&re_blocks.len()) || re_blocks.iter().any(|block| block.k != 1) {
        return None;
    }
    let n = zt.cols();
    let q = zt.rows();
    let mut memberships = vec![vec![usize::MAX; n]; re_blocks.len()];
    let mut offset = 0usize;
    for (block_idx, block) in re_blocks.iter().enumerate() {
        for level in 0..block.m {
            let row = zt.outer_view(offset + level)?;
            for (obs, &value) in row.iter() {
                if value != 1.0 || memberships[block_idx][obs] != usize::MAX {
                    return None;
                }
                memberships[block_idx][obs] = level;
            }
        }
        if memberships[block_idx].contains(&usize::MAX) {
            return None;
        }
        offset += block.m;
    }
    if offset != q {
        return None;
    }

    let mut tri = TriMat::new((q, q));
    let mut offset = 0usize;
    for (block_idx, block) in re_blocks.iter().enumerate() {
        let mut counts = vec![0usize; block.m];
        for &level in &memberships[block_idx] {
            counts[level] += 1;
        }
        for (level, count) in counts.into_iter().enumerate() {
            tri.add_triplet(offset + level, offset + level, count as f64);
        }
        offset += block.m;
    }
    if re_blocks.len() == 2 {
        let second_offset = re_blocks[0].m;
        let mut cross = HashMap::<(usize, usize), usize>::new();
        for (&left, &right) in memberships[0].iter().zip(&memberships[1]) {
            *cross.entry((left, right)).or_default() += 1;
        }
        for ((left, right), count) in cross {
            let right = second_offset + right;
            tri.add_triplet(left, right, count as f64);
            tri.add_triplet(right, left, count as f64);
        }
    }
    Some(tri.to_csr())
}

/// Build ZᵀZ directly when every observation belongs to one intercept-only row.
///
/// A single grouping factor has this shape; avoiding observation buckets and a
/// hash map is particularly important for large random-intercept setup.
fn build_zt_z_single_membership(zt: &CsMat<f64>) -> Option<CsMat<f64>> {
    let n = zt.cols();
    if zt.nnz() != n {
        return None;
    }
    let mut seen = vec![false; n];
    let mut diagonal = Vec::with_capacity(zt.rows());
    for row in zt.outer_iterator() {
        let mut sum = 0.0;
        for (col, &value) in row.iter() {
            if seen[col] {
                return None;
            }
            seen[col] = true;
            sum += value * value;
        }
        diagonal.push(sum);
    }
    if seen.iter().any(|&present| !present) {
        return None;
    }
    let q = zt.rows();
    Some(CsMat::new(
        (q, q),
        (0..=q).collect(),
        (0..q).collect(),
        diagonal,
    ))
}

fn build_intercept_a_template(zt_z: &CsMat<f64>, q: usize) -> InterceptATemplate {
    let mut tri = TriMat::new((q, q));
    let mut diag_in_zt = vec![false; q];
    for (val, (i, j)) in zt_z.iter() {
        tri.add_triplet(i, j, *val);
        if i == j {
            diag_in_zt[i] = true;
        }
    }
    for (i, &seen) in diag_in_zt.iter().enumerate().take(q) {
        if !seen {
            tri.add_triplet(i, i, 0.0);
        }
    }
    let a = tri.to_csr();
    let mut nz_rows = Vec::with_capacity(a.nnz());
    let mut nz_cols = Vec::with_capacity(a.nnz());
    let mut zt_z_coeff = Vec::with_capacity(a.nnz());
    let mut a_values = Vec::with_capacity(a.nnz());
    for (val, (i, j)) in a.iter() {
        nz_rows.push(i);
        nz_cols.push(j);
        zt_z_coeff.push(*val);
        a_values.push(*val + if i == j { 1.0 } else { 0.0 });
    }
    (
        a.indptr().raw_storage().to_vec(),
        a.indices().to_vec(),
        a_values,
        nz_rows,
        nz_cols,
        zt_z_coeff,
    )
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

    match p {
        1 => {
            for (row, row_vec) in zt.outer_iterator().enumerate() {
                let mut zy = 0.0;
                let mut zx0 = 0.0;
                for (col, &val) in row_vec.iter() {
                    zy += val * y[col];
                    zx0 += val * x[[col, 0]];
                }
                zt_y[row] = zy;
                zt_x[[row, 0]] = zx0;
            }
        }
        2 => {
            for (row, row_vec) in zt.outer_iterator().enumerate() {
                let mut zy = 0.0;
                let mut zx0 = 0.0;
                let mut zx1 = 0.0;
                for (col, &val) in row_vec.iter() {
                    zy += val * y[col];
                    zx0 += val * x[[col, 0]];
                    zx1 += val * x[[col, 1]];
                }
                zt_y[row] = zy;
                zt_x[[row, 0]] = zx0;
                zt_x[[row, 1]] = zx1;
            }
        }
        _ => {
            for (row, row_vec) in zt.outer_iterator().enumerate() {
                let mut zy = 0.0;
                for (col, &val) in row_vec.iter() {
                    zy += val * y[col];
                    for j in 0..p {
                        zt_x[[row, j]] += val * x[[col, j]];
                    }
                }
                zt_y[row] = zy;
            }
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
    use std::fs::File;
    use std::io::BufReader;

    #[derive(serde::Deserialize)]
    struct RandomSlopesFixture {
        inputs: RandomSlopesInputs,
        outputs: RandomSlopesOutputs,
    }

    #[derive(serde::Deserialize)]
    struct RandomSlopesInputs {
        #[serde(rename = "X")]
        x: Vec<Vec<f64>>,
        #[serde(rename = "Zt")]
        zt: Vec<Vec<f64>>,
        y: Vec<f64>,
    }

    #[derive(serde::Deserialize)]
    struct RandomSlopesOutputs {
        theta: Vec<f64>,
        reml_crit: f64,
    }

    fn random_slopes_lmm() -> (LmmData, Vec<f64>, f64) {
        let file = File::open("tests/data/random_slopes.json").expect("random_slopes.json");
        let data: RandomSlopesFixture =
            serde_json::from_reader(BufReader::new(file)).expect("parse random_slopes.json");
        let x = Array2::from_shape_vec(
            (data.inputs.x.len(), data.inputs.x[0].len()),
            data.inputs.x.into_iter().flatten().collect(),
        )
        .unwrap();
        let mut zt_tri = TriMat::new((data.inputs.zt.len(), data.inputs.zt[0].len()));
        for (i, row) in data.inputs.zt.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    zt_tri.add_triplet(i, j, val);
                }
            }
        }
        let zt = zt_tri.to_csr();
        let y = Array1::from_vec(data.inputs.y);
        let re_blocks = vec![ReBlock {
            m: 18,
            k: 2,
            theta_len: 3,
            group_name: "Subject".to_string(),
            effect_names: vec!["(Intercept)".to_string(), "Days".to_string()],
            group_map: HashMap::new(),
        }];
        (
            LmmData::new(x, zt, y, re_blocks),
            data.outputs.theta,
            data.outputs.reml_crit,
        )
    }

    #[test]
    fn single_factor_slopes_cache_matches_general_profile() {
        let (lmm, theta, reml_crit) = random_slopes_lmm();
        assert!(lmm.single_factor_slopes_re());
        let general = lmm.solve_profile_general(&theta, true);
        let mut cache = lmm.single_factor_slopes.as_ref().unwrap().lock().unwrap();
        let dev_fast = cache.profile_deviance(&lmm, &theta, true);
        assert!(
            (general.reml_crit - dev_fast).abs() < 1e-6,
            "profile deviance mismatch: general={} fast={} lme4={}",
            general.reml_crit,
            dev_fast,
            reml_crit
        );
        let fast = cache.solve_profile(&lmm, &theta, true);
        assert!((general.reml_crit - fast.reml_crit).abs() < 1e-6);
        for i in 0..general.beta.len() {
            assert!((general.beta[i] - fast.beta[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zt_z_fast_matches_sparse_mul() {
        let mut tri = TriMat::new((7, 12));
        for obs in 0..12 {
            let plate = obs % 3;
            let sample = 3 + (obs % 4);
            tri.add_triplet(plate, obs, 1.0);
            tri.add_triplet(sample, obs, 1.0);
        }
        let zt = tri.to_csr();
        let fast = build_zt_z_intercept_from_zt(&zt);
        let slow = (&zt * &zt.transpose_view()).to_csr();
        assert_eq!(fast.rows(), slow.rows());
        assert_eq!(fast.cols(), slow.cols());
        for (v_ref, (i, j)) in slow.iter() {
            let v_fast = fast.get(i, j).copied().unwrap_or(0.0);
            assert!(
                (v_fast - v_ref).abs() < 1e-12,
                "({i},{j}): fast={v_fast} ref={v_ref}"
            );
        }
        for (v_fast, (i, j)) in fast.iter() {
            let v_ref = slow.get(i, j).copied().unwrap_or(0.0);
            assert!(
                (v_fast - v_ref).abs() < 1e-12,
                "({i},{j}): fast={v_fast} ref={v_ref}"
            );
        }
    }

    #[test]
    fn test_zt_z_direct_two_factor_matches_sparse_mul() {
        let mut tri = TriMat::new((7, 12));
        for obs in 0..12 {
            tri.add_triplet(obs % 3, obs, 1.0);
            tri.add_triplet(3 + (obs % 4), obs, 1.0);
        }
        let zt = tri.to_csr();
        let re_blocks = vec![
            ReBlock {
                m: 3,
                k: 1,
                theta_len: 1,
                group_name: "plate".to_string(),
                effect_names: vec!["(Intercept)".to_string()],
                group_map: HashMap::new(),
            },
            ReBlock {
                m: 4,
                k: 1,
                theta_len: 1,
                group_name: "sample".to_string(),
                effect_names: vec!["(Intercept)".to_string()],
                group_map: HashMap::new(),
            },
        ];
        let fast = build_zt_z_intercept_from_re_blocks(&zt, &re_blocks).expect("direct gram");
        let slow = (&zt * &zt.transpose_view()).to_csr();
        assert_eq!(fast.indptr().raw_storage(), slow.indptr().raw_storage());
        assert_eq!(fast.indices(), slow.indices());
        assert_eq!(fast.data(), slow.data());
    }

    #[test]
    fn test_zt_z_single_membership_matches_sparse_mul() {
        let mut tri = TriMat::new((4, 12));
        for obs in 0..12 {
            tri.add_triplet(obs % 4, obs, 1.0);
        }
        let zt = tri.to_csr();
        let fast = build_zt_z_intercept_from_zt(&zt);
        let slow = (&zt * &zt.transpose_view()).to_csr();
        assert_eq!(fast.indptr().raw_storage(), slow.indptr().raw_storage());
        assert_eq!(fast.indices(), slow.indices());
        assert_eq!(fast.data(), slow.data());
    }

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
