use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Solve};
use ndarray_linalg::UPLO;

pub struct LmmData {
    pub x: Array2<f64>,
    pub zt: Array2<f64>,
    pub y: Array1<f64>,
}

impl LmmData {
    pub fn new(x: Array2<f64>, zt: Array2<f64>, y: Array1<f64>) -> Self {
        LmmData { x, zt, y }
    }

    /// Calculate REML deviance given a single scalar theta.
    /// Assumes (1 | Group) structure so Lambda is theta * I.
    pub fn log_reml_deviance(&self, theta: f64) -> f64 {
        let n = self.y.len() as f64;
        let p = self.x.ncols() as f64;
        let q = self.zt.nrows();

        let zt = &self.zt;
        let x = &self.x;
        let y = &self.y;

        // 1. A = theta^2 * Zt * Zt^T + I
        let mut a = zt.dot(&zt.t());
        a *= theta * theta;
        for i in 0..q {
            a[[i, i]] += 1.0;
        }

        // 2. L = Cholesky(A)
        // using lower triangular
        let l = a.cholesky(UPLO::Lower).expect("Cholesky of A failed");

        // 3. cu = L^{-1} * (theta * Zt * y)
        let zt_y = zt.dot(y) * theta;
        // l is lower triangular. ndarray-linalg solve for Triangular
        // Unfortunately ndarray_linalg does not have a convenient solve_triangular for general L.
        // We can just use the generic solve.
        let cu = l.solve(&zt_y).expect("Solve for cu failed");

        // 4. Rzx = L^{-1} * (theta * Zt * X)
        let zt_x = zt.dot(x) * theta;
        let p_usize = p as usize;
        let mut rzx = Array2::<f64>::zeros((q, p_usize));
        for j in 0..p_usize {
            let col = l.solve(&zt_x.column(j).to_owned()).expect("Solve for Rzx failed");
            rzx.column_mut(j).assign(&col);
        }

        // 5. X^T X
        let xt_x = x.t().dot(x);

        // 6. Rzx^T * Rzx
        let rzx_t_rzx = rzx.t().dot(&rzx);

        // 7. A_x = X^T X - Rzx^T Rzx
        let a_x = xt_x - rzx_t_rzx;

        // 8. L_x = Cholesky(A_x)
        let l_x = a_x.cholesky(UPLO::Lower).expect("Cholesky of A_x failed");

        // 9. c_beta = L_x^{-1} * (X^T y - Rzx^T cu)
        let xt_y = x.t().dot(y);
        let rhs_beta = xt_y - rzx.t().dot(&cu);
        let c_beta = l_x.solve(&rhs_beta).expect("Solve for c_beta failed");

        // 10. r^2 = ||y||^2 - ||cu||^2 - ||c_beta||^2
        let y_norm2 = y.dot(y);
        let cu_norm2 = cu.dot(&cu);
        let c_beta_norm2 = c_beta.dot(&c_beta);
        let r2 = y_norm2 - cu_norm2 - c_beta_norm2;

        let reml_df = n - p;
        let sigma2 = r2 / reml_df;

        // 11. REML deviance = (n - p) * log(2 * pi * sigma2) + 2 * sum(log(diag(L))) + 2 * sum(log(diag(L_x))) + (n - p)
        let mut log_det_l = 0.0;
        for i in 0..l.nrows() {
            log_det_l += l[[i, i]].ln();
        }
        
        let mut log_det_l_x = 0.0;
        for i in 0..l_x.nrows() {
            log_det_l_x += l_x[[i, i]].ln();
        }

        let twopi = std::f64::consts::PI * 2.0;

        let reml_crit = reml_df * (twopi * sigma2).ln()
            + 2.0 * log_det_l
            + 2.0 * log_det_l_x
            + reml_df;

        reml_crit
    }
}
