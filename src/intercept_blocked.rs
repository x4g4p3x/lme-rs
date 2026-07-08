//! Blocked augmented Cholesky for intercept-only LMMs (MixedModels.jl `updateL!` layout).

use ndarray::Array2;
use sprs::CsMat;

struct ReLower {
    n: usize,
    data: Vec<f64>,
}

impl ReLower {
    fn diagonal_init(n: usize, diag: &[f64]) -> Self {
        let mut data = vec![0.0; n * (n + 1) / 2];
        for i in 0..n {
            data[i * (i + 1) / 2 + i] = diag[i];
        }
        Self { n, data }
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        debug_assert!(c <= r);
        self.data[r * (r + 1) / 2 + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, val: f64) {
        debug_assert!(c <= r);
        self.data[r * (r + 1) / 2 + c] = val;
    }

    fn rank_sub_dense(&mut self, cross: &Array2<f64>) {
        debug_assert_eq!(cross.nrows(), self.n);
        for r in 0..self.n {
            for c in 0..=r {
                let mut s = 0.0;
                for k in 0..cross.ncols() {
                    s += cross[[r, k]] * cross[[c, k]];
                }
                if s != 0.0 {
                    self.set(r, c, self.get(r, c) - s);
                }
            }
        }
    }

    fn chol(&mut self) -> Result<f64, ()> {
        let mut log_det = 0.0;
        for j in 0..self.n {
            let mut sum = self.get(j, j);
            for k in 0..j {
                let ljk = self.get(j, k);
                sum -= ljk * ljk;
            }
            if sum <= 0.0 {
                return Err(());
            }
            let ljj = sum.sqrt();
            self.set(j, j, ljj);
            log_det += ljj.ln();
            for i in (j + 1)..self.n {
                let mut s = self.get(i, j);
                for k in 0..j {
                    s -= self.get(i, k) * self.get(j, k);
                }
                self.set(i, j, s / ljj);
            }
        }
        Ok(log_det)
    }

    fn solve_col(&self, rhs: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(rhs.len(), n);
        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut s = rhs[i];
            for (k, xk) in x.iter().enumerate().take(i) {
                s -= self.get(i, k) * xk;
            }
            x[i] = s / self.get(i, i);
        }
        rhs.copy_from_slice(&x);
    }
}

pub(crate) struct InterceptBlockedChol {
    k_re: usize,
    p: usize,
    n_re: Vec<usize>,
    theta_idx: Vec<usize>,
    a_diag: Vec<Vec<f64>>,
    a_cross: Vec<Vec<Array2<f64>>>,
    l_cross: Vec<Vec<Array2<f64>>>,
    a_xy_re: Vec<Array2<f64>>,
    l_xy_re: Vec<Array2<f64>>,
    a_xy_xy: Array2<f64>,
    l_xy_xy: Array2<f64>,
    l_diag0: Vec<f64>,
    l_dense: Vec<ReLower>,
    solve_buf: Vec<f64>,
}

impl InterceptBlockedChol {
    pub fn try_new(lmm: &super::LmmData) -> Option<Self> {
        if !lmm.intercept_only_re() {
            return None;
        }
        let re_blocks = &lmm.re_blocks;
        if re_blocks.is_empty() {
            return None;
        }
        let p = lmm.x.ncols();
        let mut order: Vec<usize> = (0..re_blocks.len()).collect();
        order.sort_by(|&a, &b| re_blocks[b].m.cmp(&re_blocks[a].m));

        let k_re = order.len();
        let mut n_re = Vec::with_capacity(k_re);
        let mut theta_idx = Vec::with_capacity(k_re);
        let mut ranges = Vec::with_capacity(k_re);
        let mut offset = 0usize;
        for &orig in &order {
            let b = &re_blocks[orig];
            if b.k != 1 {
                return None;
            }
            n_re.push(b.m);
            theta_idx.push(orig);
            ranges.push((offset, offset + b.m));
            offset += b.m;
        }
        if offset != lmm.zt_z.rows() {
            return None;
        }
        if !blocked_cross_fits(&n_re) {
            return None;
        }

        let mut a_diag = Vec::with_capacity(k_re);
        for &(s, e) in &ranges {
            a_diag.push(extract_diag(&lmm.zt_z, s, e));
        }

        let mut a_cross = Vec::with_capacity(k_re.saturating_sub(1));
        let mut l_cross = Vec::with_capacity(k_re.saturating_sub(1));
        for j in 1..k_re {
            let mut a_row = Vec::with_capacity(j);
            let mut l_row = Vec::with_capacity(j);
            for jj in 0..j {
                let dense = extract_dense_submatrix(&lmm.zt_z, ranges[j], ranges[jj]);
                l_row.push(Array2::zeros(dense.dim()));
                a_row.push(dense);
            }
            a_cross.push(a_row);
            l_cross.push(l_row);
        }

        let pq = p + 1;
        let mut a_xy_re = Vec::with_capacity(k_re);
        let mut l_xy_re = Vec::with_capacity(k_re);
        for &(s, e) in &ranges {
            let m = e - s;
            let mut block = Array2::<f64>::zeros((pq, m));
            for local in 0..m {
                let k = s + local;
                for r in 0..p {
                    block[[r, local]] = lmm.zt_x[[k, r]];
                }
                block[[p, local]] = lmm.zt_y[k];
            }
            a_xy_re.push(block.clone());
            l_xy_re.push(block);
        }

        let mut a_xy_xy = Array2::<f64>::zeros((pq, pq));
        a_xy_xy.slice_mut(ndarray::s![0..p, 0..p]).assign(&lmm.xt_x);
        for r in 0..p {
            a_xy_xy[[r, p]] = lmm.xt_y[r];
            a_xy_xy[[p, r]] = lmm.xt_y[r];
        }
        a_xy_xy[[p, p]] = lmm.y_norm2;

        let max_m = *n_re.iter().max().unwrap_or(&0);
        let n0 = n_re[0];
        let l_dense = (1..k_re)
            .map(|j| ReLower::diagonal_init(n_re[j], &vec![0.0; n_re[j]]))
            .collect();
        Some(Self {
            k_re,
            p,
            n_re,
            theta_idx,
            a_diag,
            a_cross,
            l_cross,
            a_xy_re,
            l_xy_re,
            a_xy_xy: a_xy_xy.clone(),
            l_xy_xy: a_xy_xy,
            l_diag0: vec![0.0; n0],
            l_dense,
            solve_buf: vec![0.0; max_m],
        })
    }

    pub fn profile_deviance(&mut self, lmm: &super::LmmData, theta: &[f64], reml: bool) -> f64 {
        match self.update_l_and_factor(theta) {
            Ok(log_det_re) => self.deviance_from_factor(lmm, reml, log_det_re),
            Err(()) => f64::MAX,
        }
    }

    fn update_l_and_factor(&mut self, theta: &[f64]) -> Result<f64, ()> {
        let k_re = self.k_re;

        self.l_xy_xy.assign(&self.a_xy_xy);
        for j in 0..k_re {
            let th = theta[self.theta_idx[j]];
            let th2 = th * th;
            if j == 0 {
                for i in 0..self.n_re[0] {
                    self.l_diag0[i] = th2 * self.a_diag[0][i] + 1.0;
                }
            } else {
                let diag: Vec<f64> = self.a_diag[j].iter().map(|&a| th2 * a + 1.0).collect();
                self.l_dense[j - 1] = ReLower::diagonal_init(self.n_re[j], &diag);
            }
            for i in (j + 1)..k_re {
                self.l_cross[i - 1][j].assign(&self.a_cross[i - 1][j]);
                self.l_cross[i - 1][j].mapv_inplace(|v| v * th);
            }
            self.l_xy_re[j].assign(&self.a_xy_re[j]);
            self.l_xy_re[j].mapv_inplace(|v| v * th);
            if j > 0 {
                for jj in 0..j {
                    self.l_cross[j - 1][jj].mapv_inplace(|v| v * th);
                }
            }
        }

        let mut log_det_re = 0.0;
        let kb = k_re + 1;
        for j in 0..kb {
            if j < k_re {
                for jj in 0..j {
                    if j > 0 {
                        self.l_dense[j - 1].rank_sub_dense(&self.l_cross[j - 1][jj]);
                    }
                }
                if j == 0 {
                    for i in 0..self.n_re[0] {
                        if self.l_diag0[i] <= 0.0 {
                            return Err(());
                        }
                        self.l_diag0[i] = self.l_diag0[i].sqrt();
                        log_det_re += self.l_diag0[i].ln();
                    }
                } else {
                    log_det_re += self.l_dense[j - 1].chol()?;
                }
            } else {
                for jj in 0..k_re {
                    self.schur_sub_xy_xy(jj)?;
                }
                self.chol_xy_block()?;
            }

            for i in (j + 1)..kb {
                for jj in 0..j {
                    if i < k_re {
                        self.schur_sub_re_cross(i, j, jj)?;
                    } else {
                        self.schur_sub_xy_re(j, jj)?;
                    }
                }
                if i < k_re {
                    self.trisolve_re_cross(i, j)?;
                } else {
                    self.trisolve_xy_re(j)?;
                }
            }
        }

        Ok(2.0 * log_det_re)
    }

    fn schur_sub_re_cross(&mut self, i: usize, j: usize, jj: usize) -> Result<(), ()> {
        let lij = self.l_cross[i - 1][jj].clone();
        let ljj = self.l_cross[j - 1][jj].clone();
        let target = &mut self.l_cross[i - 1][j];
        for col in 0..target.ncols() {
            for row in 0..target.nrows() {
                let mut s = target[[row, col]];
                for k in 0..lij.ncols() {
                    s -= lij[[row, k]] * ljj[[col, k]];
                }
                target[[row, col]] = s;
            }
        }
        Ok(())
    }

    fn trisolve_re_cross(&mut self, i: usize, j: usize) -> Result<(), ()> {
        let cross = &self.l_cross[i - 1][j];
        let mut updated = cross.clone();
        if j == 0 {
            for col in 0..cross.ncols() {
                let d = self.l_diag0[col];
                if d == 0.0 {
                    return Err(());
                }
                for row in 0..cross.nrows() {
                    updated[[row, col]] /= d;
                }
            }
        } else {
            for col in 0..cross.ncols() {
                for row in 0..cross.nrows() {
                    self.solve_buf[row] = cross[[row, col]];
                }
                self.l_dense[j - 1].solve_col(&mut self.solve_buf[..cross.nrows()]);
                for row in 0..cross.nrows() {
                    updated[[row, col]] = self.solve_buf[row];
                }
            }
        }
        self.l_cross[i - 1][j] = updated;
        Ok(())
    }

    fn schur_sub_xy_re(&mut self, j: usize, jj: usize) -> Result<(), ()> {
        let lij = self.l_xy_re[jj].clone();
        let ljj = self.l_cross[j - 1][jj].clone();
        let mut block = self.l_xy_re[j].clone();
        for col in 0..block.ncols() {
            for row in 0..block.nrows() {
                let mut s = block[[row, col]];
                for k in 0..lij.ncols() {
                    s -= lij[[row, k]] * ljj[[col, k]];
                }
                block[[row, col]] = s;
            }
        }
        self.l_xy_re[j] = block;
        Ok(())
    }

    fn trisolve_xy_re(&mut self, j: usize) -> Result<(), ()> {
        let m = self.n_re[j];
        let pq = self.p + 1;
        let mut block = self.l_xy_re[j].clone();
        if j == 0 {
            for col in 0..m {
                let d = self.l_diag0[col];
                if d == 0.0 {
                    return Err(());
                }
                for row in 0..pq {
                    block[[row, col]] /= d;
                }
            }
        } else {
            let ljj = &self.l_dense[j - 1];
            for row in 0..pq {
                let mut rhs: Vec<f64> = (0..m).map(|col| block[[row, col]]).collect();
                ljj.solve_col(&mut rhs);
                for col in 0..m {
                    block[[row, col]] = rhs[col];
                }
            }
        }
        self.l_xy_re[j] = block;
        Ok(())
    }

    fn schur_sub_xy_xy(&mut self, jj: usize) -> Result<(), ()> {
        let lxy = self.l_xy_re[jj].clone();
        let pq = self.p + 1;
        for col in 0..pq {
            for row in col..pq {
                let mut s = self.l_xy_xy[[row, col]];
                for k in 0..lxy.ncols() {
                    s -= lxy[[row, k]] * lxy[[col, k]];
                }
                self.l_xy_xy[[row, col]] = s;
                if row != col {
                    self.l_xy_xy[[col, row]] = s;
                }
            }
        }
        Ok(())
    }

    fn chol_xy_block(&mut self) -> Result<(), ()> {
        let n = self.p + 1;
        for j in 0..n {
            let mut sum = self.l_xy_xy[[j, j]];
            for k in 0..j {
                sum -= self.l_xy_xy[[j, k]].powi(2);
            }
            if sum <= 0.0 {
                return Err(());
            }
            let ljj = sum.sqrt();
            self.l_xy_xy[[j, j]] = ljj;
            for i in (j + 1)..n {
                let mut s = self.l_xy_xy[[i, j]];
                for k in 0..j {
                    s -= self.l_xy_xy[[i, k]] * self.l_xy_xy[[j, k]];
                }
                self.l_xy_xy[[i, j]] = s / ljj;
            }
        }
        Ok(())
    }

    fn deviance_from_factor(&self, lmm: &super::LmmData, reml: bool, log_det_a: f64) -> f64 {
        let n = lmm.y.len() as f64;
        let p = self.p as f64;
        let r_yy = self.l_xy_xy[[self.p, self.p]];
        let r2 = r_yy * r_yy;
        let reml_df = if reml { n - p } else { n };
        if r2 <= 0.0 || reml_df <= 0.0 {
            return f64::MAX;
        }
        let sigma2 = r2 / reml_df;
        let twopi = std::f64::consts::PI * 2.0;
        let mut deviance = reml_df * (twopi * sigma2).ln() + log_det_a + reml_df;
        if reml {
            let mut log_det_l_x = 0.0;
            for j in 0..self.p {
                let ljj = self.l_xy_xy[[j, j]];
                if ljj <= 0.0 {
                    return f64::MAX;
                }
                log_det_l_x += ljj.ln();
            }
            deviance += 2.0 * log_det_l_x;
        }
        deviance
    }
}

fn blocked_cross_fits(n_re: &[usize]) -> bool {
    if n_re.len() < 2 {
        return false;
    }
    let mut max_cross = 0usize;
    for j in 1..n_re.len() {
        for jj in 0..j {
            max_cross = max_cross.max(n_re[j].saturating_mul(n_re[jj]));
        }
    }
    // crossed_20k (250×100) fits; nested_10k (2000×200) does not — keep sparse LDL there.
    max_cross <= 100_000
}

fn extract_diag(zt_z: &CsMat<f64>, start: usize, end: usize) -> Vec<f64> {
    let mut d = vec![0.0; end - start];
    for i in start..end {
        let row_view = zt_z.outer_view(i).expect("missing zt_z row");
        for (j, &v) in row_view.iter() {
            if j == i {
                d[i - start] = v;
                break;
            }
        }
    }
    d
}

fn extract_dense_submatrix(
    zt_z: &CsMat<f64>,
    row_range: (usize, usize),
    col_range: (usize, usize),
) -> Array2<f64> {
    let (r0, r1) = row_range;
    let (c0, c1) = col_range;
    let mut out = Array2::<f64>::zeros((r1 - r0, c1 - c0));
    for local_col in 0..out.ncols() {
        let global_col = c0 + local_col;
        let row_view = zt_z
            .outer_view(global_col)
            .expect("zt_z column missing in cross block");
        for (global_row, &val) in row_view.iter() {
            if (r0..r1).contains(&global_row) {
                out[[global_row - r0, local_col]] = val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::LmmData;
    use crate::model_matrix::ReBlock;
    use ndarray::{Array1, Array2};
    use sprs::TriMat;

    fn load_penicillin_lmm() -> (LmmData, Vec<f64>) {
        let raw: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/penicillin.json")).unwrap();
        let x_vec: Vec<Vec<f64>> = serde_json::from_value(raw["inputs"]["X"].clone()).unwrap();
        let zt_vec: Vec<Vec<f64>> = serde_json::from_value(raw["inputs"]["Zt"].clone()).unwrap();
        let y_vec: Vec<f64> = serde_json::from_value(raw["inputs"]["y"].clone()).unwrap();
        let theta: Vec<f64> = serde_json::from_value(raw["outputs"]["theta"].clone()).unwrap();

        let x = Array2::from_shape_vec(
            (x_vec.len(), x_vec[0].len()),
            x_vec.into_iter().flatten().collect(),
        )
        .unwrap();
        let mut tri = TriMat::new((zt_vec.len(), zt_vec[0].len()));
        for (i, row) in zt_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    tri.add_triplet(i, j, val);
                }
            }
        }
        let zt = tri.to_csr();
        let y = Array1::from_vec(y_vec);
        let re_blocks = vec![
            ReBlock {
                m: 24,
                k: 1,
                theta_len: 1,
                group_name: "plate".into(),
                effect_names: vec!["(Intercept)".into()],
                group_map: Default::default(),
            },
            ReBlock {
                m: 6,
                k: 1,
                theta_len: 1,
                group_name: "sample".into(),
                effect_names: vec!["(Intercept)".into()],
                group_map: Default::default(),
            },
        ];
        (LmmData::new(x, zt, y, re_blocks), theta)
    }

    #[test]
    fn blocked_deviance_matches_full_profile_on_penicillin_grid() {
        let (lmm, golden_theta) = load_penicillin_lmm();
        let mut blocked = InterceptBlockedChol::try_new(&lmm).expect("blocked setup");
        for &t0 in &[0.5, 1.0, golden_theta[0]] {
            for &t1 in &[0.5, 1.0, golden_theta[1]] {
                let theta = [t0, t1];
                let fast = blocked.profile_deviance(&lmm, &theta, true);
                let slow = lmm.evaluate(&theta, true).reml_crit;
                let scale = slow.abs().max(1.0);
                assert!(
                    (fast - slow).abs() <= 1e-6 * scale,
                    "theta={theta:?} fast={fast} slow={slow}"
                );
            }
        }
    }
}
