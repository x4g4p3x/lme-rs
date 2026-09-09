//! Group-block normal equations for nonlinear least squares.
use ndarray::{s, Array1, Array2};
use ndarray_linalg::{Factorize, Solve};

/// An arrowhead matrix: population parameters couple to independent group blocks.
pub(super) struct BlockNormal {
    fixed: Array2<f64>,
    cross: Vec<Array2<f64>>,
    random: Vec<Array2<f64>>,
    fixed_rhs: Array1<f64>,
    random_rhs: Vec<Array1<f64>>,
}

impl BlockNormal {
    pub fn new(p: usize, k: usize, groups: usize) -> Self {
        Self {
            fixed: Array2::zeros((p, p)),
            cross: vec![Array2::zeros((p, k)); groups],
            random: vec![Array2::zeros((k, k)); groups],
            fixed_rhs: Array1::zeros(p),
            random_rhs: vec![Array1::zeros(k); groups],
        }
    }

    pub fn accumulate(&mut self, group: usize, fixed: &[f64], random: &[f64], residual: f64) {
        for (i, &a) in fixed.iter().enumerate() {
            self.fixed_rhs[i] += a * residual;
            for (j, &b) in fixed.iter().enumerate() {
                self.fixed[[i, j]] += a * b;
            }
            for (j, &b) in random.iter().enumerate() {
                self.cross[group][[i, j]] += a * b;
            }
        }
        for (i, &a) in random.iter().enumerate() {
            self.random_rhs[group][i] += a * residual;
            for (j, &b) in random.iter().enumerate() {
                self.random[group][[i, j]] += a * b;
            }
        }
    }

    pub fn add_prior(&mut self, inverse: &Array2<f64>, b: &Array1<f64>, subtract_gradient: bool) {
        let k = inverse.nrows();
        for (g, block) in self.random.iter_mut().enumerate() {
            *block += inverse;
            if subtract_gradient {
                self.random_rhs[g] -= &inverse.dot(&b.slice(s![g * k..(g + 1) * k]));
            }
        }
    }

    pub fn solve(&self, damping: f64) -> Option<Array1<f64>> {
        let p = self.fixed.nrows();
        let k = self.random[0].nrows();
        let mut schur = self.fixed.clone();
        let mut rhs = self.fixed_rhs.clone();
        for i in 0..p {
            schur[[i, i]] += damping * self.fixed[[i, i]].max(1e-8);
        }
        let mut solved = Vec::with_capacity(self.random.len());
        for g in 0..self.random.len() {
            let mut c = self.random[g].clone();
            for i in 0..k {
                c[[i, i]] += damping * self.random[g][[i, i]].max(1e-8);
            }
            let factor = c.factorize().ok()?;
            let v = factor.solve(&self.random_rhs[g]).ok()?;
            let mut w = Array2::zeros((k, p));
            for j in 0..p {
                w.column_mut(j)
                    .assign(&factor.solve(&self.cross[g].row(j).to_owned()).ok()?);
            }
            schur -= &self.cross[g].dot(&w);
            rhs -= &self.cross[g].dot(&v);
            solved.push((v, w));
        }
        let fixed = schur.solve(&rhs).ok()?;
        let mut out = Array1::zeros(p + k * self.random.len());
        out.slice_mut(s![..p]).assign(&fixed);
        for (g, (v, w)) in solved.into_iter().enumerate() {
            out.slice_mut(s![p + g * k..p + (g + 1) * k])
                .assign(&(v - w.dot(&fixed)));
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn block_step_matches_dense_normal_equations() {
        let x = array![[1., 2.], [1., 3.], [1., 4.], [1., 5.], [1., 6.], [1., 7.]];
        let residual = array![2., -1., 3., 1., -2., 4.];
        let mut blocks = BlockNormal::new(2, 2, 3);
        let mut j = Array2::zeros((6, 8));
        for i in 0..6 {
            let g = i / 2;
            let row = x.row(i).to_vec();
            blocks.accumulate(g, &row, &row, residual[i]);
            j.slice_mut(s![i, ..2]).assign(&x.row(i));
            j.slice_mut(s![i, 2 + 2 * g..4 + 2 * g]).assign(&x.row(i));
        }
        let prior = array![[2., 0.3], [0.3, 1.]];
        let b = array![0.2, -0.1, 0.5, 0.3, -0.2, 0.8];
        blocks.add_prior(&prior, &b, true);
        let mut dense = j.t().dot(&j);
        let mut rhs = j.t().dot(&residual);
        for g in 0..3 {
            for a in 0..2 {
                for c in 0..2 {
                    dense[[2 + g * 2 + a, 2 + g * 2 + c]] += prior[[a, c]];
                }
            }
            rhs.slice_mut(s![2 + g * 2..4 + g * 2])
                .scaled_add(-1., &prior.dot(&b.slice(s![g * 2..g * 2 + 2])));
        }
        for damping in [0.001, 0.1, 2.] {
            let mut a = dense.clone();
            for i in 0..8 {
                a[[i, i]] += damping * dense[[i, i]].max(1e-8);
            }
            let expected = a.solve(&rhs).unwrap();
            let actual = blocks.solve(damping).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-9, "{a} != {b}");
            }
        }
    }
}
