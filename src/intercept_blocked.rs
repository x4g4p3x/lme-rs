//! Blocked augmented Cholesky for intercept-only LMMs (MixedModels.jl `updateL!` layout).

use ndarray::linalg::general_mat_mul;
use ndarray::Array2;
use sprs::CsMat;
use std::collections::HashSet;

const SPARSE_CROSS_MAX_DENSITY: f64 = 0.35;

/// Off-diagonal RE cross block: dense when fill is high (crossed), sparse CSC otherwise (nested).
#[derive(Clone)]
enum CrossBlock {
    Dense(Array2<f64>),
    Sparse {
        nrows: usize,
        ncols: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        vals: Vec<f64>,
    },
}

impl CrossBlock {
    fn nrows(&self) -> usize {
        match self {
            CrossBlock::Dense(d) => d.nrows(),
            CrossBlock::Sparse { nrows, .. } => *nrows,
        }
    }

    fn ncols(&self) -> usize {
        match self {
            CrossBlock::Dense(d) => d.ncols(),
            CrossBlock::Sparse { ncols, .. } => *ncols,
        }
    }

    fn entry(&self, row: usize, col: usize) -> f64 {
        match self {
            CrossBlock::Dense(d) => d[[row, col]],
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ..
            } => {
                let start = col_ptr[col];
                let end = col_ptr[col + 1];
                for idx in start..end {
                    if row_idx[idx] == row {
                        return vals[idx];
                    }
                }
                0.0
            }
        }
    }

    fn assign_from(&mut self, other: &Self) {
        match (self, other) {
            (CrossBlock::Dense(a), CrossBlock::Dense(b)) => a.assign(b),
            (
                CrossBlock::Sparse {
                    col_ptr: cp,
                    row_idx: ri,
                    vals: v,
                    nrows,
                    ncols,
                },
                CrossBlock::Sparse {
                    col_ptr: cp2,
                    row_idx: ri2,
                    vals: v2,
                    nrows: n2,
                    ncols: nc2,
                },
            ) => {
                debug_assert_eq!(*nrows, *n2);
                debug_assert_eq!(*ncols, *nc2);
                cp.clone_from(cp2);
                ri.clone_from(ri2);
                v.clone_from(v2);
            }
            _ => panic!("CrossBlock storage mismatch in assign_from"),
        }
    }

    fn scale(&mut self, factor: f64) {
        if factor == 1.0 {
            return;
        }
        match self {
            CrossBlock::Dense(d) => d.mapv_inplace(|v| v * factor),
            CrossBlock::Sparse { vals, .. } => {
                for v in vals {
                    *v *= factor;
                }
            }
        }
    }

    fn zeros_like(&self) -> Self {
        match self {
            CrossBlock::Dense(d) => CrossBlock::Dense(Array2::zeros(d.dim())),
            CrossBlock::Sparse {
                nrows,
                ncols,
                col_ptr,
                ..
            } => CrossBlock::Sparse {
                nrows: *nrows,
                ncols: *ncols,
                col_ptr: col_ptr.clone(),
                row_idx: Vec::new(),
                vals: Vec::new(),
            },
        }
    }

    fn from_submatrix(
        zt_z: &CsMat<f64>,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> Self {
        Self::from_submatrix_raw(zt_z, row_range, col_range)
    }

    fn from_submatrix_raw(
        zt_z: &CsMat<f64>,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> Self {
        let (r0, r1) = row_range;
        let (c0, c1) = col_range;
        let nrows = r1 - r0;
        let ncols = c1 - c0;
        let mut col_ptr = vec![0usize; ncols + 1];
        let mut row_idx = Vec::new();
        let mut vals = Vec::new();
        for local_col in 0..ncols {
            let global_col = c0 + local_col;
            let row_view = zt_z
                .outer_view(global_col)
                .expect("zt_z column missing in cross block");
            for (global_row, &val) in row_view.iter() {
                if (r0..r1).contains(&global_row) {
                    row_idx.push(global_row - r0);
                    vals.push(val);
                }
            }
            col_ptr[local_col + 1] = row_idx.len();
        }
        let nnz = vals.len();
        let density = nnz as f64 / (nrows as f64 * ncols as f64);
        let elems = nrows.saturating_mul(ncols);
        if elems <= 100_000 || density > SPARSE_CROSS_MAX_DENSITY {
            let mut dense = Array2::<f64>::zeros((nrows, ncols));
            for local_col in 0..ncols {
                let start = col_ptr[local_col];
                let end = col_ptr[local_col + 1];
                for idx in start..end {
                    dense[[row_idx[idx], local_col]] = vals[idx];
                }
            }
            CrossBlock::Dense(dense)
        } else {
            CrossBlock::Sparse {
                nrows,
                ncols,
                col_ptr,
                row_idx,
                vals,
            }
        }
    }

    fn assign_scaled(&mut self, src: &Self, scale: f64) {
        match (self, src) {
            (CrossBlock::Dense(dst), CrossBlock::Dense(src_m)) => {
                if scale == 1.0 {
                    dst.assign(src_m);
                } else {
                    ndarray::Zip::from(dst)
                        .and(src_m)
                        .for_each(|d, &s| *d = s * scale);
                }
            }
            (dst, src) => {
                dst.assign_from(src);
                dst.scale(scale);
            }
        }
    }

    fn fits_blocked_gate(&self) -> bool {
        match self {
            CrossBlock::Dense(d) => d.nrows() * d.ncols() <= 100_000,
            // Nested `batch/cask`: one batch row per cask column; uses `ReFactor::Diagonal`
            // on the batch block instead of densifying the cross.
            CrossBlock::Sparse { .. } => self.columns_single_row(),
        }
    }

    /// Each cross column touches exactly one row (nested `batch/cask`: cask → batch).
    fn columns_single_row(&self) -> bool {
        for col in 0..self.ncols() {
            let mut count = 0usize;
            match self {
                CrossBlock::Dense(d) => {
                    for row in 0..d.nrows() {
                        if d[[row, col]] != 0.0 {
                            count += 1;
                        }
                    }
                }
                CrossBlock::Sparse { col_ptr, .. } => {
                    count = col_ptr[col + 1] - col_ptr[col];
                }
            }
            if count != 1 {
                return false;
            }
        }
        true
    }

    /// Groups cross column indices by their sole row (batch → casks for nested RE).
    fn row_grouped_columns(&self) -> Option<Vec<Vec<usize>>> {
        if !self.columns_single_row() {
            return None;
        }
        let nrows = self.nrows();
        let mut groups = vec![Vec::new(); nrows];
        for col in 0..self.ncols() {
            let row = match self {
                CrossBlock::Dense(d) => {
                    let mut r = None;
                    for row in 0..d.nrows() {
                        if d[[row, col]] != 0.0 {
                            r = Some(row);
                        }
                    }
                    r?
                }
                CrossBlock::Sparse {
                    col_ptr, row_idx, ..
                } => row_idx[col_ptr[col]],
            };
            groups[row].push(col);
        }
        Some(groups)
    }

    /// Each RE row owned by exactly one column (crossed plate×sample — not nested).
    #[allow(dead_code)]
    fn column_disjoint_partition(&self) -> Option<Vec<Vec<usize>>> {
        let nrows = self.nrows();
        let mut col_rows = Vec::with_capacity(self.ncols());
        let mut seen = HashSet::new();
        for col in 0..self.ncols() {
            let mut rows = Vec::new();
            match self {
                CrossBlock::Dense(d) => {
                    for row in 0..nrows {
                        if d[[row, col]] != 0.0 {
                            rows.push(row);
                        }
                    }
                }
                CrossBlock::Sparse {
                    col_ptr, row_idx, ..
                } => {
                    rows.extend_from_slice(&row_idx[col_ptr[col]..col_ptr[col + 1]]);
                }
            }
            for &r in &rows {
                if !seen.insert(r) {
                    return None;
                }
            }
            col_rows.push(rows);
        }
        if seen.len() == nrows {
            Some(col_rows)
        } else {
            None
        }
    }

    /// Rank update onto a diagonal RE factor (nested batch block from `batch/cask` cross).
    fn rank_sub_diagonal(&self, diag: &mut [f64]) {
        let Some(groups) = self.row_grouped_columns() else {
            return;
        };
        for (row, cols) in groups.iter().enumerate() {
            let mut sum_sq = 0.0;
            for &col in cols {
                let v = self.entry(row, col);
                sum_sq += v * v;
            }
            diag[row] -= sum_sq;
        }
    }

    /// `cross[:,col] /= diag_chol[row]` when each column has one structural row.
    fn trisolve_single_row_cols(&mut self, diag_chol: &[f64]) -> Result<(), ()> {
        match self {
            CrossBlock::Dense(d) => {
                for col in 0..d.ncols() {
                    for row in 0..d.nrows() {
                        let v = d[[row, col]];
                        if v != 0.0 {
                            let div = diag_chol[row];
                            if div == 0.0 {
                                return Err(());
                            }
                            d[[row, col]] = v / div;
                        }
                    }
                }
                Ok(())
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ncols,
                ..
            } => {
                for col in 0..*ncols {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let r = row_idx[idx];
                        let div = diag_chol[r];
                        if div == 0.0 {
                            return Err(());
                        }
                        vals[idx] /= div;
                    }
                }
                Ok(())
            }
        }
    }

    fn rank_sub_lower(&self, target: &mut ReLower, gram: &mut Array2<f64>) {
        match self {
            CrossBlock::Dense(d) => target.rank_sub_dense(d, gram),
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ..
            } => {
                for col in 0..self.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for i in start..end {
                        let r1 = row_idx[i];
                        let v1 = vals[i];
                        for j in start..=i {
                            let r2 = row_idx[j];
                            let v2 = vals[j];
                            let (r, c) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
                            target.set(r, c, target.get(r, c) - v1 * v2);
                        }
                    }
                }
            }
        }
    }

    fn rank_sub_column_blocks(&self, blocks: &mut [ReLower], col_rows: &[Vec<usize>]) {
        for (col, rows) in col_rows.iter().enumerate() {
            let mut local_vals = Vec::with_capacity(rows.len());
            match self {
                CrossBlock::Dense(d) => {
                    for &r in rows {
                        local_vals.push(d[[r, col]]);
                    }
                }
                CrossBlock::Sparse {
                    col_ptr,
                    row_idx: _,
                    vals,
                    ..
                } => {
                    local_vals.extend_from_slice(&vals[col_ptr[col]..col_ptr[col + 1]]);
                }
            }
            let local_indices: Vec<usize> = (0..rows.len()).collect();
            blocks[col].rank_sub_outer(&local_indices, &local_vals);
        }
    }

    fn trisolve_diag0(&mut self, diag0: &[f64]) -> Result<(), ()> {
        let nrows = self.nrows();
        for col in 0..self.ncols() {
            let d = diag0[col];
            if d == 0.0 {
                return Err(());
            }
            match self {
                CrossBlock::Dense(dense) => {
                    for row in 0..nrows {
                        dense[[row, col]] /= d;
                    }
                }
                CrossBlock::Sparse { col_ptr, vals, .. } => {
                    for v in &mut vals[col_ptr[col]..col_ptr[col + 1]] {
                        *v /= d;
                    }
                }
            }
        }
        Ok(())
    }

    fn trisolve_column_blocks(
        &mut self,
        col: usize,
        blocks: &[ReLower],
        col_rows: &[Vec<usize>],
    ) -> Result<(), ()> {
        let rows = &col_rows[col];
        let mut rhs = vec![0.0; rows.len()];
        match self {
            CrossBlock::Dense(d) => {
                for (li, &r) in rows.iter().enumerate() {
                    rhs[li] = d[[r, col]];
                }
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ..
            } => {
                for (li, idx) in (col_ptr[col]..col_ptr[col + 1]).enumerate() {
                    debug_assert_eq!(row_idx[idx], rows[li]);
                    rhs[li] = vals[idx];
                }
            }
        }
        let mut work = rhs.clone();
        blocks[col].solve_col(&mut work);
        match self {
            CrossBlock::Dense(d) => {
                for (li, &r) in rows.iter().enumerate() {
                    d[[r, col]] = work[li];
                }
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx: _,
                vals,
                ..
            } => {
                let span = col_ptr[col]..col_ptr[col + 1];
                let n = span.end - span.start;
                vals[span].copy_from_slice(&work[..n]);
            }
        }
        Ok(())
    }

    fn trisolve_full_lower(&mut self, lower: &ReLower, rhs_buf: &mut [f64]) -> Result<(), ()> {
        let nrows = self.nrows();
        debug_assert!(rhs_buf.len() >= nrows);
        debug_assert_eq!(nrows, lower.n);
        match self {
            CrossBlock::Dense(d) => {
                lower.solve_multi_rhs(d);
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ncols,
                ..
            } => {
                for col in 0..*ncols {
                    rhs_buf[..nrows].fill(0.0);
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        rhs_buf[row_idx[idx]] = vals[idx];
                    }
                    lower.solve_col(&mut rhs_buf[..nrows]);
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        vals[idx] = rhs_buf[row_idx[idx]];
                    }
                }
            }
        }
        Ok(())
    }
}

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

    fn rank_sub_dense(&mut self, cross: &Array2<f64>, gram: &mut Array2<f64>) {
        debug_assert_eq!(cross.nrows(), self.n);
        let n = self.n;
        if gram.nrows() != n || gram.ncols() != n {
            *gram = Array2::zeros((n, n));
        }
        general_mat_mul(1.0, cross, &cross.t(), 0.0, gram);
        for r in 0..n {
            let base = r * (r + 1) / 2;
            for c in 0..=r {
                self.data[base + c] -= gram[[r, c]];
            }
        }
    }

    fn rank_sub_outer(&mut self, rows: &[usize], vals: &[f64]) {
        for i in 0..vals.len() {
            for j in 0..=i {
                let (r, c) = if rows[i] >= rows[j] {
                    (rows[i], rows[j])
                } else {
                    (rows[j], rows[i])
                };
                self.set(r, c, self.get(r, c) - vals[i] * vals[j]);
            }
        }
    }

    fn chol(&mut self) -> Result<f64, ()> {
        let n = self.n;
        let data = &mut self.data;
        let mut log_det = 0.0;
        for j in 0..n {
            let base_j = j * (j + 1) / 2;
            let mut sum = data[base_j + j];
            for k in 0..j {
                let ljk = data[base_j + k];
                sum -= ljk * ljk;
            }
            if sum <= 0.0 {
                return Err(());
            }
            let ljj = sum.sqrt();
            data[base_j + j] = ljj;
            log_det += ljj.ln();
            for i in (j + 1)..n {
                let base_i = i * (i + 1) / 2;
                let mut s = data[base_i + j];
                for k in 0..j {
                    s -= data[base_i + k] * data[base_j + k];
                }
                data[base_i + j] = s / ljj;
            }
        }
        Ok(log_det)
    }

    fn solve_col(&self, rhs: &mut [f64]) {
        let n = self.n;
        debug_assert_eq!(rhs.len(), n);
        for i in 0..n {
            let mut s = rhs[i];
            let base = i * (i + 1) / 2;
            for (k, &rk) in rhs[..i].iter().enumerate() {
                s -= self.data[base + k] * rk;
            }
            rhs[i] = s / self.data[base + i];
        }
    }

    /// Solve `L X = rhs` in place; `rhs` is `n × nrhs` with one RHS per column.
    fn solve_multi_rhs(&self, rhs: &mut Array2<f64>) {
        let n = self.n;
        debug_assert_eq!(rhs.nrows(), n);
        let nrhs = rhs.ncols();
        let data = &self.data;
        for i in 0..n {
            let base_i = i * (i + 1) / 2;
            let diag = data[base_i + i];
            for k in 0..i {
                let l_ik = data[base_i + k];
                for col in 0..nrhs {
                    rhs[[i, col]] -= l_ik * rhs[[k, col]];
                }
            }
            for col in 0..nrhs {
                rhs[[i, col]] /= diag;
            }
        }
    }
}

/// Cholesky storage for an RE diagonal block after fill-in.
enum ReFactor {
    Full(ReLower),
    /// Nested batch block: rank updates from `batch/cask` cross stay diagonal.
    Diagonal(Vec<f64>),
    /// Nested: each cross column updates an independent diagonal block (casks within a batch).
    #[allow(dead_code)]
    ColumnBlocks {
        blocks: Vec<ReLower>,
        col_rows: Vec<Vec<usize>>,
    },
}

impl ReFactor {
    fn from_diag_and_cross(diag: &[f64], cross: &CrossBlock) -> Self {
        if cross.columns_single_row() {
            ReFactor::Diagonal(diag.to_vec())
        } else {
            ReFactor::Full(ReLower::diagonal_init(diag.len(), diag))
        }
    }

    fn reset_diag(&mut self, diag: &[f64]) {
        match self {
            ReFactor::Full(l) => {
                for (i, &d) in diag.iter().enumerate().take(l.n) {
                    l.set(i, i, d);
                    for j in 0..i {
                        l.set(i, j, 0.0);
                    }
                }
            }
            ReFactor::Diagonal(d) => {
                d.clone_from_slice(diag);
            }
            ReFactor::ColumnBlocks { blocks, col_rows } => {
                for (b, rows) in col_rows.iter().enumerate() {
                    let sub: Vec<f64> = rows.iter().map(|&r| diag[r]).collect();
                    blocks[b] = ReLower::diagonal_init(sub.len(), &sub);
                }
            }
        }
    }

    fn rank_sub_cross(&mut self, cross: &CrossBlock, gram: &mut Array2<f64>) {
        match self {
            ReFactor::Full(l) => cross.rank_sub_lower(l, gram),
            ReFactor::Diagonal(d) => cross.rank_sub_diagonal(d),
            ReFactor::ColumnBlocks { blocks, col_rows } => {
                cross.rank_sub_column_blocks(blocks, col_rows);
            }
        }
    }

    fn chol(&mut self) -> Result<f64, ()> {
        match self {
            ReFactor::Full(l) => l.chol(),
            ReFactor::Diagonal(d) => {
                let mut log_det = 0.0;
                for val in d.iter_mut() {
                    if *val <= 0.0 {
                        return Err(());
                    }
                    *val = val.sqrt();
                    log_det += val.ln();
                }
                Ok(log_det)
            }
            ReFactor::ColumnBlocks { blocks, .. } => {
                let mut log_det = 0.0;
                for b in blocks {
                    log_det += b.chol()?;
                }
                Ok(log_det)
            }
        }
    }

    fn trisolve_cross(&self, cross: &mut CrossBlock, rhs_buf: &mut [f64]) -> Result<(), ()> {
        match self {
            ReFactor::Full(l) => cross.trisolve_full_lower(l, rhs_buf),
            ReFactor::Diagonal(d) => cross.trisolve_single_row_cols(d),
            ReFactor::ColumnBlocks { blocks, col_rows } => {
                for col in 0..cross.ncols() {
                    cross.trisolve_column_blocks(col, blocks, col_rows)?;
                }
                Ok(())
            }
        }
    }

    fn trisolve_xy_block(
        &self,
        block: &mut Array2<f64>,
        m: usize,
        pq: usize,
        scratch: &mut Array2<f64>,
    ) -> Result<(), ()> {
        match self {
            ReFactor::Full(l) => {
                if scratch.dim() != (m, pq) {
                    *scratch = Array2::zeros((m, pq));
                }
                scratch.assign(&block.t());
                l.solve_multi_rhs(scratch);
                block.assign(&scratch.t());
                Ok(())
            }
            ReFactor::Diagonal(d) => {
                for col in 0..m {
                    let div = d[col];
                    if div == 0.0 {
                        return Err(());
                    }
                    for row in 0..pq {
                        block[[row, col]] /= div;
                    }
                }
                Ok(())
            }
            ReFactor::ColumnBlocks { .. } => Err(()),
        }
    }

    fn forward_solve_block(&self, vi: &mut [f64]) -> Result<(), ()> {
        match self {
            ReFactor::Full(l) => {
                l.solve_col(vi);
                Ok(())
            }
            ReFactor::Diagonal(d) => {
                for (val, &div) in vi.iter_mut().zip(d.iter()) {
                    if div == 0.0 {
                        return Err(());
                    }
                    *val /= div;
                }
                Ok(())
            }
            ReFactor::ColumnBlocks { .. } => Err(()),
        }
    }

    fn backward_solve_block(&self, wi: &mut [f64]) -> Result<(), ()> {
        match self {
            ReFactor::Full(l) => {
                let n = l.n;
                for idx in (0..n).rev() {
                    let mut s = wi[idx];
                    let base = idx * (idx + 1) / 2;
                    let diag = l.data[base + idx];
                    if diag == 0.0 {
                        return Err(());
                    }
                    for (k, wk) in wi.iter().enumerate().skip(idx + 1) {
                        let base_k = k * (k + 1) / 2;
                        s -= l.data[base_k + idx] * wk;
                    }
                    wi[idx] = s / diag;
                }
                Ok(())
            }
            ReFactor::Diagonal(d) => {
                for (val, &div) in wi.iter_mut().zip(d.iter()) {
                    if div == 0.0 {
                        return Err(());
                    }
                    *val /= div;
                }
                Ok(())
            }
            ReFactor::ColumnBlocks { .. } => Err(()),
        }
    }
}

pub(crate) struct InterceptBlockedChol {
    k_re: usize,
    p: usize,
    n_re: Vec<usize>,
    /// Global `q` row index where sorted RE block `j` begins in `zt` / `w` vectors.
    block_row_start: Vec<usize>,
    theta_idx: Vec<usize>,
    a_diag: Vec<Vec<f64>>,
    a_cross: Vec<Vec<CrossBlock>>,
    l_cross: Vec<Vec<CrossBlock>>,
    a_xy_re: Vec<Array2<f64>>,
    l_xy_re: Vec<Array2<f64>>,
    a_xy_xy: Array2<f64>,
    l_xy_xy: Array2<f64>,
    l_diag0: Vec<f64>,
    l_re_factor: Vec<ReFactor>,
    diag_buf: Vec<f64>,
    cross_rhs_buf: Vec<f64>,
    schur_li_scratch: Array2<f64>,
    schur_lj_scratch: Array2<f64>,
    rank_gram_buf: Array2<f64>,
    xy_trisolve_buf: Array2<f64>,
}

/// Stable perf-diag label when [`InterceptBlockedChol::try_new`] returns `None`.
pub(crate) fn blocked_unavailable_reason(lmm: &super::LmmData) -> &'static str {
    blocked_gate_failure(lmm).unwrap_or("blocked_unavailable_unknown")
}

pub(crate) fn blocked_gate_failure(lmm: &super::LmmData) -> Option<&'static str> {
    if !lmm.intercept_only_re() {
        return Some("blocked_unavailable_not_intercept_only");
    }
    let re_blocks = &lmm.re_blocks;
    if re_blocks.is_empty() {
        return Some("blocked_unavailable_empty_re");
    }
    let mut order: Vec<usize> = (0..re_blocks.len()).collect();
    order.sort_by(|&a, &b| re_blocks[b].m.cmp(&re_blocks[a].m));
    let mut q = 0usize;
    for &orig in &order {
        let b = &re_blocks[orig];
        if b.k != 1 {
            return Some("blocked_unavailable_slopes");
        }
        q += b.m;
    }
    if q != lmm.zt_z.rows() {
        return Some("blocked_unavailable_dim_mismatch");
    }
    let mut block_row_off = Vec::with_capacity(re_blocks.len());
    let mut off = 0usize;
    for b in re_blocks {
        block_row_off.push(off);
        off += b.m;
    }
    let mut ranges = Vec::with_capacity(order.len());
    for &orig in &order {
        let start = block_row_off[orig];
        ranges.push((start, start + re_blocks[orig].m));
    }
    for j in 1..order.len() {
        for jj in 0..j {
            let block = CrossBlock::from_submatrix(&lmm.zt_z, ranges[j], ranges[jj]);
            if !block.fits_blocked_gate() {
                return Some("blocked_unavailable_cross_gate");
            }
        }
    }
    None
}

impl InterceptBlockedChol {
    pub fn try_new(lmm: &super::LmmData) -> Option<Self> {
        if blocked_gate_failure(lmm).is_some() {
            return None;
        }
        Self::try_new_inner(lmm)
    }

    fn try_new_inner(lmm: &super::LmmData) -> Option<Self> {
        let re_blocks = &lmm.re_blocks;
        let p = lmm.x.ncols();
        let mut order: Vec<usize> = (0..re_blocks.len()).collect();
        order.sort_by(|&a, &b| re_blocks[b].m.cmp(&re_blocks[a].m));

        let k_re = order.len();
        let mut block_row_off = Vec::with_capacity(re_blocks.len());
        let mut off = 0usize;
        for b in re_blocks {
            block_row_off.push(off);
            off += b.m;
        }

        let mut n_re = Vec::with_capacity(k_re);
        let mut theta_idx = Vec::with_capacity(k_re);
        let mut ranges = Vec::with_capacity(k_re);
        let mut block_row_start = Vec::with_capacity(k_re);
        for &orig in &order {
            let b = &re_blocks[orig];
            let start = block_row_off[orig];
            n_re.push(b.m);
            theta_idx.push(orig);
            block_row_start.push(start);
            ranges.push((start, start + b.m));
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
                let block = CrossBlock::from_submatrix(&lmm.zt_z, ranges[j], ranges[jj]);
                debug_assert!(block.fits_blocked_gate());
                l_row.push(block.zeros_like());
                a_row.push(block);
            }
            a_cross.push(a_row);
            l_cross.push(l_row);
        }

        let mut l_re_factor = Vec::with_capacity(k_re.saturating_sub(1));
        for j in 1..k_re {
            let diag: Vec<f64> = a_diag[j].iter().map(|_| 0.0).collect();
            let cross0 = &a_cross[j - 1][0];
            l_re_factor.push(ReFactor::from_diag_and_cross(&diag, cross0));
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

        let n0 = n_re[0];
        let max_m = *n_re.iter().max().unwrap_or(&0);
        let mut max_cross_rows = 0usize;
        let mut max_cross_cols = 0usize;
        for row in &a_cross {
            for block in row {
                max_cross_rows = max_cross_rows.max(block.nrows());
                max_cross_cols = max_cross_cols.max(block.ncols());
            }
        }
        let max_n_re = *n_re.iter().max().unwrap_or(&0);
        Some(Self {
            k_re,
            p,
            n_re,
            block_row_start,
            theta_idx,
            a_diag,
            a_cross,
            l_cross,
            a_xy_re,
            l_xy_re,
            a_xy_xy: a_xy_xy.clone(),
            l_xy_xy: a_xy_xy,
            l_diag0: vec![0.0; n0],
            l_re_factor,
            diag_buf: Vec::new(),
            cross_rhs_buf: vec![0.0; max_cross_rows],
            schur_li_scratch: Array2::zeros((max_cross_rows, max_cross_cols)),
            schur_lj_scratch: Array2::zeros((max_cross_rows, max_cross_cols)),
            rank_gram_buf: Array2::zeros((max_n_re, max_n_re)),
            xy_trisolve_buf: Array2::zeros((max_m, pq)),
        })
    }

    fn cross_to_dense_scratch(block: &CrossBlock, scratch: &mut Array2<f64>) {
        match block {
            CrossBlock::Dense(d) => {
                if scratch.dim() != d.dim() {
                    *scratch = d.clone();
                } else {
                    scratch.assign(d);
                }
            }
            CrossBlock::Sparse {
                nrows,
                ncols,
                col_ptr,
                row_idx,
                vals,
                ..
            } => {
                if scratch.dim() != (*nrows, *ncols) {
                    *scratch = Array2::zeros((*nrows, *ncols));
                } else {
                    scratch.fill(0.0);
                }
                for col in 0..*ncols {
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        scratch[[row_idx[idx], col]] = vals[idx];
                    }
                }
            }
        }
    }

    fn schur_sub_dense_blocks(
        target: &mut CrossBlock,
        li: &Array2<f64>,
        lj: &Array2<f64>,
        product: &Array2<f64>,
    ) {
        debug_assert_eq!(li.dim(), lj.dim());
        match target {
            CrossBlock::Dense(d) => {
                let nrows = d.nrows();
                let subcols = d.ncols().min(product.ncols());
                ndarray::Zip::from(d.slice_mut(ndarray::s![..nrows, 0..subcols]))
                    .and(product.slice(ndarray::s![..nrows, 0..subcols]))
                    .for_each(|dst, &src| *dst -= src);
            }
            CrossBlock::Sparse { .. } => {
                for col in 0..target.ncols() {
                    for row in 0..target.nrows() {
                        let mut s = target.entry(row, col);
                        for k in 0..li.ncols() {
                            let ljv = if col < lj.nrows() { lj[[col, k]] } else { 0.0 };
                            s -= li[[row, k]] * ljv;
                        }
                        Self::set_cross_entry(target, row, col, s);
                    }
                }
            }
        }
    }

    pub fn profile_deviance(&mut self, lmm: &super::LmmData, theta: &[f64], reml: bool) -> f64 {
        match self.update_l_and_factor(theta) {
            Ok(log_det_re) => {
                crate::perf_diag::scope(crate::perf_diag::Phase::BlockedDevianceTail, || {
                    self.deviance_from_factor(lmm, reml, log_det_re)
                })
            }
            Err(()) => f64::MAX,
        }
    }

    fn update_l_and_factor(&mut self, theta: &[f64]) -> Result<f64, ()> {
        let k_re = self.k_re;

        {
            let _phase = crate::perf_diag::PhaseGuard::new(crate::perf_diag::Phase::BlockedReset);
            self.l_xy_xy.assign(&self.a_xy_xy);
            for j in 0..k_re {
                let th = theta[self.theta_idx[j]];
                let th2 = th * th;
                if j == 0 {
                    for i in 0..self.n_re[0] {
                        self.l_diag0[i] = th2 * self.a_diag[0][i] + 1.0;
                    }
                } else {
                    self.diag_buf.clear();
                    self.diag_buf
                        .extend(self.a_diag[j].iter().map(|&a| th2 * a + 1.0));
                    self.l_re_factor[j - 1].reset_diag(&self.diag_buf);
                }
                for i in (j + 1)..k_re {
                    self.l_cross[i - 1][j].assign_scaled(&self.a_cross[i - 1][j], th);
                }
                ndarray::Zip::from(&mut self.l_xy_re[j])
                    .and(&self.a_xy_re[j])
                    .for_each(|l, &a| *l = a * th);
                if j > 0 {
                    for jj in 0..j {
                        self.l_cross[j - 1][jj].scale(th);
                    }
                }
            }
        }

        let mut log_det_re = 0.0;
        let kb = k_re + 1;
        for j in 0..kb {
            if j < k_re {
                {
                    let _phase =
                        crate::perf_diag::PhaseGuard::new(crate::perf_diag::Phase::BlockedRankChol);
                    for jj in 0..j {
                        if j > 0 {
                            self.l_re_factor[j - 1]
                                .rank_sub_cross(&self.l_cross[j - 1][jj], &mut self.rank_gram_buf);
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
                        log_det_re += self.l_re_factor[j - 1].chol()?;
                    }
                }
            } else {
                let _phase =
                    crate::perf_diag::PhaseGuard::new(crate::perf_diag::Phase::BlockedSchurXy);
                for jj in 0..k_re {
                    self.schur_sub_xy_xy(jj)?;
                }
                self.chol_xy_block()?;
            }

            for i in (j + 1)..kb {
                for jj in 0..j {
                    if i < k_re {
                        let _phase = crate::perf_diag::PhaseGuard::new(
                            crate::perf_diag::Phase::BlockedSchurRe,
                        );
                        self.schur_sub_re_cross(i, j, jj)?;
                    } else {
                        let _phase = crate::perf_diag::PhaseGuard::new(
                            crate::perf_diag::Phase::BlockedSchurXy,
                        );
                        self.schur_sub_xy_re(j, jj)?;
                    }
                }
                if i < k_re {
                    let _phase =
                        crate::perf_diag::PhaseGuard::new(crate::perf_diag::Phase::BlockedTrisolve);
                    self.trisolve_re_cross(i, j)?;
                } else {
                    let _phase =
                        crate::perf_diag::PhaseGuard::new(crate::perf_diag::Phase::BlockedTrisolve);
                    self.trisolve_xy_re(j)?;
                }
            }
        }

        Ok(2.0 * log_det_re)
    }

    fn schur_sub_re_cross(&mut self, i: usize, j: usize, jj: usize) -> Result<(), ()> {
        let li_r = i - 1;
        let lj_r = j - 1;
        let sparse_target = matches!(self.l_cross[li_r][j], CrossBlock::Sparse { .. });
        if sparse_target {
            let li = self.l_cross[li_r][jj].clone();
            let lj = if li_r == lj_r {
                li.clone()
            } else {
                self.l_cross[lj_r][jj].clone()
            };
            Self::schur_sub_sparse_at_structure(&mut self.l_cross[li_r][j], &li, &lj);
            return Ok(());
        }
        Self::cross_to_dense_scratch(&self.l_cross[li_r][jj], &mut self.schur_li_scratch);
        if li_r == lj_r {
            self.schur_lj_scratch.assign(&self.schur_li_scratch);
        } else {
            Self::cross_to_dense_scratch(&self.l_cross[lj_r][jj], &mut self.schur_lj_scratch);
        }
        let nrows = self.schur_li_scratch.nrows();
        let lj_rows = self.schur_lj_scratch.nrows();
        if self.rank_gram_buf.nrows() != nrows || self.rank_gram_buf.ncols() != lj_rows {
            self.rank_gram_buf = Array2::zeros((nrows, lj_rows));
        }
        general_mat_mul(
            1.0,
            &self.schur_li_scratch,
            &self.schur_lj_scratch.t(),
            0.0,
            &mut self.rank_gram_buf,
        );
        let target = &mut self.l_cross[li_r][j];
        Self::schur_sub_dense_blocks(
            target,
            &self.schur_li_scratch,
            &self.schur_lj_scratch,
            &self.rank_gram_buf,
        );
        Ok(())
    }

    /// `target -= li * ljᵀ` at structural nonzeros only (nested sparse crosses).
    fn schur_sub_sparse_at_structure(target: &mut CrossBlock, li: &CrossBlock, lj: &CrossBlock) {
        let CrossBlock::Sparse {
            col_ptr,
            row_idx,
            vals,
            ncols,
            ..
        } = target
        else {
            panic!("schur_sub_sparse_at_structure expects sparse target");
        };
        let inner = li.ncols().min(lj.nrows());
        for col in 0..*ncols {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                let row = row_idx[idx];
                let mut s = vals[idx];
                for k in 0..inner {
                    s -= li.entry(row, k) * lj.entry(col, k);
                }
                vals[idx] = s;
            }
        }
    }

    fn set_cross_entry(block: &mut CrossBlock, row: usize, col: usize, val: f64) {
        match block {
            CrossBlock::Dense(d) => d[[row, col]] = val,
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ..
            } => {
                for idx in col_ptr[col]..col_ptr[col + 1] {
                    if row_idx[idx] == row {
                        vals[idx] = val;
                        return;
                    }
                }
                if val != 0.0 {
                    panic!("sparse cross structural nonzero missing at ({row}, {col})");
                }
            }
        }
    }

    fn trisolve_re_cross(&mut self, i: usize, j: usize) -> Result<(), ()> {
        if j == 0 {
            self.l_cross[i - 1][j].trisolve_diag0(&self.l_diag0)?;
            return Ok(());
        }
        self.l_re_factor[j - 1].trisolve_cross(&mut self.l_cross[i - 1][j], &mut self.cross_rhs_buf)
    }

    fn schur_sub_xy_re(&mut self, j: usize, jj: usize) -> Result<(), ()> {
        debug_assert!(jj < j);
        let (xy_lo, xy_hi) = self.l_xy_re.split_at_mut(j);
        let lij = &xy_lo[jj];
        let block = &mut xy_hi[0];
        match &self.l_cross[j - 1][jj] {
            CrossBlock::Dense(ljj) => {
                general_mat_mul(1.0, lij, &ljj.t(), -1.0, block);
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ..
            } => {
                let m = block.ncols();
                for col in 0..m {
                    for row in 0..block.nrows() {
                        let mut s = block[[row, col]];
                        for idx in col_ptr[col]..col_ptr[col + 1] {
                            let k = row_idx[idx];
                            s -= lij[[row, k]] * vals[idx];
                        }
                        block[[row, col]] = s;
                    }
                }
            }
        }
        Ok(())
    }

    fn trisolve_xy_re(&mut self, j: usize) -> Result<(), ()> {
        let m = self.n_re[j];
        let pq = self.p + 1;
        if j == 0 {
            for col in 0..m {
                let d = self.l_diag0[col];
                if d == 0.0 {
                    return Err(());
                }
                for row in 0..pq {
                    self.l_xy_re[j][[row, col]] /= d;
                }
            }
            return Ok(());
        }
        self.l_re_factor[j - 1].trisolve_xy_block(
            &mut self.l_xy_re[j],
            m,
            pq,
            &mut self.xy_trisolve_buf,
        )
    }

    fn schur_sub_xy_xy(&mut self, jj: usize) -> Result<(), ()> {
        let pq = self.p + 1;
        let lxy = &self.l_xy_re[jj];
        let ncol = lxy.ncols();
        for col in 0..pq {
            for row in col..pq {
                let mut s = self.l_xy_xy[[row, col]];
                for k in 0..ncol {
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

    /// Global `zt` row range for sorted RE block `i`.
    fn block_row_range(&self, i: usize) -> std::ops::Range<usize> {
        let start = self.block_row_start[i];
        start..start + self.n_re[i]
    }

    /// `dst[r] -= Σ_c cross[r,c] · src[c]` (`cross` has `nrows = |dst|`, `ncols = |src|`).
    fn cross_matvec_sub(cross: &CrossBlock, src: &[f64], dst: &mut [f64]) {
        match cross {
            CrossBlock::Dense(d) => {
                debug_assert_eq!(d.nrows(), dst.len());
                debug_assert_eq!(d.ncols(), src.len());
                for row in 0..d.nrows() {
                    let mut s = 0.0;
                    for col in 0..d.ncols() {
                        s += d[[row, col]] * src[col];
                    }
                    dst[row] -= s;
                }
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ncols,
                ..
            } => {
                debug_assert_eq!(*ncols, src.len());
                for col in 0..*ncols {
                    let sc = src[col];
                    if sc == 0.0 {
                        continue;
                    }
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        dst[row_idx[idx]] -= vals[idx] * sc;
                    }
                }
            }
        }
    }

    /// `dst[c] -= Σ_r cross[r,c] · src[r]` (`cross` has `nrows = |src|`, `ncols = |dst|`).
    fn cross_matvec_t_sub(cross: &CrossBlock, src: &[f64], dst: &mut [f64]) {
        match cross {
            CrossBlock::Dense(d) => {
                debug_assert_eq!(d.nrows(), src.len());
                debug_assert_eq!(d.ncols(), dst.len());
                for col in 0..d.ncols() {
                    let mut s = 0.0;
                    for row in 0..d.nrows() {
                        s += d[[row, col]] * src[row];
                    }
                    dst[col] -= s;
                }
            }
            CrossBlock::Sparse {
                col_ptr,
                row_idx,
                vals,
                ncols,
                ..
            } => {
                debug_assert_eq!(*ncols, dst.len());
                for col in 0..*ncols {
                    let mut s = 0.0;
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        s += vals[idx] * src[row_idx[idx]];
                    }
                    dst[col] -= s;
                }
            }
        }
    }

    /// `v_i -= cross(i,j) * v_j` with `cross` stored as rows in block `i`, cols in block `j`.
    fn sub_cross_matvec(&self, i: usize, j: usize, vj: &[f64], vi: &mut [f64]) {
        Self::cross_matvec_sub(&self.l_cross[i - 1][j], vj, vi);
    }

    /// Forward substitution: `L z = rhs` (block lower `L` from `update_l_and_factor`).
    fn forward_solve_ld(&self, rhs: &mut [f64]) -> Result<(), ()> {
        for i in 0..self.k_re {
            let range_i = self.block_row_range(i);
            let mut vi = rhs[range_i.clone()].to_vec();
            for j in 0..i {
                let zj = &rhs[self.block_row_range(j)];
                self.sub_cross_matvec(i, j, zj, &mut vi);
            }
            if i == 0 {
                for (val, &d) in vi.iter_mut().zip(self.l_diag0.iter()) {
                    if d == 0.0 {
                        return Err(());
                    }
                    *val /= d;
                }
            } else {
                self.l_re_factor[i - 1].forward_solve_block(&mut vi)?;
            }
            rhs[range_i].copy_from_slice(&vi);
        }
        Ok(())
    }

    /// Back substitution: `Lᵀ w = z`.
    fn backward_solve_ld(&self, z: &mut [f64]) -> Result<(), ()> {
        for i in (0..self.k_re).rev() {
            let range_i = self.block_row_range(i);
            let mut wi = z[range_i.clone()].to_vec();
            for j in (i + 1)..self.k_re {
                let wj = &z[self.block_row_range(j)];
                Self::cross_matvec_t_sub(&self.l_cross[j - 1][i], wj, &mut wi);
            }
            if i == 0 {
                for (val, &d) in wi.iter_mut().zip(self.l_diag0.iter()) {
                    if d == 0.0 {
                        return Err(());
                    }
                    *val /= d;
                }
            } else {
                self.l_re_factor[i - 1].backward_solve_block(&mut wi)?;
            }
            z[range_i].copy_from_slice(&wi);
        }
        Ok(())
    }

    /// `w = A⁻¹ rhs` for the intercept-only `A = Λ⁻¹ + ZᵀZ` matrix at the current factorization.
    pub(crate) fn backsolve_a_inv(&self, rhs: &mut [f64]) -> Result<(), ()> {
        self.forward_solve_ld(rhs)?;
        self.backward_solve_ld(rhs)?;
        Ok(())
    }

    /// Full profile solve reusing the blocked `updateL!` factor (no sparse LDL).
    pub(crate) fn solve_profile_blocked(
        &mut self,
        lmm: &super::LmmData,
        theta: &[f64],
        reml: bool,
        row_block: &[usize],
        w_y: &mut [f64],
        w_col_bufs: &mut [Vec<f64>],
    ) -> Result<super::ProfileSolution, ()> {
        let log_det_a = self.update_l_and_factor(theta)?;
        let q = lmm.zt.rows();
        let p_usize = lmm.x.ncols();
        debug_assert_eq!(w_y.len(), q);
        super::scale_block_rhs_buf(w_y, lmm.zt_y.view(), theta, row_block);
        self.backsolve_a_inv(w_y).map_err(|_| ())?;

        let d = super::theta_diagonal(theta, q, &lmm.re_blocks);
        let v_y = &lmm.zt_y * &d;
        let mut v_cols = Vec::with_capacity(p_usize);
        for (j, buf) in w_col_bufs.iter_mut().enumerate().take(p_usize) {
            let v_j = &lmm.zt_x.column(j) * &d;
            debug_assert_eq!(buf.len(), q);
            super::scale_block_rhs_buf(buf, lmm.zt_x.column(j), theta, row_block);
            self.backsolve_a_inv(buf).map_err(|_| ())?;
            v_cols.push(v_j);
        }
        let w_col_slices: Vec<&[f64]> = w_col_bufs.iter().map(|v| v.as_slice()).collect();
        let n = lmm.y.len() as f64;
        let p = p_usize as f64;
        super::solve_profile_finish(
            lmm,
            super::ProfileFinishInput {
                reml,
                n,
                p,
                p_usize,
                q,
                log_det_a,
                v_y: &v_y,
                w_y,
                v_cols: &v_cols,
                w_cols: &w_col_slices,
            },
            |u| &d * u,
        )
    }
}

impl ReFactor {
    #[allow(dead_code)]
    fn column_block_count(&self) -> usize {
        match self {
            ReFactor::Full(_) | ReFactor::Diagonal(_) => 0,
            ReFactor::ColumnBlocks { blocks, .. } => blocks.len(),
        }
    }

    #[allow(dead_code)]
    fn col_rows(&self, col: usize) -> &[usize] {
        match self {
            ReFactor::Full(_) | ReFactor::Diagonal(_) => &[],
            ReFactor::ColumnBlocks { col_rows, .. } => &col_rows[col],
        }
    }
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

    #[test]
    fn blocked_profile_solve_matches_sparse_evaluate_on_penicillin() {
        let (lmm, golden_theta) = load_penicillin_lmm();
        let mut blocked = InterceptBlockedChol::try_new(&lmm).expect("blocked setup");
        let p = lmm.x.ncols();
        let q = lmm.zt.rows();
        let mut row_block = Vec::with_capacity(q);
        for (bi, block) in lmm.re_blocks.iter().enumerate() {
            for _ in 0..block.m {
                row_block.push(bi);
            }
        }
        let mut w_y = vec![0.0; q];
        let mut w_col_bufs: Vec<Vec<f64>> = (0..p).map(|_| vec![0.0; q]).collect();
        let blocked_sol = blocked
            .solve_profile_blocked(
                &lmm,
                &golden_theta,
                true,
                &row_block,
                &mut w_y,
                &mut w_col_bufs,
            )
            .expect("blocked profile solve");
        let sparse = lmm.evaluate(&golden_theta, true);
        let scale = sparse.reml_crit.abs().max(1.0);
        assert!(
            (blocked_sol.reml_crit - sparse.reml_crit).abs() <= 1e-6 * scale,
            "reml_crit blocked={} sparse={}",
            blocked_sol.reml_crit,
            sparse.reml_crit
        );
        for (i, (&b, &s)) in blocked_sol.beta.iter().zip(sparse.beta.iter()).enumerate() {
            assert!((b - s).abs() <= 1e-8, "beta[{i}] blocked={b} sparse={s}");
        }
        assert!(
            (blocked_sol.sigma2 - sparse.sigma2).abs() <= 1e-8 * sparse.sigma2.abs().max(1.0),
            "sigma2 blocked={} sparse={}",
            blocked_sol.sigma2,
            sparse.sigma2
        );
    }

    #[test]
    fn blocked_profile_solve_matches_sparse_on_crossed_ml_grid() {
        use crate::prepare_lmer;
        use polars::prelude::*;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmark-results/fair-rust-julia-data/crossed_20k.csv"
        );
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut file = std::fs::File::open(path).unwrap();
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(&mut file)
            .finish()
            .unwrap();
        let prep = prepare_lmer("y ~ x + (1 | plate) + (1 | sample)", &df).unwrap();
        let lmm = &prep.lmm;
        let mut blocked = InterceptBlockedChol::try_new(lmm).expect("blocked setup");
        let p = lmm.x.ncols();
        let q = lmm.zt.rows();
        let row_block: Vec<usize> = lmm
            .re_blocks
            .iter()
            .enumerate()
            .flat_map(|(bi, b)| std::iter::repeat_n(bi, b.m))
            .collect();
        let mut w_y = vec![0.0; q];
        let mut w_col_bufs: Vec<Vec<f64>> = (0..p).map(|_| vec![0.0; q]).collect();
        for &t0 in &[0.5, 1.0, 2.0] {
            for &t1 in &[0.5, 1.0, 2.0] {
                let theta = [t0, t1];
                let blocked_sol = blocked
                    .solve_profile_blocked(
                        lmm,
                        &theta,
                        false,
                        &row_block,
                        &mut w_y,
                        &mut w_col_bufs,
                    )
                    .expect("blocked profile solve");
                let sparse = lmm.evaluate(&theta, false);
                let scale = sparse.reml_crit.abs().max(1.0);
                assert!(
                    (blocked_sol.reml_crit - sparse.reml_crit).abs() <= 1e-5 * scale,
                    "theta={theta:?} blocked={} sparse={}",
                    blocked_sol.reml_crit,
                    sparse.reml_crit
                );
            }
        }
    }

    #[test]
    fn nested_sparse_gate_uses_diagonal_batch_factor() {
        use crate::prepare_lmer;
        use polars::prelude::*;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmark-results/fair-rust-julia-data/nested_10k.csv"
        );
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut file = std::fs::File::open(path).unwrap();
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(&mut file)
            .finish()
            .unwrap();
        let prep = prepare_lmer("y ~ x + (1 | batch/cask)", &df).unwrap();
        assert!(
            prep.blocked_kernel,
            "nested_10k should use blocked kernel (detail: {})",
            prep.blocked_kernel_detail
        );
        assert_eq!(prep.blocked_kernel_detail, "blocked_active");
    }
}
