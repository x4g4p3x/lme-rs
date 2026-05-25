//! Kenward–Roger adjusted covariance (`pbkrtest::vcovAdj16` / `vcovAdj16_internal`).

use ndarray::Array2;
use ndarray_linalg::{Cholesky, Inverse, UPLO};

use crate::formula::parse;
use crate::kr_modcomp::KenwardRogerModcompData;
use crate::math::LmmData;
use crate::model_matrix::build_design_matrices;
use crate::{LmeError, LmeFit};

/// Build `PhiA`, `P`, and `W` via the pbkrtest `vcovAdj16` recipe (structural `G` matrices).
pub fn kenward_roger_modcomp_data(
    fit: &LmeFit,
    data: &polars::prelude::DataFrame,
) -> crate::Result<KenwardRogerModcompData> {
    let theta = fit.theta.as_ref().ok_or_else(|| LmeError::NotImplemented {
        feature: "Theta required for Kenward-Roger vcovAdj.".to_string(),
    })?;
    let formula_str = fit
        .formula
        .as_ref()
        .ok_or_else(|| LmeError::NotImplemented {
            feature: "Formula missing for Kenward-Roger vcovAdj.".to_string(),
        })?;
    let sigma2 = fit.sigma2.ok_or_else(|| LmeError::NotImplemented {
        feature: "sigma2 missing for Kenward-Roger vcovAdj.".to_string(),
    })?;

    let ast = parse(formula_str)?;
    let matrices = build_design_matrices(&ast, data)?;
    let lmm = LmmData::new(
        matrices.x.clone(),
        matrices.zt.clone(),
        matrices.y.clone(),
        matrices.re_blocks.clone(),
    );

    let reml = fit.reml.is_some();
    let base = lmm.evaluate(theta.as_slice().unwrap(), reml);
    let inv_lx = base.l_x.inv().map_err(|e| LmeError::NotImplemented {
        feature: format!("Failed to invert L_x for vcovAdj: {}", e),
    })?;
    let phi = inv_lx.t().dot(&inv_lx) * sigma2;

    let n = lmm.y.len();
    let (g_list, ggamma) = build_sigma_g(&lmm, theta.view(), sigma2)?;
    let (phi_a, p_list, w) = vcov_adj16_with_pw(&phi, &matrices.x, &g_list, &ggamma, n)?;

    Ok(KenwardRogerModcompData {
        phi,
        phi_a,
        p_list,
        w,
    })
}

fn build_sigma_g(
    lmm: &LmmData,
    theta: ndarray::ArrayView1<f64>,
    sigma2: f64,
) -> crate::Result<(Vec<Array2<f64>>, Vec<f64>)> {
    let n = lmm.y.len();
    let q = lmm.zt.rows();
    let mut z = Array2::<f64>::zeros((n, q));
    for (val, (row, col)) in lmm.zt.iter() {
        z[[col, row]] = *val;
    }

    let mut g_list: Vec<Array2<f64>> = Vec::new();
    let mut ggamma: Vec<f64> = Vec::new();

    let mut row_offset = 0usize;
    let mut theta_offset = 0usize;

    for block in &lmm.re_blocks {
        let m = block.m;
        let k = block.k;
        let q_block = m * k;

        // pbkrtest `ggamma` uses `VarCorr` lower-tri entries (= λ λᵀ σ²), not raw θ².
        let th = theta
            .as_slice()
            .unwrap()[theta_offset..theta_offset + block.theta_len]
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let mut lambda = Array2::<f64>::zeros((k, k));
        let mut idx = 0usize;
        for j in 0..k {
            for i in j..k {
                lambda[[i, j]] = th[idx];
                idx += 1;
            }
        }
        let cov = lambda.dot(&lambda.t()) * sigma2;

        let z_block = z.slice(ndarray::s![.., row_offset..row_offset + q_block]);

        for param_idx in 0..block.theta_len {
            let ee = chol_basis_matrix(k, param_idx);
            let mut g = Array2::<f64>::zeros((n, n));
            for group in 0..m {
                let col_start = group * k;
                let z_g = z_block.slice(ndarray::s![.., col_start..col_start + k]);
                let u = z_g.dot(&ee);
                g = g + &u.dot(&z_g.t());
            }
            // Covariance entry matching the Cholesky basis direction (lower-tri order).
            let mut gi = 0usize;
            let mut gj = 0usize;
            'find_ij: for j in 0..k {
                for i in j..k {
                    if ee[[i, j]].abs() > 0.5 {
                        gi = i;
                        gj = j;
                        break 'find_ij;
                    }
                }
            }
            ggamma.push(cov[[gi, gj]]);
            g_list.push(g);
        }
        row_offset += q_block;
        theta_offset += block.theta_len;
    }

    let mut identity = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = 1.0;
    }
    ggamma.push(sigma2);
    g_list.push(identity);

    Ok((g_list, ggamma))
}

/// Lower-triangular Cholesky basis matrix with a single non-zero at `param_idx`.
fn chol_basis_matrix(k: usize, param_idx: usize) -> Array2<f64> {
    let mut ee = Array2::<f64>::zeros((k, k));
    let mut idx = 0usize;
    for j in 0..k {
        for i in j..k {
            if idx == param_idx {
                ee[[i, j]] = 1.0;
                if i != j {
                    ee[[j, i]] = 1.0;
                }
                return ee;
            }
            idx += 1;
        }
    }
    ee
}

/// Upper-triangular index into packed symmetric `n × n` storage (0-based).
fn index_symmat2vec(ii: usize, jj: usize, n: usize) -> usize {
    let (a, b) = if ii <= jj { (ii, jj) } else { (jj, ii) };
    a * n - a * (a + 1) / 2 + b
}

fn vcov_adj16_with_pw(
    phi: &Array2<f64>,
    x: &Array2<f64>,
    g_list: &[Array2<f64>],
    ggamma: &[f64],
    _n_obs: usize,
) -> crate::Result<(Array2<f64>, Vec<Array2<f64>>, Array2<f64>)> {
    let n_g = g_list.len();
    let mut sigma = ggamma[0] * g_list[0].clone();
    for i in 1..n_g {
        sigma = sigma + &(ggamma[i] * &g_list[i]);
    }

    let sigma_chol = sigma
        .cholesky(UPLO::Lower)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Sigma Cholesky failed in vcovAdj16: {}", e),
        })?;
    // `chol2inv(chol(Sigma))` in R: Σ⁻¹ = L⁻ᵀ L⁻¹, not the inverse of the Cholesky factor alone.
    let l_inv = sigma_chol.inv().map_err(|e| LmeError::NotImplemented {
        feature: format!("Cholesky factor inverse failed in vcovAdj16: {}", e),
    })?;
    let sigma_inv = l_inv.t().dot(&l_inv);

    let tt = sigma_inv.dot(x);
    let mut hh = Vec::with_capacity(n_g);
    let mut oo = Vec::with_capacity(n_g);
    for g in g_list {
        let h = g.dot(&sigma_inv);
        hh.push(h.clone());
        oo.push(h.dot(x));
    }

    let mut pp = Vec::with_capacity(n_g);
    let mut qq = Vec::new();
    for r in 0..n_g {
        let oot = oo[r].t();
        pp.push(-1.0 * oot.dot(&tt));
        for s in r..n_g {
            qq.push(oot.dot(&sigma_inv).dot(&oo[s]));
        }
    }

    let mut ktrace = Array2::<f64>::zeros((n_g, n_g));
    for r in 0..n_g {
        let hr = hh[r].t();
        for s in r..n_g {
            let tr: f64 = hr.iter().zip(hh[s].iter()).map(|(&a, &b)| a * b).sum();
            ktrace[[r, s]] = tr;
            ktrace[[s, r]] = tr;
        }
    }

    let mut ie2 = Array2::<f64>::zeros((n_g, n_g));
    for ii in 0..n_g {
        let phi_p_ii = phi.dot(&pp[ii]);
        for jj in ii..n_g {
            let www = index_symmat2vec(ii, jj, n_g);
            let term = ktrace[[ii, jj]]
                - 2.0 * sum_elementwise(phi, &qq[www])
                + sum_elementwise(&phi_p_ii, &(pp[jj].dot(phi)));
            ie2[[ii, jj]] = term;
            ie2[[jj, ii]] = term;
        }
    }

    let w = {
        use ndarray_linalg::Eigh;
        let eval = ie2
            .eigh(UPLO::Upper)
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("IE2 eigen failed: {}", e),
            })?
            .0;
        let condi = eval.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
        if condi > 1e-10 {
            2.0 * ie2.inv().map_err(|e| LmeError::NotImplemented {
                feature: format!("IE2 inverse failed: {}", e),
            })?
        } else {
            let mut ie2_r = ie2.clone();
            for i in 0..n_g {
                ie2_r[[i, i]] += 1e-8;
            }
            2.0 * ie2_r.inv().map_err(|e| LmeError::NotImplemented {
                feature: format!("IE2 ridge inverse failed: {}", e),
            })?
        }
    };

    let p = x.ncols();
    let mut uu = Array2::<f64>::zeros((p, p));
    for ii in 0..n_g {
        for jj in (ii + 1)..n_g {
            let www = index_symmat2vec(ii, jj, n_g);
            let term = &qq[www] - &pp[ii].dot(phi).dot(&pp[jj]);
            uu = uu + &(term * w[[ii, jj]]);
        }
    }
    let uu_sym = &uu + &uu.t();
    uu = uu_sym;
    for ii in 0..n_g {
        let www = index_symmat2vec(ii, ii, n_g);
        let term = &qq[www] - &pp[ii].dot(phi).dot(&pp[ii]);
        uu = uu + &(term * w[[ii, ii]]);
    }

    let gamma = phi.dot(&uu).dot(phi);
    let phi_a = phi + &(gamma * 2.0);

    Ok((phi_a, pp, w))
}

fn sum_elementwise(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod vcov_adj_tests {
    use super::*;
    use crate::kr_modcomp::phi_a_near_phi;
    use crate::lmer;
    use polars::prelude::*;
    use std::fs::File;

    #[test]
    fn pastes_vcov_adj_equals_vcov() {
        let mut file = File::open("tests/data/pastes.csv").unwrap();
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(&mut file)
            .finish()
            .unwrap();
        let fit = lmer("strength ~ cask + (1 | batch)", &df, true).unwrap();
        let data = kenward_roger_modcomp_data(&fit, &df).unwrap();
        assert!(phi_a_near_phi(&data.phi, &data.phi_a, 1e-8));
    }
}
