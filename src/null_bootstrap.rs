//! Unconditional parametric calibration of nested fixed-effect LMM tests.
//!
//! Draw y = X beta_null + offset + Z Lambda u + epsilon, with fresh independent
//! u ~ N(0, sigma² I) and epsilon ~ N(0, sigma² W^-1). Both models are refit by ML.

use crate::{FitControl, LmeError, LmeFit, LmerPrepared, Result};
use ndarray::Array1;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Auditable result of an unconditional null-model likelihood-ratio bootstrap.
#[derive(Debug, Clone)]
pub struct NullBootstrapResult {
    /// Observed ML likelihood-ratio statistic.
    pub observed: f64,
    /// Requested number of simulations.
    pub requested: usize,
    /// Number of converged, finite refit pairs.
    pub valid: usize,
    /// Valid statistics at least as large as the observed statistic.
    pub exceedances: usize,
    /// (1 + exceedances) / (1 + valid); absent when every refit failed.
    pub p_value: Option<f64>,
    /// Plug-in Monte Carlo standard error; absent without valid refits.
    pub mc_se: Option<f64>,
    /// Reproducibility seed, shared across worker counts.
    pub seed: u64,
    /// Statistic for each requested replicate, in original simulation order.
    pub statistics: Vec<Option<f64>>,
    /// Error for each failed replicate; failures are never counted as successes.
    pub errors: Vec<Option<String>>,
}

fn invalid(message: impl Into<String>) -> LmeError {
    LmeError::InvalidInput {
        message: message.into(),
    }
}

fn statistic(full: &LmeFit, null: &LmeFit) -> Result<f64> {
    full.ensure_converged()?;
    null.ensure_converged()?;
    let a = null
        .deviance
        .ok_or_else(|| invalid("Null deviance missing"))?;
    let b = full
        .deviance
        .ok_or_else(|| invalid("Full deviance missing"))?;
    let lr = a - b;
    if !lr.is_finite() || lr < -1e-7 * (1.0 + a.abs() + b.abs()) {
        return Err(invalid(
            "Nonfinite LRT or full-model likelihood below the nested null",
        ));
    }
    Ok(lr.max(0.0))
}

fn validate_models(full: &LmerPrepared, null: &LmerPrepared) -> Result<()> {
    let a = &full.lmm;
    let b = &null.lmm;
    if a.y != b.y
        || full.matrices.offset != null.matrices.offset
        || a.weights != b.weights
        || a.zt != b.zt
        || a.re_blocks != b.re_blocks
    {
        return Err(invalid("Models must use the same ordered observations, response, offsets, weights, and random-effect structure"));
    }
    if a.x.ncols() <= b.x.ncols() || a.re_blocks.is_empty() {
        return Err(invalid(
            "Full model must add fixed-effect dimensions to a nested LMM null",
        ));
    }
    // QR projection tests the actual column spaces, independently of coding/names.
    for column in b.x.columns() {
        let projection = crate::lm(&column.to_owned(), &a.x)?;
        let error = projection.residuals.dot(&projection.residuals).sqrt();
        if error > 1e-9 * column.dot(&column).sqrt().max(1.0) {
            return Err(invalid(
                "Null fixed-effect design is not nested in the full design",
            ));
        }
    }
    Ok(())
}

fn draw(prepared: &LmerPrepared, fit: &LmeFit, seed: u64) -> Result<Array1<f64>> {
    let sigma = fit
        .sigma2
        .filter(|s| s.is_finite() && *s >= 0.0)
        .ok_or_else(|| invalid("Null fit requires finite nonnegative residual variance"))?
        .sqrt();
    let theta = fit
        .theta
        .as_ref()
        .ok_or_else(|| invalid("Null covariance parameters missing"))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut effects = Array1::<f64>::zeros(prepared.lmm.zt.rows());
    let mut offset = 0;
    let mut theta_offset = 0;
    for block in &prepared.lmm.re_blocks {
        for group in 0..block.m {
            let u: Vec<f64> = (0..block.k)
                .map(|_| rng.sample::<f64, _>(StandardNormal) * sigma)
                .collect();
            let mut index = theta_offset;
            for (j, &value) in u.iter().enumerate() {
                for i in j..block.k {
                    effects[offset + group * block.k + i] += theta[index] * value;
                    index += 1;
                }
            }
        }
        offset += block.m * block.k;
        theta_offset += block.theta_len;
    }
    let mut y = prepared.lmm.x.dot(&fit.coefficients);
    if let Some(offset) = &prepared.matrices.offset {
        y += offset;
    }
    for (effect, row) in prepared.lmm.zt.outer_iterator().enumerate() {
        for (obs, &z) in row.iter() {
            y[obs] += z * effects[effect];
        }
    }
    for (i, value) in y.iter_mut().enumerate() {
        let weight = prepared.lmm.weights.as_ref().map_or(1.0, |w| w[i]);
        *value += rng.sample::<f64, _>(StandardNormal) * sigma / weight.sqrt();
    }
    Ok(y)
}

/// Bootstrap a nested fixed-effects LRT, with fresh random effects under the null.
///
/// Both observed models and every replicate are fit by ML. `full` and `null` must
/// be prepared on the identical row order and random-effect structure. Missing
/// model values must be handled before preparing either design. Seed i is
/// `seed.wrapping_add(i)`; results are independent of `n_jobs`. Failures are retained
/// and the p-value uses valid pairs only: inspect `valid` before interpreting it.
pub fn bootstrap_lrt(
    full: &LmerPrepared,
    null: &LmerPrepared,
    nsim: usize,
    seed: u64,
    n_jobs: usize,
    control: &FitControl,
) -> Result<NullBootstrapResult> {
    if nsim == 0 || n_jobs == 0 {
        return Err(invalid("nsim and n_jobs must be positive"));
    }
    validate_models(full, null)?;
    let mut strict = control.clone();
    strict.require_convergence = true;
    let observed_full = full.fit(None, false, &strict)?;
    let observed_null = null.fit(None, false, &strict)?;
    let observed = statistic(&observed_full, &observed_null)?;
    let pool = crate::execution::ExecutionContext::new(n_jobs.min(nsim))?;
    let results: Vec<_> = pool.install(|| {
        (0..nsim)
            .into_par_iter()
            .map_init(
                || (full.workspace(), null.workspace()),
                |(full_worker, null_worker), i| {
                    let y = draw(null, &observed_null, seed.wrapping_add(i as u64))?;
                    let null_fit = null_worker.fit_response(y.clone(), false, &strict)?;
                    let full_fit = full_worker.fit_response(y, false, &strict)?;
                    statistic(&full_fit, &null_fit)
                },
            )
            .collect()
    });
    let statistics: Vec<_> = results.iter().map(|r| r.as_ref().ok().copied()).collect();
    let errors = results
        .into_iter()
        .map(|r| r.err().map(|e| e.to_string()))
        .collect();
    let valid = statistics.iter().flatten().count();
    let exceedances = statistics
        .iter()
        .flatten()
        .filter(|&&s| s >= observed)
        .count();
    let p_value = (valid > 0).then(|| (1 + exceedances) as f64 / (1 + valid) as f64);
    let mc_se = p_value.map(|p| (p * (1.0 - p) / valid as f64).sqrt());
    Ok(NullBootstrapResult {
        observed,
        requested: nsim,
        valid,
        exceedances,
        p_value,
        mc_se,
        seed,
        statistics,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn fresh_effects_have_expected_covariance_with_weights_and_offsets() {
        let data = df!("y" => [1., 2., 3., 4., 5., 7., 8., 9.],
            "off" => [0., 1., 2., 3., 4., 5., 6., 7.],
            "id" => ["a", "a", "b", "b", "c", "c", "d", "d"])
        .unwrap();
        let prepared = crate::prepare_lmer_weighted(
            "y ~ 1 + offset(off) + (1|id)",
            &data,
            Some(Array1::from(vec![2.; 8])),
        )
        .unwrap();
        let mut fit = prepared.fit(None, false, &FitControl::default()).unwrap();
        fit.coefficients[0] = 10.0;
        fit.theta = Some(Array1::from(vec![2.0]));
        fit.sigma2 = Some(3.0);
        // Conditional fitted values deliberately disagree: draws must ignore them.
        fit.fitted.fill(-1000.0);
        let n = 12_000;
        let mut sum = [0.; 3];
        let mut products = [[0.; 3]; 3];
        for seed in 0..n {
            let y = draw(&prepared, &fit, seed as u64).unwrap();
            for i in 0..3 {
                let vi = y[i] - 10.0 - i as f64;
                sum[i] += vi / n as f64;
                for j in 0..3 {
                    products[i][j] += vi * (y[j] - 10.0 - j as f64) / n as f64;
                }
            }
        }
        for mean in sum {
            assert!(mean.abs() < 0.12, "mean {mean}");
        }
        // sigma² theta² + sigma² / weight = 13.5; shared-group covariance = 12.
        assert!((products[0][0] - 13.5).abs() < 0.6);
        assert!((products[0][1] - 12.0).abs() < 0.6);
        assert!(products[0][2].abs() < 0.6);
    }
}
