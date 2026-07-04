//! Population and conditional prediction for nonlinear mixed models.

use crate::nlmm::fit::{column_f64, column_str};
use crate::nlmm::formula::{re_param_indices, NlmerFormula};
use crate::nlmm::mean_fn::{eval_mean_with_re, NlmmMeanEval};
use crate::LmeFit;
use ndarray::Array1;
use polars::prelude::DataFrame;
use std::sync::Arc;

type NlmmPredictState = (NlmerFormula, Arc<dyn NlmmMeanEval>, Vec<f64>, Vec<usize>);

/// Population-level predictions (`re.form = NA`): nonlinear mean with fixed effects only.
pub fn predict_population(fit: &LmeFit, newdata: &DataFrame) -> anyhow::Result<Array1<f64>> {
    let (parsed, mean, params, re_indices) = nlmm_state(fit)?;
    let cov = column_f64(newdata, &parsed.covariate).map_err(|e| anyhow::anyhow!("{e}"))?;
    let n = cov.len();
    let zeros = vec![0.0; re_indices.len()];
    let mut mu = Array1::<f64>::zeros(n);
    for i in 0..n {
        mu[i] = eval_mean_with_re(mean.as_ref(), cov[i], &params, &re_indices, &zeros).0;
    }
    Ok(mu)
}

/// Conditional predictions (`re.form = NULL`): add stored random effects on RE parameters.
pub fn predict_conditional(
    fit: &LmeFit,
    newdata: &DataFrame,
    allow_new_levels: bool,
) -> anyhow::Result<Array1<f64>> {
    let (parsed, mean, params, re_indices) = nlmm_state(fit)?;
    let b = fit.b.as_ref().ok_or_else(|| {
        anyhow::anyhow!("No random effects available for conditional nlmm predictions")
    })?;
    let re_blocks = fit
        .re_blocks
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No RE block metadata available for nlmm predictions"))?;
    let block = re_blocks
        .first()
        .ok_or_else(|| anyhow::anyhow!("No RE blocks stored on nlmm fit"))?;
    let k_re = block.k;

    let cov = column_f64(newdata, &parsed.covariate).map_err(|e| anyhow::anyhow!("{e}"))?;
    let groups = column_str(newdata, &block.group_name).map_err(|e| anyhow::anyhow!("{e}"))?;
    if groups.len() != cov.len() {
        return Err(anyhow::anyhow!(
            "Grouping column '{}' length ({}) does not match covariate length ({})",
            block.group_name,
            groups.len(),
            cov.len()
        ));
    }

    let n = cov.len();
    let mut mu = Array1::<f64>::zeros(n);
    for i in 0..n {
        let re_off = match block.group_map.get(&groups[i]) {
            Some(&idx) => (0..k_re).map(|r| b[idx * k_re + r]).collect(),
            None => {
                if !allow_new_levels {
                    return Err(anyhow::anyhow!(
                        "New level '{}' found in grouping factor '{}', but allow_new_levels is false",
                        groups[i],
                        block.group_name
                    ));
                }
                vec![0.0; k_re]
            }
        };
        mu[i] = eval_mean_with_re(mean.as_ref(), cov[i], &params, &re_indices, &re_off).0;
    }
    Ok(mu)
}

fn nlmm_state(fit: &LmeFit) -> anyhow::Result<NlmmPredictState> {
    let mean = fit
        .nlmm_mean
        .clone()
        .ok_or_else(|| anyhow::anyhow!("No nonlinear mean stored on nlmm fit"))?;
    let parsed = fit
        .nlmm_formula
        .clone()
        .ok_or_else(|| anyhow::anyhow!("No nlmm formula metadata stored on fit"))?;
    let names = fit
        .fixed_names
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No fixed-effect names stored on nlmm fit"))?;
    let coef = fit
        .coefficients
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("nlmm coefficients must be a contiguous vector"))?;
    let mut params = Vec::with_capacity(names.len());
    for name in &parsed.fixed_param_names {
        let idx = names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| anyhow::anyhow!("Missing fixed parameter '{name}' in fitted model"))?;
        params.push(coef[idx]);
    }
    let re_indices = re_param_indices(&parsed).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((parsed, mean, params, re_indices))
}
