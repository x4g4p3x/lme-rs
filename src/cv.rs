//! Group-structure-preserving cross-validation for linear mixed models.
//!
//! Splits by grouping units (e.g. subjects) so all observations from one unit stay
//! in train or test together. Fits on each train fold and evaluates population-level
//! predictions on held-out groups.

use std::collections::HashSet;

use ndarray::Array1;
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::formula::parse;
use crate::{fit_prepared, prepare_lmer, LmeError, LmeFit, Result};

/// Per-fold metrics from [`cv_grouped`].
#[derive(Debug, Clone)]
pub struct CvFoldMetric {
    /// Zero-based fold index.
    pub fold: usize,
    /// Number of grouping units in the training set.
    pub n_train_groups: usize,
    /// Number of grouping units in the test set.
    pub n_test_groups: usize,
    /// Training observations.
    pub n_train_obs: usize,
    /// Test observations.
    pub n_test_obs: usize,
    /// Root mean squared error on the test fold.
    pub rmse: f64,
    /// Mean absolute error on the test fold.
    pub mae: f64,
    /// Whether the training fit reported optimizer convergence.
    pub converged: bool,
}

/// Out-of-fold predictions and summary metrics from [`cv_grouped`].
#[derive(Debug, Clone)]
pub struct CvGroupedResult {
    /// Population-level out-of-fold predictions (one per observation).
    pub oof_predictions: Array1<f64>,
    /// Test-fold index for each observation (0 .. `n_splits` - 1).
    pub test_fold: Array1<i32>,
    /// Overall RMSE across all out-of-fold predictions.
    pub rmse: f64,
    /// Overall MAE across all out-of-fold predictions.
    pub mae: f64,
    /// Per-fold breakdown.
    pub folds: Vec<CvFoldMetric>,
    /// True when every fold fit converged.
    pub all_converged: bool,
    /// Number of CV folds used.
    pub n_splits: usize,
    /// Grouping column used for splitting.
    pub group_col: String,
}

/// Grouped k-fold cross-validation for LMMs.
///
/// Observations sharing a grouping level (e.g. the same `Subject`) are kept
/// entirely in train or test. Each held-out group is predicted with
/// population-level fixed-effects predictions from a model fit on the
/// remaining groups.
///
/// # Arguments
/// * `formula_str` - Wilkinson formula (LMM only).
/// * `data` - Full dataset.
/// * `group_col` - Column whose levels define CV folds (must appear in the formula).
/// * `n_splits` - Number of folds (capped at the number of unique groups).
/// * `reml` - Use REML (`true`) or ML (`false`) when fitting each training fold.
/// * `seed` - Optional RNG seed for reproducible group shuffling.
pub fn cv_grouped(
    formula_str: &str,
    data: &DataFrame,
    group_col: &str,
    n_splits: usize,
    reml: bool,
    seed: Option<u64>,
) -> Result<CvGroupedResult> {
    if formula_str.trim().is_empty() {
        return Err(LmeError::EmptyFormula);
    }
    if n_splits < 2 {
        return Err(LmeError::NotImplemented {
            feature: "cv_grouped requires n_splits >= 2".to_string(),
        });
    }
    if data.column(group_col).is_err() {
        return Err(LmeError::NotImplemented {
            feature: format!("Grouping column '{group_col}' not found in data"),
        });
    }

    let response_col = response_column_name(formula_str)?;
    let mut groups = unique_group_labels(data, group_col)?;
    if groups.len() < n_splits {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "cv_grouped: n_splits ({n_splits}) exceeds number of unique groups ({})",
                groups.len()
            ),
        });
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };
    groups.shuffle(&mut rng);

    let n_obs = data.height();
    let mut oof = Array1::<f64>::from_elem(n_obs, f64::NAN);
    let mut test_fold = Array1::<i32>::from_elem(n_obs, -1);
    let y_all = column_to_f64_vec(data.column(&response_col).map_err(|e| {
        LmeError::NotImplemented {
            feature: format!("Response column '{response_col}' not found: {e}"),
        }
    })?)?;

    let chunk = groups.len().div_ceil(n_splits);
    let mut folds = Vec::with_capacity(n_splits);
    let mut all_converged = true;

    for fold in 0..n_splits {
        let start = fold * chunk;
        if start >= groups.len() {
            break;
        }
        let end = ((fold + 1) * chunk).min(groups.len());
        let test_set: HashSet<&str> = groups[start..end].iter().map(|s| s.as_str()).collect();
        let train_set: HashSet<&str> = groups
            .iter()
            .filter(|g| !test_set.contains(g.as_str()))
            .map(|s| s.as_str())
            .collect();

        let train_df = filter_by_groups(data, group_col, &train_set)?;
        let test_df = filter_by_groups(data, group_col, &test_set)?;
        if train_df.height() == 0 || test_df.height() == 0 {
            return Err(LmeError::NotImplemented {
                feature: format!("cv_grouped fold {fold} produced empty train or test split"),
            });
        }

        let prepared = prepare_lmer(formula_str, &train_df)?;
        let fit = fit_prepared(&prepared, reml)?;
        let preds = fit.predict(&test_df).map_err(|e| LmeError::NotImplemented {
            feature: format!("cv_grouped prediction failed on fold {fold}: {e}"),
        })?;

        let test_indices = row_indices_for_groups(data, group_col, &test_set)?;
        let mut fold_sq = 0.0;
        let mut fold_abs = 0.0;
        for (j, &row) in test_indices.iter().enumerate() {
            oof[row] = preds[j];
            test_fold[row] = fold as i32;
            let err = y_all[row] - preds[j];
            fold_sq += err * err;
            fold_abs += err.abs();
        }
        let n_test = test_indices.len() as f64;
        let fold_rmse = (fold_sq / n_test).sqrt();
        let fold_mae = fold_abs / n_test;
        let converged = fit.converged.unwrap_or(false);
        all_converged &= converged;

        folds.push(CvFoldMetric {
            fold,
            n_train_groups: train_set.len(),
            n_test_groups: test_set.len(),
            n_train_obs: train_df.height(),
            n_test_obs: test_df.height(),
            rmse: fold_rmse,
            mae: fold_mae,
            converged,
        });
    }

    let mut total_sq = 0.0;
    let mut total_abs = 0.0;
    let mut n_pred = 0.0;
    for i in 0..n_obs {
        if test_fold[i] >= 0 {
            let err = y_all[i] - oof[i];
            total_sq += err * err;
            total_abs += err.abs();
            n_pred += 1.0;
        }
    }
    let rmse = (total_sq / n_pred).sqrt();
    let mae = total_abs / n_pred;

    Ok(CvGroupedResult {
        oof_predictions: oof,
        test_fold,
        rmse,
        mae,
        folds,
        all_converged,
        n_splits,
        group_col: group_col.to_string(),
    })
}

/// Refit the same formula and data with a new REML/ML setting.
///
/// Convenience wrapper around [`fit_prepared`] after [`prepare_lmer`].
pub fn refit_lmer(formula_str: &str, data: &DataFrame, reml: bool) -> Result<LmeFit> {
    let prepared = prepare_lmer(formula_str, data)?;
    fit_prepared(&prepared, reml)
}

fn response_column_name(formula_str: &str) -> Result<String> {
    let ast = parse(formula_str)?;
    for (name, info) in &ast.columns {
        if info.roles.contains(&"Response".to_string()) {
            return Ok(name.clone());
        }
    }
    Err(LmeError::NotImplemented {
        feature: "Formula has no response column".to_string(),
    })
}

fn unique_group_labels(df: &DataFrame, group_col: &str) -> Result<Vec<String>> {
    let col = df.column(group_col).map_err(|e| LmeError::NotImplemented {
        feature: format!("Grouping column '{group_col}': {e}"),
    })?;
    let str_col = col
        .cast(&DataType::String)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Grouping column '{group_col}' could not be cast to string: {e}"),
        })?;
    let strs = str_col.str().map_err(|e| LmeError::NotImplemented {
        feature: format!("Grouping column '{group_col}' is not string-like: {e}"),
    })?;
    let mut labels: Vec<String> = strs
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    labels.sort();
    Ok(labels)
}

fn filter_by_groups(
    df: &DataFrame,
    group_col: &str,
    allowed: &HashSet<&str>,
) -> Result<DataFrame> {
    let mask = group_mask(df, group_col, allowed)?;
    df.filter(&mask).map_err(|e| LmeError::NotImplemented {
        feature: format!("Failed to filter data by groups: {e}"),
    })
}

fn row_indices_for_groups(
    df: &DataFrame,
    group_col: &str,
    allowed: &HashSet<&str>,
) -> Result<Vec<usize>> {
    let col = df.column(group_col).map_err(|e| LmeError::NotImplemented {
        feature: format!("Grouping column '{group_col}': {e}"),
    })?;
    let str_col = col
        .cast(&DataType::String)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Grouping column '{group_col}' could not be cast to string: {e}"),
        })?;
    let strs = str_col.str().map_err(|e| LmeError::NotImplemented {
        feature: format!("Grouping column '{group_col}' is not string-like: {e}"),
    })?;
    Ok(strs
        .into_no_null_iter()
        .enumerate()
        .filter_map(|(i, g)| allowed.contains(g).then_some(i))
        .collect())
}

fn group_mask(
    df: &DataFrame,
    group_col: &str,
    allowed: &HashSet<&str>,
) -> Result<BooleanChunked> {
    let col = df.column(group_col).map_err(|e| LmeError::NotImplemented {
        feature: format!("Grouping column '{group_col}': {e}"),
    })?;
    let str_col = col
        .cast(&DataType::String)
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Grouping column '{group_col}' could not be cast to string: {e}"),
        })?;
    let strs = str_col.str().map_err(|e| LmeError::NotImplemented {
        feature: format!("Grouping column '{group_col}' is not string-like: {e}"),
    })?;
    Ok(strs
        .into_iter()
        .map(|opt| opt.map(|s| allowed.contains(s)).unwrap_or(false))
        .collect())
}

fn column_to_f64_vec(col: &Column) -> Result<Vec<f64>> {
    if let Ok(f) = col.f64() {
        return Ok(f.into_no_null_iter().collect());
    }
    if let Ok(i) = col.i64() {
        return Ok(i.into_no_null_iter().map(|v| v as f64).collect());
    }
    Err(LmeError::NotImplemented {
        feature: "Response column must be numeric".to_string(),
    })
}
