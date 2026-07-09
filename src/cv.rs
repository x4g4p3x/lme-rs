//! Group-structure-preserving cross-validation for linear mixed models.
//!
//! Splits by grouping units (e.g. subjects) so all observations from one unit stay
//! in train or test together. Fits on each train fold and evaluates population-level
//! predictions on held-out groups.

use std::collections::HashSet;
use std::sync::Arc;

use ndarray::Array1;
use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::formula::parse;
use crate::{fit_prepared, prepare_lmer, LmeError, Result};

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
    /// Per-fold breakdown (sorted by fold index).
    pub folds: Vec<CvFoldMetric>,
    /// True when every fold fit converged.
    pub all_converged: bool,
    /// Number of CV folds used.
    pub n_splits: usize,
    /// Grouping column used for splitting.
    pub group_col: String,
}

struct FoldSpec {
    fold: usize,
    test_groups: HashSet<String>,
}

struct FoldWorkResult {
    metric: CvFoldMetric,
    oof_updates: Vec<(usize, f64)>,
    test_fold_value: i32,
    converged: bool,
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
/// * `n_jobs` - Parallel fold workers. `None` uses all logical CPUs (capped at fold
///   count); `Some(1)` runs folds sequentially.
pub fn cv_grouped(
    formula_str: &str,
    data: &DataFrame,
    group_col: &str,
    n_splits: usize,
    reml: bool,
    seed: Option<u64>,
    n_jobs: Option<usize>,
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
    if matches!(n_jobs, Some(0)) {
        return Err(LmeError::NotImplemented {
            feature: "cv_grouped requires n_jobs >= 1".to_string(),
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
    let y_all = column_to_f64_vec(data.column(&response_col).map_err(|e| {
        LmeError::NotImplemented {
            feature: format!("Response column '{response_col}' not found: {e}"),
        }
    })?)?;

    let chunk = groups.len().div_ceil(n_splits);
    let mut fold_specs = Vec::with_capacity(n_splits);
    for fold in 0..n_splits {
        let start = fold * chunk;
        if start >= groups.len() {
            break;
        }
        let end = ((fold + 1) * chunk).min(groups.len());
        let test_groups: HashSet<String> = groups[start..end].iter().cloned().collect();
        fold_specs.push(FoldSpec { fold, test_groups });
    }

    let workers = resolve_n_jobs(n_jobs, fold_specs.len());
    let data = Arc::new(data.clone());
    let formula = Arc::new(formula_str.to_string());
    let group_col = Arc::new(group_col.to_string());

    let fold_results = if workers == 1 {
        fold_specs
            .iter()
            .map(|spec| {
                run_fold(
                    spec,
                    &data,
                    &formula,
                    &group_col,
                    &groups,
                    reml,
                    &y_all,
                )
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        pin_competing_threadpools_single_thread();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("cv_grouped failed to build thread pool: {e}"),
            })?;
        pool.install(|| {
            fold_specs
                .par_iter()
                .map(|spec| {
                    run_fold(
                        spec,
                        &data,
                        &formula,
                        &group_col,
                        &groups,
                        reml,
                        &y_all,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?
    };

    let mut oof = Array1::<f64>::from_elem(n_obs, f64::NAN);
    let mut test_fold = Array1::<i32>::from_elem(n_obs, -1);
    let mut folds = Vec::with_capacity(fold_results.len());
    let mut all_converged = true;

    for result in fold_results {
        all_converged &= result.converged;
        for (row, pred) in result.oof_updates {
            oof[row] = pred;
            test_fold[row] = result.test_fold_value;
        }
        folds.push(result.metric);
    }
    folds.sort_by_key(|m| m.fold);

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
pub fn refit_lmer(formula_str: &str, data: &DataFrame, reml: bool) -> Result<crate::LmeFit> {
    let prepared = prepare_lmer(formula_str, data)?;
    fit_prepared(&prepared, reml)
}

fn resolve_n_jobs(n_jobs: Option<usize>, n_folds: usize) -> usize {
    let requested = n_jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    requested.max(1).min(n_folds.max(1))
}

/// Avoid BLAS/OpenMP oversubscription when each rayon worker also spawns threads.
fn pin_competing_threadpools_single_thread() {
    // BLAS backends read these when entering threaded regions; set before parallel folds.
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
}

fn run_fold(
    spec: &FoldSpec,
    data: &DataFrame,
    formula_str: &str,
    group_col: &str,
    all_groups: &[String],
    reml: bool,
    y_all: &[f64],
) -> Result<FoldWorkResult> {
    let train_set: HashSet<String> = all_groups
        .iter()
        .filter(|g| !spec.test_groups.contains(*g))
        .cloned()
        .collect();

    let train_df = filter_by_groups(data, group_col, &train_set)?;
    let test_df = filter_by_groups(data, group_col, &spec.test_groups)?;
    if train_df.height() == 0 || test_df.height() == 0 {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "cv_grouped fold {} produced empty train or test split",
                spec.fold
            ),
        });
    }

    let prepared = prepare_lmer(formula_str, &train_df)?;
    let fit = fit_prepared(&prepared, reml)?;
    let preds = fit.predict(&test_df).map_err(|e| LmeError::NotImplemented {
        feature: format!("cv_grouped prediction failed on fold {}: {e}", spec.fold),
    })?;

    let test_indices = row_indices_for_groups(data, group_col, &spec.test_groups)?;
    let mut fold_sq = 0.0;
    let mut fold_abs = 0.0;
    let mut oof_updates = Vec::with_capacity(test_indices.len());
    for (j, &row) in test_indices.iter().enumerate() {
        oof_updates.push((row, preds[j]));
        let err = y_all[row] - preds[j];
        fold_sq += err * err;
        fold_abs += err.abs();
    }
    let n_test = test_indices.len() as f64;
    let converged = fit.converged.unwrap_or(false);

    Ok(FoldWorkResult {
        metric: CvFoldMetric {
            fold: spec.fold,
            n_train_groups: train_set.len(),
            n_test_groups: spec.test_groups.len(),
            n_train_obs: train_df.height(),
            n_test_obs: test_df.height(),
            rmse: (fold_sq / n_test).sqrt(),
            mae: fold_abs / n_test,
            converged,
        },
        oof_updates,
        test_fold_value: spec.fold as i32,
        converged,
    })
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
    allowed: &HashSet<String>,
) -> Result<DataFrame> {
    let mask = group_mask(df, group_col, allowed)?;
    df.filter(&mask).map_err(|e| LmeError::NotImplemented {
        feature: format!("Failed to filter data by groups: {e}"),
    })
}

fn row_indices_for_groups(
    df: &DataFrame,
    group_col: &str,
    allowed: &HashSet<String>,
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
    allowed: &HashSet<String>,
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
