//! Estimated marginal means for categorical fixed effects.
//!
//! The reference grid follows the core `emmeans` convention: numeric columns are
//! held at their arithmetic means and nuisance categorical fixed effects are
//! averaged with equal weight over their stored levels. The resulting design rows
//! remain linear functions of the fitted fixed effects, so estimates, covariance,
//! confidence intervals, and pairwise comparisons all use the fitted `beta`/`vcov`.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::collections::HashMap;
use std::fmt;

use crate::anova::DdfMethod;
use crate::contrast::{fixed_effect_contrast_test, fixed_effect_vcov};
use crate::mcp::{adjust_p_values, McpAdjust};
use crate::{LmeError, LmeFit};

const MAX_REFERENCE_GRID_ROWS: usize = 4096;

/// Estimated marginal means for the levels of one categorical fixed effect.
#[derive(Debug, Clone)]
pub struct EmmeansResult {
    /// Categorical fixed-effect term.
    pub term: String,
    /// Stored factor levels in model-matrix order.
    pub levels: Vec<String>,
    /// Confidence level used for `lower` and `upper`.
    pub confidence_level: f64,
    /// `"z"` for asymptotic inference or `"t"` for a denominator-df method.
    pub statistic: String,
    /// Estimated marginal mean for each level.
    pub estimate: Array1<f64>,
    /// Standard error for each marginal mean.
    pub std_error: Array1<f64>,
    /// Denominator degrees of freedom (`infinity` for asymptotic inference).
    pub den_df: Array1<f64>,
    /// Lower confidence limit.
    pub lower: Array1<f64>,
    /// Upper confidence limit.
    pub upper: Array1<f64>,
    /// Reference-grid linear functions, one row per level (`L beta`).
    pub linfct: Array2<f64>,
}

/// Pairwise comparisons among estimated marginal means.
#[derive(Debug, Clone)]
pub struct EmmeansPairsResult {
    /// Categorical fixed-effect term.
    pub term: String,
    /// Multiplicity adjustment applied to `p_adjust`.
    pub adjust: McpAdjust,
    /// `"z"` for asymptotic inference or `"t"` for a denominator-df method.
    pub statistic: String,
    /// Labels such as `"b - a"`.
    pub comparisons: Vec<String>,
    /// Difference between marginal means.
    pub estimate: Array1<f64>,
    /// Standard error of each difference.
    pub std_error: Array1<f64>,
    /// `estimate / std_error`.
    pub statistic_values: Array1<f64>,
    /// Denominator degrees of freedom (`infinity` for asymptotic inference).
    pub den_df: Array1<f64>,
    /// Unadjusted two-sided p-values.
    pub p_value: Array1<f64>,
    /// Multiplicity-adjusted p-values.
    pub p_adjust: Array1<f64>,
}

impl LmeFit {
    /// Estimate marginal means for a categorical fixed-effect term.
    ///
    /// `data` supplies the observed column types and numeric means used to build the
    /// reference grid. Other categorical fixed effects are averaged equally over
    /// their stored levels. `ddf = None` uses asymptotic Wald z inference; a
    /// Satterthwaite or Kenward-Roger method requires the corresponding
    /// `with_satterthwaite()` or `with_kenward_roger()` call first.
    pub fn emmeans(
        &self,
        term: &str,
        data: &DataFrame,
        confidence_level: f64,
        ddf: Option<DdfMethod>,
    ) -> crate::Result<EmmeansResult> {
        validate_confidence_level(confidence_level)?;
        let (linfct, levels) = reference_grid_linfct(self, term, data)?;
        let v_beta = fixed_effect_vcov(self)?;
        let mut estimate = Array1::<f64>::zeros(levels.len());
        let mut std_error = Array1::<f64>::zeros(levels.len());
        let mut den_df = Array1::<f64>::zeros(levels.len());
        let mut lower = Array1::<f64>::zeros(levels.len());
        let mut upper = Array1::<f64>::zeros(levels.len());

        for i in 0..levels.len() {
            let row = linfct.row(i).to_owned();
            let est = row.dot(&self.coefficients);
            let var = row.dot(&v_beta.dot(&row));
            if !var.is_finite() || var <= 0.0 {
                return Err(LmeError::NotImplemented {
                    feature: format!(
                        "Non-positive marginal-mean variance for {term}={}",
                        levels[i]
                    ),
                });
            }
            let se = var.sqrt();
            let df = denominator_df(self, &row, ddf)?;
            let critical = critical_value(confidence_level, df)?;
            estimate[i] = est;
            std_error[i] = se;
            den_df[i] = df;
            lower[i] = est - critical * se;
            upper[i] = est + critical * se;
        }

        Ok(EmmeansResult {
            term: term.to_string(),
            levels,
            confidence_level,
            statistic: if ddf.is_some() { "t" } else { "z" }.to_string(),
            estimate,
            std_error,
            den_df,
            lower,
            upper,
            linfct,
        })
    }

    /// Compute all pairwise comparisons among estimated marginal means.
    ///
    /// This uses the same reference grid as [`Self::emmeans`]. Tukey-Kramer,
    /// Holm, Bonferroni, and unadjusted p-values are available through `adjust`.
    pub fn emmeans_pairs(
        &self,
        term: &str,
        data: &DataFrame,
        adjust: McpAdjust,
        ddf: Option<DdfMethod>,
    ) -> crate::Result<EmmeansPairsResult> {
        let (means_l, levels) = reference_grid_linfct(self, term, data)?;
        let p = self.coefficients.len();
        let pairs: Vec<(usize, usize)> = (0..levels.len())
            .flat_map(|i| ((i + 1)..levels.len()).map(move |j| (i, j)))
            .collect();
        let mut l_mat = Array2::<f64>::zeros((pairs.len(), p));
        let mut comparisons = Vec::with_capacity(pairs.len());
        for (r, &(i, j)) in pairs.iter().enumerate() {
            l_mat
                .row_mut(r)
                .assign(&(&means_l.row(j) - &means_l.row(i)));
            comparisons.push(format!("{} - {}", levels[j], levels[i]));
        }

        let v_beta = fixed_effect_vcov(self)?;
        let mut estimate = Array1::<f64>::zeros(pairs.len());
        let mut std_error = Array1::<f64>::zeros(pairs.len());
        let mut statistic_values = Array1::<f64>::zeros(pairs.len());
        let mut den_df = Array1::<f64>::zeros(pairs.len());
        let mut p_value = Array1::<f64>::zeros(pairs.len());
        let normal = Normal::new(0.0, 1.0).expect("standard normal");

        for i in 0..pairs.len() {
            let row = l_mat.row(i).to_owned();
            let est = row.dot(&self.coefficients);
            let var = row.dot(&v_beta.dot(&row));
            if !var.is_finite() || var <= 0.0 {
                return Err(LmeError::NotImplemented {
                    feature: format!(
                        "Non-positive EMM contrast variance for '{}'",
                        comparisons[i]
                    ),
                });
            }
            let se = var.sqrt();
            let stat = est / se;
            let df = denominator_df(self, &row, ddf)?;
            let raw_p = match ddf {
                None => (2.0 * (1.0 - normal.cdf(stat.abs()))).clamp(0.0, 1.0),
                Some(method) => {
                    let one = l_mat.slice(ndarray::s![i..i + 1, ..]).to_owned();
                    fixed_effect_contrast_test(self, &one, method, None)?.p_value
                }
            };
            estimate[i] = est;
            std_error[i] = se;
            statistic_values[i] = stat;
            den_df[i] = df;
            p_value[i] = raw_p;
        }

        let p_adjust = adjust_p_values(
            adjust,
            p_value.as_slice().expect("contiguous p-values"),
            levels.len(),
            &statistic_values,
            &den_df,
        )?;

        Ok(EmmeansPairsResult {
            term: term.to_string(),
            adjust,
            statistic: if ddf.is_some() { "t" } else { "z" }.to_string(),
            comparisons,
            estimate,
            std_error,
            statistic_values,
            den_df,
            p_value,
            p_adjust: Array1::from_vec(p_adjust),
        })
    }
}

fn validate_confidence_level(level: f64) -> crate::Result<()> {
    if level.is_finite() && level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(LmeError::NotImplemented {
            feature: format!("confidence level must be between 0 and 1, got {level}"),
        })
    }
}

fn denominator_df(fit: &LmeFit, row: &Array1<f64>, ddf: Option<DdfMethod>) -> crate::Result<f64> {
    match ddf {
        None => Ok(f64::INFINITY),
        Some(method) => {
            let l_mat = Array2::from_shape_vec((1, row.len()), row.to_vec()).map_err(|e| {
                LmeError::NotImplemented {
                    feature: format!("Could not form marginal-mean contrast: {e}"),
                }
            })?;
            Ok(fixed_effect_contrast_test(fit, &l_mat, method, None)?.den_df)
        }
    }
}

fn critical_value(level: f64, df: f64) -> crate::Result<f64> {
    let probability = 0.5 + level / 2.0;
    if df.is_infinite() {
        return Ok(Normal::new(0.0, 1.0)
            .expect("standard normal")
            .inverse_cdf(probability));
    }
    StudentsT::new(0.0, 1.0, df)
        .map(|dist| dist.inverse_cdf(probability))
        .map_err(|e| LmeError::NotImplemented {
            feature: format!("Invalid denominator degrees of freedom {df}: {e}"),
        })
}

fn reference_grid_linfct(
    fit: &LmeFit,
    term: &str,
    data: &DataFrame,
) -> crate::Result<(Array2<f64>, Vec<String>)> {
    if fit.family_name.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "Estimated marginal means currently support linear models and LMMs only"
                .to_string(),
        });
    }
    if data.height() == 0 {
        return Err(LmeError::NotImplemented {
            feature: "Estimated marginal means require at least one data row".to_string(),
        });
    }

    let formula = fit
        .formula
        .as_deref()
        .ok_or_else(|| LmeError::NotImplemented {
            feature: "Fitted formula metadata is missing".to_string(),
        })?;
    let ast = crate::formula::parse(formula)?;
    if ast.offset.is_some() {
        return Err(LmeError::NotImplemented {
            feature: "Estimated marginal means with formula offsets are not implemented"
                .to_string(),
        });
    }

    let categorical = fit
        .categorical_levels
        .as_ref()
        .ok_or_else(|| LmeError::NotImplemented {
            feature: "Fitted categorical-level metadata is missing".to_string(),
        })?;
    let target_levels = categorical
        .get(term)
        .filter(|levels| levels.len() >= 2)
        .cloned()
        .ok_or_else(|| LmeError::NotImplemented {
            feature: format!(
                "Estimated marginal means require a categorical fixed term; '{term}' was not found"
            ),
        })?;

    let mut factors: Vec<(String, Vec<String>)> = categorical
        .iter()
        .filter(|(name, levels)| name.as_str() != term && levels.len() >= 2)
        .map(|(name, levels)| (name.clone(), levels.clone()))
        .collect();
    factors.sort_by(|a, b| a.0.cmp(&b.0));
    factors.insert(0, (term.to_string(), target_levels.clone()));

    let n_grid = factors
        .iter()
        .try_fold(1usize, |acc, (_, levels)| acc.checked_mul(levels.len()))
        .ok_or_else(|| LmeError::NotImplemented {
            feature: "Reference-grid row count overflowed".to_string(),
        })?;
    if n_grid > MAX_REFERENCE_GRID_ROWS {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Reference grid has {n_grid} rows; maximum is {MAX_REFERENCE_GRID_ROWS}"
            ),
        });
    }

    let indices = IdxCa::from_vec("reference_row".into(), vec![0 as IdxSize; n_grid]);
    let mut grid = data.take(&indices).map_err(|e| LmeError::NotImplemented {
        feature: format!("Could not initialize reference grid: {e}"),
    })?;

    let factor_map: HashMap<&str, &Vec<String>> = factors
        .iter()
        .map(|(name, levels)| (name.as_str(), levels))
        .collect();
    for column in data.get_columns() {
        let name = column.name();
        if factor_map.contains_key(name.as_str()) {
            continue;
        }
        if is_native_numeric_dtype(column.dtype()) {
            let cast = column
                .cast(&DataType::Float64)
                .map_err(|e| LmeError::NotImplemented {
                    feature: format!("Could not cast '{name}' while building reference grid: {e}"),
                })?;
            if let Some(mean) = cast.f64().ok().and_then(|values| values.mean()) {
                grid.with_column(Column::new(name.clone(), vec![mean; n_grid]))
                    .map_err(|e| LmeError::NotImplemented {
                        feature: format!("Could not set mean for '{name}' in reference grid: {e}"),
                    })?;
            }
        }
    }

    for (factor_idx, (name, levels)) in factors.iter().enumerate() {
        let stride: usize = factors[(factor_idx + 1)..]
            .iter()
            .map(|(_, later_levels)| later_levels.len())
            .product();
        let values: Vec<String> = (0..n_grid)
            .map(|row| levels[(row / stride) % levels.len()].clone())
            .collect();
        grid.with_column(Column::new(name.clone().into(), values))
            .map_err(|e| LmeError::NotImplemented {
                feature: format!("Could not set factor '{name}' in reference grid: {e}"),
            })?;
    }

    let response_name = ast
        .columns
        .iter()
        .find(|(_, info)| info.has_role(crate::formula::ColumnRole::Response))
        .map(|(name, _)| name.as_str())
        .unwrap_or("");
    let (x_grid, names, _, _, _) = crate::model_matrix::build_x_matrix(
        &ast,
        &grid,
        response_name,
        n_grid,
        fit.categorical_levels.as_ref(),
        fit.basis_encodings.as_ref(),
    )?;
    let fitted_names = fit.fixed_names.clone().unwrap_or_default();
    if names != fitted_names {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Reference-grid columns {names:?} do not match fitted columns {fitted_names:?}"
            ),
        });
    }

    let per_target = n_grid / target_levels.len();
    let mut linfct = Array2::<f64>::zeros((target_levels.len(), x_grid.ncols()));
    for level_idx in 0..target_levels.len() {
        let start = level_idx * per_target;
        for row in start..(start + per_target) {
            linfct
                .row_mut(level_idx)
                .scaled_add(1.0 / per_target as f64, &x_grid.row(row));
        }
    }
    Ok((linfct, target_levels))
}

fn is_native_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Float64
            | DataType::Float32
            | DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8
    )
}

impl fmt::Display for EmmeansResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Estimated marginal means for {}", self.term)?;
        writeln!(
            f,
            "{:<16} {:>12} {:>12} {:>12} {:>12}",
            "Level", "Estimate", "SE", "Lower", "Upper"
        )?;
        for i in 0..self.levels.len() {
            writeln!(
                f,
                "{:<16} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                self.levels[i], self.estimate[i], self.std_error[i], self.lower[i], self.upper[i]
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for EmmeansPairsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pairwise estimated marginal means for {}", self.term)?;
        writeln!(
            f,
            "{:<16} {:>12} {:>12} {:>12} {:>12}",
            "Contrast", "Estimate", "SE", self.statistic, "p adj"
        )?;
        for i in 0..self.comparisons.len() {
            writeln!(
                f,
                "{:<16} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                self.comparisons[i],
                self.estimate[i],
                self.std_error[i],
                self.statistic_values[i],
                self.p_adjust[i]
            )?;
        }
        Ok(())
    }
}
