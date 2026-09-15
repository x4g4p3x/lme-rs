//! Estimated marginal means for categorical fixed effects.
//!
//! The reference grid follows the core `emmeans` convention: numeric columns are
//! held at their arithmetic means and nuisance categorical fixed effects are
//! averaged with equal weight over their stored levels. The resulting design rows
//! remain linear functions of the fitted fixed effects, so estimates, covariance,
//! confidence intervals, and pairwise comparisons all use the fitted `beta`/`vcov`.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::BTreeMap;
use std::fmt;

use crate::anova::DdfMethod;
use crate::contrast::{
    fixed_effect_contrast_test, fixed_effect_vcov_for_method, wald_critical_value,
};
use crate::mcp::{adjust_p_values, McpAdjust};
use crate::{LmeError, LmeFit};

const MAX_REFERENCE_GRID_ROWS: usize = 4096;

/// Weights used to average over nuisance-factor combinations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GridWeights {
    /// Equal weight for every nuisance-factor combination.
    #[default]
    Equal,
    /// Pooled observed joint frequencies of nuisance factors, shared across cells.
    Proportional,
}

/// Reference-grid contract for marginal means and simple comparisons.
#[derive(Debug, Clone, Default)]
pub struct ReferenceGrid {
    /// Target factors; their Cartesian product defines the compared cells.
    pub terms: Vec<String>,
    /// Conditioning factors; comparisons and multiplicity families stay within each cell.
    pub by: Vec<String>,
    /// Explicit numeric covariate reference values; unspecified covariates use their means.
    pub at: BTreeMap<String, f64>,
    /// Nuisance-factor averaging policy.
    pub weights: GridWeights,
}

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
    /// Named target and conditioning factor levels for each row.
    pub cells: Vec<BTreeMap<String, String>>,
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
    /// Conditioning levels identifying each independently adjusted comparison family.
    pub groups: Vec<BTreeMap<String, String>>,
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
        self.emmeans_with_grid(
            data,
            &ReferenceGrid {
                terms: vec![term.into()],
                ..Default::default()
            },
            confidence_level,
            ddf,
        )
    }

    /// Marginal means for target-factor combinations at an explicit reference grid.
    pub fn emmeans_with_grid(
        &self,
        data: &DataFrame,
        options: &ReferenceGrid,
        confidence_level: f64,
        ddf: Option<DdfMethod>,
    ) -> crate::Result<EmmeansResult> {
        let term = options.terms.join(":");
        validate_confidence_level(confidence_level)?;
        let (linfct, levels, cells) = reference_grid_linfct(self, options, data)?;
        let v_beta = fixed_effect_vcov_for_method(self, ddf)?;
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
            let critical = wald_critical_value(confidence_level, df)?;
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
            cells,
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
        self.emmeans_pairs_with_grid(
            data,
            &ReferenceGrid {
                terms: vec![term.into()],
                ..Default::default()
            },
            adjust,
            ddf,
        )
    }

    /// Compare target cells within each `by` stratum, adjusting each family separately.
    pub fn emmeans_pairs_with_grid(
        &self,
        data: &DataFrame,
        options: &ReferenceGrid,
        adjust: McpAdjust,
        ddf: Option<DdfMethod>,
    ) -> crate::Result<EmmeansPairsResult> {
        let term = options.terms.join(":");
        let (means_l, levels, cells) = reference_grid_linfct(self, options, data)?;
        let cell_groups: Vec<BTreeMap<String, String>> = cells
            .iter()
            .map(|cell| {
                options
                    .by
                    .iter()
                    .map(|name| (name.clone(), cell[name].clone()))
                    .collect()
            })
            .collect();
        let p = self.coefficients.len();
        let pairs: Vec<(usize, usize)> = (0..levels.len())
            .flat_map(|i| ((i + 1)..levels.len()).map(move |j| (i, j)))
            .filter(|&(i, j)| cell_groups[i] == cell_groups[j])
            .collect();
        let groups: Vec<_> = pairs.iter().map(|&(i, _)| cell_groups[i].clone()).collect();
        let mut l_mat = Array2::<f64>::zeros((pairs.len(), p));
        let mut comparisons = Vec::with_capacity(pairs.len());
        for (r, &(i, j)) in pairs.iter().enumerate() {
            l_mat
                .row_mut(r)
                .assign(&(&means_l.row(j) - &means_l.row(i)));
            comparisons.push(format!("{} - {}", levels[j], levels[i]));
        }

        let v_beta = fixed_effect_vcov_for_method(self, ddf)?;
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
                None => (2.0 * normal.sf(stat.abs())).clamp(0.0, 1.0),
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

        let mut p_adjust = vec![f64::NAN; pairs.len()];
        let mut families: BTreeMap<BTreeMap<String, String>, Vec<usize>> = BTreeMap::new();
        for (i, group) in groups.iter().enumerate() {
            families.entry(group.clone()).or_default().push(i);
        }
        for (group, indices) in families {
            let n_groups = cell_groups.iter().filter(|g| **g == group).count();
            let raw: Vec<_> = indices.iter().map(|&i| p_value[i]).collect();
            let stats = Array1::from_iter(indices.iter().map(|&i| statistic_values[i]));
            let dfs = Array1::from_iter(indices.iter().map(|&i| den_df[i]));
            let adjusted = adjust_p_values(adjust, &raw, n_groups, &stats, &dfs)?;
            for (&i, value) in indices.iter().zip(adjusted) {
                p_adjust[i] = value;
            }
        }

        Ok(EmmeansPairsResult {
            groups,
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

#[allow(clippy::type_complexity)]
fn reference_grid_linfct(
    fit: &LmeFit,
    options: &ReferenceGrid,
    data: &DataFrame,
) -> crate::Result<(Array2<f64>, Vec<String>, Vec<BTreeMap<String, String>>)> {
    let invalid = |message: String| LmeError::InvalidInput { message };
    if fit.family_name.is_some() || data.height() == 0 {
        return Err(invalid(
            "Reference grids require a linear model and nonempty data".into(),
        ));
    }
    let ast = fit.model_spec().formula_model()?;
    if ast.offset.is_some() {
        return Err(invalid("Reference grids with offsets require an explicit prediction design; automatic averaging is unsupported".into()));
    }
    let categorical = fit
        .categorical_levels
        .as_ref()
        .ok_or_else(|| invalid("Fitted factor metadata is missing".into()))?;
    if options.terms.is_empty() {
        return Err(invalid("At least one target factor is required".into()));
    }
    let mut retained = options.by.clone();
    retained.extend(options.terms.clone());
    if retained
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != retained.len()
    {
        return Err(invalid(
            "Target and conditioning factors must be distinct".into(),
        ));
    }
    for name in &retained {
        if !categorical.contains_key(name) {
            return Err(invalid(format!(
                "Unknown categorical fixed factor '{name}'"
            )));
        }
    }
    let mut nuisance: Vec<_> = categorical
        .keys()
        .filter(|name| !retained.contains(name))
        .cloned()
        .collect();
    nuisance.sort();
    let names: Vec<_> = retained.iter().chain(nuisance.iter()).cloned().collect();
    let mut n_grid = 1usize;
    for name in &names {
        n_grid = n_grid
            .checked_mul(categorical[name].len())
            .filter(|&n| n <= MAX_REFERENCE_GRID_ROWS)
            .ok_or_else(|| {
                invalid(format!(
                    "Reference grid exceeds {MAX_REFERENCE_GRID_ROWS} rows"
                ))
            })?;
    }
    let per_cell: usize = nuisance.iter().map(|n| categorical[n].len()).product();
    let n_cells = n_grid / per_cell;
    let indices = IdxCa::from_vec("reference_row".into(), vec![0 as IdxSize; n_grid]);
    let mut grid = data.take(&indices).map_err(|e| invalid(e.to_string()))?;
    for (name, value) in &options.at {
        if !value.is_finite()
            || categorical.contains_key(name)
            || !ast
                .columns
                .get(name)
                .is_some_and(|info| !info.has_role(crate::formula::ColumnRole::Response))
            || !data
                .column(name)
                .is_ok_and(|c| is_native_numeric_dtype(c.dtype()))
        {
            return Err(invalid(format!(
                "Reference value '{name}' must name a numeric model covariate and be finite"
            )));
        }
    }
    for column in data.get_columns() {
        let name = column.name();
        if categorical.contains_key(name.as_str()) || !is_native_numeric_dtype(column.dtype()) {
            continue;
        }
        let mean = match options.at.get(name.as_str()) {
            Some(&value) => value,
            None => column
                .cast(&DataType::Float64)
                .map_err(|e| invalid(e.to_string()))?
                .f64()
                .map_err(|e| invalid(e.to_string()))?
                .mean()
                .ok_or_else(|| invalid(format!("No numeric reference value for '{name}'")))?,
        };
        grid.with_column(Column::new(name.clone(), vec![mean; n_grid]))
            .map_err(|e| invalid(e.to_string()))?;
    }
    let mut cells = vec![BTreeMap::new(); n_cells];
    for (factor_idx, name) in names.iter().enumerate() {
        let levels = &categorical[name];
        let stride: usize = names[factor_idx + 1..]
            .iter()
            .map(|n| categorical[n].len())
            .product();
        let values: Vec<_> = (0..n_grid)
            .map(|row| levels[(row / stride) % levels.len()].clone())
            .collect();
        if factor_idx < retained.len() {
            for (i, cell) in cells.iter_mut().enumerate() {
                cell.insert(name.clone(), values[i * per_cell].clone());
            }
        }
        grid.with_column(Column::new(name.clone().into(), values))
            .map_err(|e| invalid(e.to_string()))?;
    }
    let mut weights = vec![1.0 / per_cell as f64; per_cell];
    if options.weights == GridWeights::Proportional && !nuisance.is_empty() {
        weights.fill(0.0);
        let columns = nuisance
            .iter()
            .map(|name| {
                data.column(name)
                    .map_err(|e| invalid(e.to_string()))?
                    .cast(&DataType::String)
                    .map_err(|e| invalid(e.to_string()))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        for row in 0..data.height() {
            let mut index = 0;
            for (j, name) in nuisance.iter().enumerate() {
                let value = columns[j]
                    .str()
                    .map_err(|e| invalid(e.to_string()))?
                    .get(row)
                    .ok_or_else(|| invalid(format!("Missing nuisance factor '{name}'")))?;
                let level = categorical[name]
                    .iter()
                    .position(|v| v == value)
                    .ok_or_else(|| invalid(format!("Unknown nuisance level '{value}'")))?;
                index = index * categorical[name].len() + level;
            }
            weights[index] += 1.0 / data.height() as f64;
        }
    }
    let (x, _) = fit.model_spec().prediction_matrix(&grid)?;
    let mut linfct = Array2::zeros((n_cells, x.ncols()));
    for cell in 0..n_cells {
        for (j, &weight) in weights.iter().enumerate() {
            linfct
                .row_mut(cell)
                .scaled_add(weight, &x.row(cell * per_cell + j));
        }
    }
    let labels = cells
        .iter()
        .map(|cell| {
            retained
                .iter()
                .map(|name| {
                    if retained.len() == 1 {
                        cell[name].clone()
                    } else {
                        format!("{name}={}", cell[name])
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect();
    Ok((linfct, labels, cells))
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
