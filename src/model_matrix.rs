use crate::formula::FiastoModel;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;

use sprs::TriMat;

/// Captures structural layout and variable indices for a multi-dimensional Random Effect correlation block.
#[derive(Clone, Debug)]
pub struct ReBlock {
    /// The number of unique grouping clusters / levels.
    pub m: usize,
    /// The number of random effect variables per group (e.g., 1 for intercept-only, 2 for intercept+slope).
    pub k: usize,
    /// The number of variance-covariance parameters associated with this block (e.g. k(k+1)/2 ).
    pub theta_len: usize,
    /// The name of the grouping factor variable (e.g. "Subject").
    pub group_name: String,
    /// The canonical names of the effects in the block (e.g., `["(Intercept)", "Days"]`).
    pub effect_names: Vec<String>,
    /// Maps group labels (e.g. subject IDs) to their positional index in the Z/b vectors.
    pub group_map: HashMap<String, usize>,
}

/// Container encompassing the finalized design matrices and mapping arrays extracted from input DataFrames.
pub struct DesignMatrices {
    /// The Wilkinson formula string that generated the matrices.
    pub formula: String,
    /// Dense fixed-effects design matrix ($X$).
    pub x: Array2<f64>,
    /// Sparse transposed random-effects design matrix ($Z^T$).
    pub zt: sprs::CsMat<f64>,
    /// Dependent variable vector ($y$).
    pub y: Array1<f64>,
    /// Collection of random effect dimensional tracking blocks.
    pub re_blocks: Vec<ReBlock>,
    /// Extracted names of fixed-effect features from the dataframe.
    pub fixed_names: Vec<String>,
    /// Model-term label per fixed-effects column (parallel to `fixed_names`).
    pub fixed_term_assign: Vec<String>,
    /// Optional vector of offset terms to shift predictions by.
    pub offset: Option<Array1<f64>>,
    /// Saved categorical dummy variable levels during training.
    pub categorical_levels: HashMap<String, Vec<String>>,
}

/// Constructs structural matrices formatting Fixed Effects ($X$) and Random Effects ($Z$) bounds out of `fiasto` inputs.
pub fn build_design_matrices(ast: &FiastoModel, data: &DataFrame) -> crate::Result<DesignMatrices> {
    let n_obs = data.height();

    // 1. Determine Response (y)
    let mut response_col_name = None;
    for (name, info) in &ast.columns {
        if info.roles.contains(&"Response".to_string()) {
            response_col_name = Some(name.clone());
            break;
        }
    }
    let response_name = response_col_name.ok_or_else(|| crate::LmeError::NotImplemented {
        feature: "Missing response variable".to_string(),
    })?;

    let y_series_cast = data
        .column(&response_name)
        .map_err(|e| crate::LmeError::NotImplemented {
            feature: format!("Data missing response column: {}", e),
        })?
        .cast(&DataType::Float64)
        .map_err(|e| crate::LmeError::NotImplemented {
            feature: format!("Response must be castable to float: {}", e),
        })?;
    if y_series_cast.null_count() > 0 {
        return Err(crate::LmeError::NotImplemented {
            feature: "Response column contains nulls or invalid floats".to_string(),
        });
    }
    let y_vec: Vec<f64> = y_series_cast
        .f64()
        .map_err(|_| crate::LmeError::NotImplemented {
            feature: "Response must be float".to_string(),
        })?
        .into_no_null_iter()
        .collect();
    let y = Array1::from_vec(y_vec);

    let (x, fixed_names, fixed_term_assign, categorical_levels) =
        build_x_matrix(ast, data, &response_name, n_obs, None)?;

    // 2. Extract Offset (if any)
    let offset = if let Some(off_name) = &ast.offset {
        let off_series = data
            .column(off_name)
            .map_err(|e| crate::LmeError::NotImplemented {
                feature: format!("Data missing offset column '{}': {}", off_name, e),
            })?
            .cast(&DataType::Float64)
            .map_err(|e| crate::LmeError::NotImplemented {
                feature: format!("Offset must be castable to float: {}", e),
            })?;
        if off_series.null_count() > 0 {
            return Err(crate::LmeError::NotImplemented {
                feature: format!(
                    "Offset column '{}' contains nulls or invalid floats",
                    off_name
                ),
            });
        }
        let off_vec: Vec<f64> = off_series
            .f64()
            .map_err(|_| crate::LmeError::NotImplemented {
                feature: "Offset must be float".to_string(),
            })?
            .into_no_null_iter()
            .collect();
        Some(Array1::from_vec(off_vec))
    } else {
        None
    };

    // 3. Build Random Effects Matrix (Z) -- now supporting Crossed and Multiple Slopes
    let mut triplet_rows = Vec::new();
    let mut triplet_cols = Vec::new();
    let mut triplet_vals = Vec::new();

    let mut re_blocks = Vec::new();
    let mut current_q_offset = 0;

    for (name, info) in &ast.columns {
        if info.roles.contains(&"GroupingVariable".to_string()) {
            let g_var = name;
            let mut slope_vars = Vec::new();

            for re in &info.random_effects {
                if let Some(vars) = &re.variables {
                    slope_vars.extend(vars.clone());
                }
            }

            // Deduplicate slope vars
            slope_vars.sort();
            slope_vars.dedup();

            // Determine has_intercept for this RE group.
            // fiasto's `has_intercept` on `RandomEffect` means "intercept-only RE" (true only
            // for `(1|g)`), NOT "RE group includes an intercept alongside slopes".
            // R convention: intercept is always included unless explicitly suppressed with `0 +`.
            // We detect suppression by scanning the raw formula for patterns like `(0 + ... | g)`
            // or `(0+ ... | g)`.
            let suppress_pattern = format!("| {}", g_var);
            let has_zero_suppression = ast
                .formula
                .split(&suppress_pattern)
                .next() // text before `| group`
                .and_then(|before| before.rfind('(')) // find the opening `(`
                .map(|paren_pos| {
                    let inside = &ast.formula[paren_pos + 1..];
                    let trimmed = inside.trim_start();
                    trimmed.starts_with("0 +") || trimmed.starts_with("0+")
                })
                .unwrap_or(false);
            let has_intercept = !has_zero_suppression;

            // Handle interaction grouping variables (e.g., "school:student")
            // created by nested RE expansion: paste column values to create interaction groups
            let g_series = if g_var.contains(':') {
                let parts: Vec<&str> = g_var.split(':').collect();
                let mut interaction_values: Vec<String> = Vec::with_capacity(n_obs);

                for i in 0..n_obs {
                    let mut parts_str = Vec::new();
                    for part in &parts {
                        let col =
                            data.column(part)
                                .map_err(|e| crate::LmeError::NotImplemented {
                                    feature: format!(
                                        "Nested RE column '{}' not found: {}",
                                        part, e
                                    ),
                                })?;
                        let val = col.get(i).unwrap();
                        parts_str.push(format!("{}", val));
                    }
                    interaction_values.push(parts_str.join("_"));
                }

                Column::new(g_var.as_str().into(), &interaction_values)
            } else {
                data
                    .column(g_var)
                    .map_err(|e| crate::LmeError::NotImplemented {
                        feature: format!("Grouping column '{}' not found: {}", g_var, e),
                    })?
                    .cast(&DataType::String)
                    .map_err(|e| crate::LmeError::NotImplemented {
                        feature: format!(
                            "Grouping column '{}' could not be cast to string: {}",
                            g_var, e
                        ),
                    })?
            };
            let g_str = g_series.str().unwrap();

            let mut unique_groups = Vec::new();
            let mut group_map = HashMap::new();

            for val_opt in g_str.into_iter() {
                let val = val_opt.unwrap().to_string();
                if !group_map.contains_key(&val) {
                    group_map.insert(val.clone(), unique_groups.len());
                    unique_groups.push(val);
                }
            }

            let m = unique_groups.len();
            let k = if has_intercept {
                1 + slope_vars.len()
            } else {
                slope_vars.len()
            };
            let q_block = m * k;

            let mut slope_data: Vec<Vec<f64>> = Vec::new();
            for s_var in &slope_vars {
                let s_series = data
                    .column(s_var)
                    .unwrap()
                    .cast(&DataType::Float64)
                    .unwrap();
                let s_f64_series = s_series.f64().unwrap();
                slope_data.push(s_f64_series.into_no_null_iter().collect());
            }

            for (i, val_opt) in g_str.into_iter().enumerate() {
                let val = val_opt.unwrap();
                let group_idx = group_map[val];
                let offset = current_q_offset + group_idx * k;

                let mut current_k = 0;
                if has_intercept {
                    triplet_rows.push(i);
                    triplet_cols.push(offset);
                    triplet_vals.push(1.0);
                    current_k += 1;
                }

                for (slope_idx, s_vec) in slope_data.iter().enumerate() {
                    triplet_rows.push(i);
                    triplet_cols.push(offset + current_k + slope_idx);
                    triplet_vals.push(s_vec[i]);
                }
            }
            let theta_len = k * (k + 1) / 2;

            let mut effect_names = Vec::new();
            if has_intercept {
                effect_names.push("(Intercept)".to_string());
            }
            effect_names.extend(slope_vars.iter().map(|s| s.to_string()));

            re_blocks.push(ReBlock {
                m,
                k,
                theta_len,
                group_name: g_var.to_string(),
                effect_names,
                group_map,
            });
            current_q_offset += q_block;
        }
    }

    let zt = if current_q_offset > 0 {
        // We dynamically update the ncols bound of Z to match the full horizontal width
        let mut final_z_tri = TriMat::new((n_obs, current_q_offset));
        for i in 0..triplet_rows.len() {
            final_z_tri.add_triplet(triplet_rows[i], triplet_cols[i], triplet_vals[i]);
        }
        let z_csc = final_z_tri.to_csc();
        z_csc.transpose_into()
    } else {
        sprs::CsMat::zero((0, n_obs))
    };

    Ok(DesignMatrices {
        formula: ast.formula.clone(),
        x,
        zt,
        y,
        re_blocks,
        fixed_names,
        fixed_term_assign,
        offset,
        categorical_levels,
    })
}

fn is_fixed_effect_column(
    info: &crate::formula::ColumnInfo,
    col_name: &str,
    response_name: &str,
) -> bool {
    if col_name == response_name || col_name == "intercept" {
        return false;
    }
    info.roles
        .iter()
        .any(|r| matches!(r.as_str(), "Identity" | "FixedEffect" | "InteractionTerm"))
}

/// Map fiasto interaction column name (`trt_blk`) to R-style term label (`trt:blk`).
fn interaction_term_label(col_name: &str, factor_names: &[String]) -> String {
    if col_name.contains(':') {
        return col_name.to_string();
    }
    for a in factor_names {
        if let Some(rest) = col_name.strip_prefix(a.as_str()) {
            if let Some(b) = rest.strip_prefix('_') {
                if factor_names.iter().any(|f| f == b) {
                    return format!("{}:{}", a, b);
                }
            }
        }
    }
    col_name.replace('_', ":")
}

fn interaction_factor_names(col_name: &str, factor_names: &[String]) -> Option<Vec<String>> {
    if col_name.contains(':') {
        return Some(col_name.split(':').map(|s| s.to_string()).collect());
    }
    for a in factor_names {
        if let Some(rest) = col_name.strip_prefix(a.as_str()) {
            if let Some(b) = rest.strip_prefix('_') {
                if factor_names.iter().any(|f| f == b) {
                    return Some(vec![a.clone(), b.to_string()]);
                }
            }
        }
    }
    None
}

/// Evaluates the `ast` structural formula to isolate and resolve pure population-level ($X$) design matrices.
#[allow(clippy::type_complexity)]
pub fn build_x_matrix(
    ast: &FiastoModel,
    data: &DataFrame,
    response_name: &str,
    n_obs: usize,
    training_levels: Option<&HashMap<String, Vec<String>>>,
) -> crate::Result<(
    Array2<f64>,
    Vec<String>,
    Vec<String>,
    HashMap<String, Vec<String>>,
)> {
    let mut fixed_cols = Vec::new();
    let mut fixed_names = Vec::new();
    let mut fixed_term_assign = Vec::new();
    let mut intercept_handled = ast.metadata.has_intercept;
    let mut extracted_levels: HashMap<String, Vec<String>> = HashMap::new();

    let factor_names: Vec<String> = ast
        .columns
        .iter()
        .filter(|(name, info)| {
            *name != response_name
                && !info.roles.contains(&"InteractionTerm".to_string())
                && is_fixed_effect_column(info, name, response_name)
        })
        .map(|(n, _)| n.clone())
        .collect();

    let mut term_col_arrays: HashMap<String, Vec<Array1<f64>>> = HashMap::new();

    if ast.metadata.has_intercept {
        fixed_cols.push(Array1::<f64>::ones(n_obs));
        fixed_names.push("(Intercept)".to_string());
        fixed_term_assign.push("(Intercept)".to_string());
    }

    // Ordered columns per the formula
    for col_name in ast.all_generated_columns.iter() {
        let Some(info) = ast.columns.get(col_name) else {
            continue;
        };
        if !is_fixed_effect_column(info, col_name, response_name) {
            continue;
        }

        if info.roles.contains(&"InteractionTerm".to_string()) {
            let factors = interaction_factor_names(col_name, &factor_names).ok_or_else(|| {
                crate::LmeError::NotImplemented {
                    feature: format!("Cannot resolve interaction factors for '{}'", col_name),
                }
            })?;
            let term_label = interaction_term_label(col_name, &factor_names);
            let mut factor_dummy_cols: Vec<Vec<Array1<f64>>> = Vec::new();
            for f in &factors {
                let cols =
                    term_col_arrays
                        .get(f)
                        .ok_or_else(|| crate::LmeError::NotImplemented {
                            feature: format!(
                            "Interaction '{}' requires main effect '{}' in the design matrix first",
                            col_name, f
                        ),
                        })?;
                factor_dummy_cols.push(cols.clone());
            }
            let mut interaction_cols: Vec<Array1<f64>> = Vec::new();
            if factor_dummy_cols.len() == 2 {
                for a_col in &factor_dummy_cols[0] {
                    for b_col in &factor_dummy_cols[1] {
                        interaction_cols.push(a_col * b_col);
                    }
                }
            } else {
                return Err(crate::LmeError::NotImplemented {
                    feature: format!(
                        "Interactions with {} factors are not supported yet",
                        factor_dummy_cols.len()
                    ),
                });
            }
            let mut dummy_names: Vec<String> = Vec::new();
            if factor_dummy_cols.len() == 2 && !factor_dummy_cols[0].is_empty() {
                let a_prefix = &factors[0];
                let b_prefix = &factors[1];
                let a_levels = extracted_levels.get(a_prefix).cloned().unwrap_or_default();
                let b_levels = extracted_levels.get(b_prefix).cloned().unwrap_or_default();
                let a_start = if intercept_handled && a_levels.len() > 1 {
                    1
                } else {
                    0
                };
                let b_start = if intercept_handled && b_levels.len() > 1 {
                    1
                } else {
                    0
                };
                for (ai, _) in factor_dummy_cols[0].iter().enumerate() {
                    for (bi, _) in factor_dummy_cols[1].iter().enumerate() {
                        let av = a_levels.get(a_start + ai).map(|s| s.as_str()).unwrap_or("");
                        let bv = b_levels.get(b_start + bi).map(|s| s.as_str()).unwrap_or("");
                        dummy_names.push(format!("{}:{}", av, bv));
                    }
                }
            }
            let int_start = fixed_cols.len();
            for (k, col) in interaction_cols.into_iter().enumerate() {
                let name = dummy_names
                    .get(k)
                    .map(|d| format!("{}{}", term_label, d))
                    .unwrap_or_else(|| format!("{}{}_i{}", term_label, col_name, k));
                fixed_cols.push(col);
                fixed_names.push(name);
                fixed_term_assign.push(term_label.clone());
            }
            term_col_arrays.insert(term_label.clone(), fixed_cols[int_start..].to_vec());
            continue;
        }

        let s = data
            .column(col_name)
            .map_err(|_| crate::LmeError::NotImplemented {
                feature: format!("Missing or invalid column: {}", col_name),
            })?;

        let is_categorical = match s.dtype() {
            DataType::String | DataType::Boolean => true,
            dt if dt.is_categorical() => true,
            _ => false,
        };

        if is_categorical {
            let unique_vals = if let Some(tr_levels) = training_levels {
                // If predicting, strictly use the training levels
                tr_levels.get(col_name).cloned().unwrap_or_else(Vec::new)
            } else {
                let unique_series = s.unique().map_err(|e| crate::LmeError::NotImplemented {
                    feature: format!("Failed to compute unique values for {}: {}", col_name, e),
                })?;
                let unique_cast = unique_series.cast(&DataType::String).map_err(|e| {
                    crate::LmeError::NotImplemented {
                        feature: format!(
                            "Cannot cast unique values of {} to string: {}",
                            col_name, e
                        ),
                    }
                })?;
                let unique_str = unique_cast.str().unwrap();

                let mut vals: Vec<String> = unique_str
                    .into_iter()
                    .flatten()
                    .map(|x| x.to_string())
                    .collect();
                vals.sort();
                vals
            };

            // Store for returning if we are training
            extracted_levels.insert(col_name.clone(), unique_vals.clone());

            let drop_first = intercept_handled;
            intercept_handled = true;

            let start_idx = if drop_first && unique_vals.len() > 1 {
                1
            } else {
                0
            };

            let full_str_col =
                s.cast(&DataType::String)
                    .map_err(|e| crate::LmeError::NotImplemented {
                        feature: format!("Cannot cast {} to string: {}", col_name, e),
                    })?;
            let str_data: Vec<String> = full_str_col
                .str()
                .unwrap()
                .into_iter()
                .map(|o| o.unwrap_or("").to_string())
                .collect();

            let mut term_cols: Vec<Array1<f64>> = Vec::new();
            for val in unique_vals.iter().skip(start_idx) {
                let mut col_data = ndarray::Array1::<f64>::zeros(n_obs);
                for i in 0..n_obs {
                    if str_data[i] == *val {
                        col_data[i] = 1.0;
                    }
                }
                term_cols.push(col_data.clone());
                fixed_cols.push(col_data);
                fixed_names.push(format!("{}{}", col_name, val));
                fixed_term_assign.push(col_name.clone());
            }
            if !term_cols.is_empty() {
                term_col_arrays.insert(col_name.clone(), term_cols);
            }
        } else {
            let s_cast =
                s.cast(&DataType::Float64)
                    .map_err(|_| crate::LmeError::NotImplemented {
                        feature: format!("Missing or invalid column: {}", col_name),
                    })?;
            let s_f64 = s_cast.f64().map_err(|_| crate::LmeError::NotImplemented {
                feature: format!("Missing or invalid column: {}", col_name),
            })?;

            let vec: Vec<f64> = s_f64.into_no_null_iter().collect();
            let col = Array1::from_vec(vec);
            fixed_cols.push(col.clone());
            fixed_names.push(col_name.clone());
            fixed_term_assign.push(col_name.clone());
            term_col_arrays.insert(col_name.clone(), vec![col]);
        }
    }

    let p = fixed_cols.len();
    let mut x = Array2::<f64>::zeros((n_obs, p));
    for (j, col) in fixed_cols.iter().enumerate() {
        x.column_mut(j).assign(col);
    }

    Ok((x, fixed_names, fixed_term_assign, extracted_levels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_df() -> DataFrame {
        df!(
            "y" => &[1.0, 2.0, 3.0, 4.0],
            "x" => &[1.0, 1.0, 2.0, 2.0],
            "group" => &["A", "A", "B", "B"],
            "subgroup" => &["a1", "a2", "b1", "b2"]
        )
        .unwrap()
    }

    #[test]
    fn test_no_random_effects() {
        let df = create_test_df();
        let ast = crate::formula::parse("y ~ x").unwrap();
        let matrices = build_design_matrices(&ast, &df).unwrap();

        assert_eq!(matrices.re_blocks.len(), 0);
        assert_eq!(matrices.zt.rows(), 0);
        assert_eq!(matrices.zt.cols(), 4);
    }

    #[test]
    fn test_nested_random_effects() {
        let df = create_test_df();
        // Nested random effect: formula syntax translates `(1 | group / subgroup)`
        // effectively to `(1 | group) + (1 | group:subgroup)` conceptually,
        // but here we just test if `group:subgroup` directly works if we parse it.
        // Fiasto transforms `(1 | group / subgroup)` into `1 | subgroup:group`
        let ast = crate::formula::parse("y ~ x + (1 | group:subgroup)").unwrap();
        let matrices = build_design_matrices(&ast, &df).unwrap();

        // We should have 1 block for `group:subgroup`
        assert_eq!(matrices.re_blocks.len(), 1);
        let block = &matrices.re_blocks[0];
        assert_eq!(block.group_name, "group:subgroup");
        assert_eq!(block.m, 4); // 4 unique interactions
    }
}
