use crate::formula::FiastoModel;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;

use sprs::TriMat;

/// Captures structural layout and variable indices for a multi-dimensional Random Effect correlation block.
#[derive(Clone, Debug)]
pub struct ReBlock {
    pub m: usize,
    pub k: usize,
    pub theta_len: usize,
    pub group_name: String,
    pub effect_names: Vec<String>,
}

/// Container encompassing the finalized design matrices and mapping arrays extracted from input DataFrames.
pub struct DesignMatrices {
    pub formula: String,
    pub x: Array2<f64>,
    pub zt: sprs::CsMat<f64>,
    pub y: Array1<f64>,
    pub re_blocks: Vec<ReBlock>,
    pub fixed_names: Vec<String>,
}

/// Constructs structural matrices formatting Fixed Effects ($X$) and Random Effects ($Z$) bounds out of `fiasto` inputs.
pub fn build_design_matrices(
    ast: &FiastoModel,
    data: &DataFrame,
) -> crate::Result<DesignMatrices> {
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
        feature: "Missing response variable".to_string() 
    })?;
    
    let y_series_cast = data.column(&response_name)
        .map_err(|e| crate::LmeError::NotImplemented { feature: format!("Data missing response column: {}", e) })?
        .cast(&DataType::Float64).unwrap();
    let y_vec: Vec<f64> = y_series_cast.f64()
        .map_err(|_| crate::LmeError::NotImplemented { feature: "Response must be float".to_string() })?
        .into_no_null_iter()
        .collect();
    let y = Array1::from_vec(y_vec);

    let (x, fixed_names) = build_x_matrix(ast, data, &response_name, n_obs)?;

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
            
            // Gap 5: Intercept detection follows R convention — always included unless
            // explicitly suppressed (e.g., `(0 + Days | Subject)`). Since fiasto does not
            // currently emit a reliable suppression signal for complex formulas, we default
            // to true. When fiasto gains explicit `0 +` support, this logic should check
            // for the suppression flag.
            let has_intercept = true; // R convention default

            let g_series = data.column(g_var).unwrap().cast(&DataType::String).unwrap();
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
            let k = if has_intercept { 1 + slope_vars.len() } else { slope_vars.len() };
            let q_block = m * k;
            
            let mut slope_data: Vec<Vec<f64>> = Vec::new();
            for s_var in &slope_vars {
                    let s_series = data.column(s_var).unwrap().cast(&DataType::Float64).unwrap();
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
                
                re_blocks.push(ReBlock { m, k, theta_len, group_name: g_var.to_string(), effect_names });
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
    })
}

/// Evaluates the `ast` structural formula to isolate and resolve pure population-level ($X$) design matrices.
pub fn build_x_matrix(
    ast: &FiastoModel,
    data: &DataFrame,
    response_name: &str,
    n_obs: usize,
) -> crate::Result<(Array2<f64>, Vec<String>)> {
    let mut fixed_cols = Vec::new();
    let mut fixed_names = Vec::new();
    if ast.metadata.has_intercept {
        fixed_cols.push(Array1::<f64>::ones(n_obs));
        fixed_names.push("(Intercept)".to_string());
    }
    
    // Ordered columns per the formula
    for col_name in ast.all_generated_columns.iter() {
        if let Some(info) = ast.columns.get(col_name) {
            if info.roles.contains(&"Identity".to_string()) && col_name != response_name {
                let mut col_found = false;
                if let Ok(s) = data.column(col_name) {
                    if let Ok(s_cast) = s.cast(&DataType::Float64) {
                        if let Ok(s_f64) = s_cast.f64() {
                            let vec: Vec<f64> = s_f64.into_no_null_iter().collect();
                            fixed_cols.push(Array1::from_vec(vec));
                            fixed_names.push(col_name.clone());
                            col_found = true;
                        }
                    }
                }
                if !col_found {
                    // Fail if identity column is missing from new prediction sets
                    return Err(crate::LmeError::NotImplemented { feature: format!("Missing or invalid column: {}", col_name) });
                }
            }
        }
    }
    
    let p = fixed_cols.len();
    let mut x = Array2::<f64>::zeros((n_obs, p));
    for (j, col) in fixed_cols.iter().enumerate() {
        x.column_mut(j).assign(col);
    }
    
    Ok((x, fixed_names))
}
