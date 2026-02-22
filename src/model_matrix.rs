use crate::formula::FiastoModel;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;

use sprs::TriMat;

pub struct DesignMatrices {
    pub x: Array2<f64>,
    pub zt: sprs::CsMat<f64>,
    pub y: Array1<f64>,
    pub theta_len: usize,
}

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

    // 2. Build Fixed Effects Matrix (X)
    let mut fixed_cols = Vec::new();
    if ast.metadata.has_intercept {
        fixed_cols.push(Array1::<f64>::ones(n_obs));
    }
    
    // Ordered columns per the formula
    for col_name in ast.all_generated_columns.iter() {
        if let Some(info) = ast.columns.get(col_name) {
            if info.roles.contains(&"Identity".to_string()) && col_name != &response_name {
                let s = data.column(col_name).unwrap().cast(&DataType::Float64).unwrap();
                let s_f64 = s.f64().unwrap();
                let vec: Vec<f64> = s_f64.into_no_null_iter().collect();
                fixed_cols.push(Array1::from_vec(vec));
            }
        }
    }
    
    let p = fixed_cols.len();
    let mut x = Array2::<f64>::zeros((n_obs, p));
    for (j, col) in fixed_cols.iter().enumerate() {
        x.column_mut(j).assign(col);
    }

    // 3. Build Random Effects Matrix (Z) -- now supporting Random Slopes!
    let mut grouping_var = None;
    let mut slope_vars = Vec::new();

    for (name, info) in &ast.columns {
        if info.roles.contains(&"GroupingVariable".to_string()) {
            grouping_var = Some(name.clone());
            if let Some(re) = info.random_effects.first() {
                if let Some(vars) = &re.variables {
                    slope_vars = vars.clone();
                }
            }
            break;
        }
    }
    
    let (zt, theta_len) = if let Some(g_var) = grouping_var {
        let g_series = data.column(&g_var).unwrap().cast(&DataType::String).unwrap();
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
        let k = 1 + slope_vars.len(); // intercept + slopes
        let q = m * k;
        let mut z_tri = TriMat::new((n_obs, q));
        
        let mut slope_data: Vec<Vec<f64>> = Vec::new();
        for s_var in &slope_vars {
            let s_series = data.column(s_var).unwrap().cast(&DataType::Float64).unwrap();
            let s_f64_series = s_series.f64().unwrap();
            slope_data.push(s_f64_series.into_no_null_iter().collect());
        }

        for (i, val_opt) in g_str.into_iter().enumerate() {
            let val = val_opt.unwrap();
            let group_idx = group_map[val];
            let offset = group_idx * k;
            
            z_tri.add_triplet(i, offset, 1.0);
            
            for (slope_idx, s_vec) in slope_data.iter().enumerate() {
                z_tri.add_triplet(i, offset + 1 + slope_idx, s_vec[i]);
            }
        }
        
        let theta_len = k * (k + 1) / 2;
        let z_csc = z_tri.to_csc();
        let zt_csr = z_csc.transpose_into();
        (zt_csr, theta_len)
    } else {
        (sprs::CsMat::zero((0, n_obs)), 1)
    };

    Ok(DesignMatrices { x, zt, y, theta_len })
}
