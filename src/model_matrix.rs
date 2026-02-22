use crate::formula::FiastoModel;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;

pub struct DesignMatrices {
    pub x: Array2<f64>,
    pub zt: Array2<f64>,
    pub y: Array1<f64>,
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
    
    let y_series = data.column(&response_name)
        .map_err(|e| crate::LmeError::NotImplemented { feature: format!("Data missing response column: {}", e) })?;
    let y_vec: Vec<f64> = y_series.f64()
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
            if info.roles.contains(&"Identity".to_string()) && !info.roles.contains(&"RandomEffect".to_string()) && col_name != &response_name {
                let s = data.column(col_name).unwrap().f64().unwrap();
                let vec: Vec<f64> = s.into_no_null_iter().collect();
                fixed_cols.push(Array1::from_vec(vec));
            }
        }
    }
    
    let p = fixed_cols.len();
    let mut x = Array2::<f64>::zeros((n_obs, p));
    for (j, col) in fixed_cols.iter().enumerate() {
        x.column_mut(j).assign(col);
    }

    // 3. Build Random Effects Matrix (Z) -- highly simplified for (1 | Group)
    // We only support exactly one grouping variable with intercept-only for now
    let mut grouping_var = None;
    for (name, info) in &ast.columns {
        if info.roles.contains(&"GroupingVariable".to_string()) {
            grouping_var = Some(name.clone());
            break;
        }
    }
    
    let zt = if let Some(g_var) = grouping_var {
        let g_series = data.column(&g_var).unwrap().cast(&DataType::String).unwrap();
        let g_str = g_series.str().unwrap();
        
        // Find unique groups to set dimension q
        let mut unique_groups = Vec::new();
        let mut group_map = HashMap::new();
        
        for val_opt in g_str.into_iter() {
            let val = val_opt.unwrap().to_string();
            if !group_map.contains_key(&val) {
                group_map.insert(val.clone(), unique_groups.len());
                unique_groups.push(val);
            }
        }
        
        let q = unique_groups.len();
        let mut z_mat = Array2::<f64>::zeros((n_obs, q));
        
        for (i, val_opt) in g_str.into_iter().enumerate() {
            let val = val_opt.unwrap();
            let j = group_map[val];
            z_mat[[i, j]] = 1.0;
        }
        
        z_mat.t().to_owned()
    } else {
        Array2::<f64>::zeros((0, n_obs))
    };

    Ok(DesignMatrices { x, zt, y })
}
