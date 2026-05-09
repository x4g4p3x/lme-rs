//! LibFuzzer harness: parse then `build_design_matrices` on a synthetic `DataFrame`
//! built from every column name referenced in the AST (so random slopes / grouping /
//! offsets get a concrete column). Any panic or abort is a bug.

#![no_main]

use libfuzzer_sys::fuzz_target;
use lme_rs::formula::{parse, FiastoModel};
use lme_rs::model_matrix::build_design_matrices;
use polars::prelude::*;
use std::collections::HashSet;

const MAX_INPUT: usize = 32 * 1024;
const N_OBS: usize = 6;

/// Collect names that may be read from `data` during matrix construction.
fn referenced_column_names(ast: &FiastoModel) -> HashSet<String> {
    let mut out = HashSet::new();
    if let Some(o) = ast.offset.as_ref() {
        out.insert(o.clone());
    }
    for (name, info) in &ast.columns {
        out.insert(name.clone());
        if info.roles.contains(&"GroupingVariable".to_string()) && name.contains(':') {
            for part in name.split(':') {
                let p = part.trim();
                if !p.is_empty() {
                    out.insert(p.to_string());
                }
            }
        }
        for re in &info.random_effects {
            if let Some(vars) = &re.variables {
                for v in vars {
                    out.insert(v.clone());
                }
            }
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

fn synthetic_df(ast: &FiastoModel) -> DataFrame {
    let names = referenced_column_names(ast);
    let mut cols: Vec<Column> = Vec::with_capacity(names.len());
    for name in names {
        // Float columns cast to string for grouping factors in `model_matrix`.
        let v: Vec<f64> = (0..N_OBS).map(|i| (i as f64) * 0.25).collect();
        cols.push(Series::new(name.as_str().into(), v).into());
    }
    DataFrame::new(cols).unwrap()
}

fuzz_target!(|data: &[u8]| {
    let slice = if data.len() > MAX_INPUT {
        &data[..MAX_INPUT]
    } else {
        data
    };
    let s = String::from_utf8_lossy(slice);
    let ast = match parse(s.as_ref()) {
        Ok(a) => a,
        Err(_) => return,
    };
    let df = synthetic_df(&ast);
    let _ = build_design_matrices(&ast, &df);
});
