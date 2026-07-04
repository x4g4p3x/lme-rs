//! Fixed-effect contrast matrices for Type II and Type III ANOVA (`lmerTest`-style).

use ndarray::Array2;
use std::collections::HashMap;

/// ANOVA sum-of-squares type for fixed-effect terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnovaType {
    /// Type I: sequential contrasts in formula term order (each term after preceding terms).
    Type1,
    /// Type II: terms not contained elsewhere use marginal contrasts; contained terms use sequential (Doolittle) contrasts.
    Type2,
    /// Type III: marginal contrasts for each term (default).
    #[default]
    Type3,
}

/// One row per ANOVA table entry (excluding intercept).
#[derive(Debug, Clone)]
pub struct AnovaTerm {
    /// Display name (e.g. `Days`, `cask`, `trt:blk`).
    pub name: String,
    /// Coefficient column indices belonging to this term.
    pub col_indices: Vec<usize>,
}

/// Build ANOVA term list from per-column term labels (same length as `fixed_names`).
pub fn anova_terms_from_assign(
    fixed_names: &[String],
    fixed_term_assign: &[String],
) -> Vec<AnovaTerm> {
    debug_assert_eq!(fixed_names.len(), fixed_term_assign.len());
    let mut order: Vec<String> = Vec::new();
    let mut cols_by_term: HashMap<String, Vec<usize>> = HashMap::new();

    for (j, label) in fixed_term_assign.iter().enumerate() {
        if label == "(Intercept)" || fixed_names.get(j).is_some_and(|n| n == "(Intercept)") {
            continue;
        }
        if !order.contains(label) {
            order.push(label.clone());
        }
        cols_by_term.entry(label.clone()).or_default().push(j);
    }

    order
        .into_iter()
        .map(|name| AnovaTerm {
            col_indices: cols_by_term.remove(&name).unwrap_or_default(),
            name,
        })
        .collect()
}

/// Factor names for a model term label (`trt:blk` → `["trt", "blk"]`).
pub fn term_factors(term: &str) -> Vec<String> {
    if term.contains(':') {
        term.split(':').map(|s| s.to_string()).collect()
    } else {
        vec![term.to_string()]
    }
}

/// `containment[term]` lists terms that strictly contain `term` (more factors).
pub fn term_containment(term_names: &[String]) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for t in term_names {
        let ft = term_factors(t);
        let mut contained_by = Vec::new();
        for u in term_names {
            if t == u {
                continue;
            }
            let fu = term_factors(u);
            if fu.len() > ft.len() && ft.iter().all(|f| fu.contains(f)) {
                contained_by.push(u.clone());
            }
        }
        out.insert(t.clone(), contained_by);
    }
    out
}

/// Marginal contrast: rows of identity on `col_indices`.
pub fn marginal_contrast(p: usize, col_indices: &[usize]) -> Array2<f64> {
    let q = col_indices.len();
    let mut l = Array2::<f64>::zeros((q, p));
    for (row, &j) in col_indices.iter().enumerate() {
        l[[row, j]] = 1.0;
    }
    l
}

/// Doolittle / Cholesky factor L with `XtX = L L'` (lower triangular), matching `lmerTest::doolittle`.
fn doolittle_lower(xtx: &Array2<f64>) -> Array2<f64> {
    let p = xtx.nrows();
    let mut l = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..=i {
            let mut s = xtx[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = s.max(0.0).sqrt();
            } else {
                l[[i, j]] = if l[[j, j]].abs() > 1e-15 {
                    s / l[[j, j]]
                } else {
                    0.0
                };
            }
        }
    }
    l
}

/// Type II contrast matrix for one term (`lmerTest::get_contrasts_type2`).
pub fn type2_contrast(
    x: &Array2<f64>,
    col_terms: &[String],
    term: &str,
    contained: &[String],
) -> Array2<f64> {
    let p = x.ncols();
    if p == 0 {
        return Array2::zeros((0, 0));
    }

    let mut cols_term: Vec<usize> = (0..p)
        .filter(|&j| col_terms[j] == term || contained.iter().any(|t| col_terms[j] == *t))
        .collect();
    let cols_rest: Vec<usize> = (0..p).filter(|j| !cols_term.contains(j)).collect();
    cols_term.sort_unstable();
    let col_order: Vec<usize> = cols_rest.into_iter().chain(cols_term).collect();

    let x_new = {
        let mut out = Array2::<f64>::zeros((x.nrows(), p));
        for (new_j, &old_j) in col_order.iter().enumerate() {
            out.column_mut(new_j).assign(&x.column(old_j));
        }
        out
    };
    let new_col_terms: Vec<String> = col_order.iter().map(|&j| col_terms[j].clone()).collect();

    let xtx = x_new.t().dot(&x_new);
    let l_lower = doolittle_lower(&xtx);
    let lc = l_lower.t(); // t(doolittle(...)$L) in R

    let term_rows: Vec<usize> = (0..p).filter(|&j| new_col_terms[j] == term).collect();
    let q = term_rows.len();
    let mut l_term = Array2::<f64>::zeros((q, p));
    for (out_row, &r) in term_rows.iter().enumerate() {
        for old_j in 0..p {
            let orig_j = col_order[old_j];
            l_term[[out_row, orig_j]] = lc[[r, old_j]];
        }
    }
    l_term
}

/// Contrast matrix for a term under the chosen ANOVA type.
pub fn contrast_for_term(
    anova_type: AnovaType,
    x: &Array2<f64>,
    col_terms: &[String],
    term: &str,
    col_indices: &[usize],
    containment: &HashMap<String, Vec<String>>,
    term_order: &[String],
) -> Array2<f64> {
    let p = x.ncols();
    match anova_type {
        AnovaType::Type3 => marginal_contrast(p, col_indices),
        AnovaType::Type1 => {
            let pos = term_order
                .iter()
                .position(|t| t == term)
                .unwrap_or(term_order.len());
            let prior = &term_order[..pos];
            type2_contrast(x, col_terms, term, prior)
        }
        AnovaType::Type2 => {
            let contained = containment.get(term).map(|v| v.as_slice()).unwrap_or(&[]);
            if contained.is_empty() {
                marginal_contrast(p, col_indices)
            } else {
                type2_contrast(x, col_terms, term, contained)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containment_marks_interaction() {
        let c = term_containment(&["trt".into(), "blk".into(), "trt:blk".into()]);
        assert!(c["trt"].contains(&"trt:blk".to_string()));
        assert!(c["blk"].contains(&"trt:blk".to_string()));
        assert!(c["trt:blk"].is_empty());
    }

    #[test]
    fn type1_first_term_matches_marginal_on_additive_model() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        )
        .unwrap();
        let col_terms = vec!["(Intercept)".into(), "a".into(), "b".into()];
        let order = vec!["a".into(), "b".into()];
        let l1 = contrast_for_term(
            AnovaType::Type1,
            &x,
            &col_terms,
            "a",
            &[1],
            &term_containment(&order),
            &order,
        );
        let l3 = marginal_contrast(3, &[1]);
        assert!((l1 - l3).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn type2_equals_marginal_for_main_effect_in_additive_model() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        )
        .unwrap();
        let col_terms = vec!["(Intercept)".into(), "a".into(), "b".into()];
        let order = vec!["a".into(), "b".into()];
        let c = term_containment(&order);
        let l2 = contrast_for_term(AnovaType::Type2, &x, &col_terms, "a", &[1], &c, &order);
        let l3 = marginal_contrast(3, &[1]);
        assert!((l2 - l3).iter().all(|v| v.abs() < 1e-10));
    }
}
