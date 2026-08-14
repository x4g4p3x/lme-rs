//! Multiple comparisons for a categorical fixed-effect term (`multcomp::glht` / `mcp`).
//!
//! Pairwise (Tukey) and vs-control (Dunnett) Wald tests with Bonferroni, Holm, or
//! Tukey–Kramer (`stats::ptukey`) p-value adjustment. Default inference is asymptotic
//! `z`, matching `multcomp::glht` on `merMod`; pass a [`DdfMethod`](crate::anova::DdfMethod)
//! for `t` tests with Satterthwaite or Kenward–Roger denominator df.

use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use crate::anova::DdfMethod;
use crate::contrast::{fixed_effect_contrast_test, fixed_effect_vcov};
use crate::studentized_range::tukey_kramer_p;
use crate::LmeError;
use crate::LmeFit;

/// Multiple-comparison contrast family (`multcomp::mcp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpType {
    /// All pairwise differences of factor levels (`mcp(factor = "Tukey")`).
    Tukey,
    /// Each non-control level versus a control (`mcp(factor = "Dunnett")`).
    ///
    /// `control` is a level label; `None` uses the first sorted level (treatment
    /// reference), matching R's default control.
    Dunnett {
        /// Control level, or `None` for the first sorted level.
        control: Option<String>,
    },
}

/// P-value adjustment for an MCP family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpAdjust {
    /// Unadjusted two-sided Wald p-values.
    None,
    /// Bonferroni: `min(1, m p)`.
    Bonferroni,
    /// Holm step-down (R `p.adjust(..., "holm")`).
    Holm,
    /// Tukey–Kramer via the studentized range (`emmeans` `adjust = "tukey"`).
    ///
    /// Uses `nmeans` = number of factor levels and each contrast's denominator df
    /// (`Inf` for Wald `z`). Not valid for Dunnett families.
    Tukey,
}

/// One family of linear hypotheses for a categorical term.
#[derive(Debug, Clone)]
pub struct GlhtResult {
    /// Factor / ANOVA term that was compared.
    pub term: String,
    /// Contrast family.
    pub mcp: McpType,
    /// Adjustment applied to [`Self::p_adjust`].
    pub adjust: McpAdjust,
    /// `"z"` (Wald) or `"t"` (Satterthwaite / Kenward–Roger).
    pub statistic: String,
    /// Labels such as `"b - a"`.
    pub comparisons: Vec<String>,
    /// `L β` for each contrast row.
    pub estimate: Array1<f64>,
    /// `√(L V L')`.
    pub std_error: Array1<f64>,
    /// `estimate / std_error`.
    pub statistic_values: Array1<f64>,
    /// Denominator df (`∞` for Wald `z`).
    pub den_df: Array1<f64>,
    /// Unadjusted two-sided p-values.
    pub p_value: Array1<f64>,
    /// Multiplicity-adjusted p-values.
    pub p_adjust: Array1<f64>,
}

impl LmeFit {
    /// Multiple comparisons for a categorical fixed-effect term.
    ///
    /// `ddf = None` uses Wald `z` (the `multcomp::glht` default for `merMod`).
    /// Tukey–Kramer adjustment (`McpAdjust::Tukey`) is only valid for
    /// [`McpType::Tukey`].
    pub fn glht(
        &self,
        term: &str,
        mcp: McpType,
        adjust: McpAdjust,
        ddf: Option<DdfMethod>,
    ) -> crate::Result<GlhtResult> {
        if matches!(mcp, McpType::Dunnett { .. }) && matches!(adjust, McpAdjust::Tukey) {
            return Err(LmeError::NotImplemented {
                feature: "Tukey–Kramer adjustment is for all-pairwise MCP; use Bonferroni or Holm for Dunnett".to_string(),
            });
        }

        let (l_mat, comparisons, n_groups) = mcp_contrast_matrix(self, term, &mcp)?;
        let v_beta = fixed_effect_vcov(self)?;
        let beta = &self.coefficients;
        let m = l_mat.nrows();
        let mut estimate = Array1::<f64>::zeros(m);
        let mut std_error = Array1::<f64>::zeros(m);
        let mut statistic_values = Array1::<f64>::zeros(m);
        let mut den_df = Array1::<f64>::zeros(m);
        let mut p_value = Array1::<f64>::zeros(m);
        let norm = Normal::new(0.0, 1.0).expect("standard normal");

        for i in 0..m {
            let row = l_mat.row(i).to_owned();
            let est = row.dot(beta);
            let lv = v_beta.dot(&row);
            let var = row.dot(&lv);
            if !var.is_finite() || var <= 0.0 {
                return Err(LmeError::NotImplemented {
                    feature: format!("Non-positive contrast variance for '{}'", comparisons[i]),
                });
            }
            let se = var.sqrt();
            let stat = est / se;
            estimate[i] = est;
            std_error[i] = se;
            statistic_values[i] = stat;

            match ddf {
                None => {
                    den_df[i] = f64::INFINITY;
                    p_value[i] = (2.0 * (1.0 - norm.cdf(stat.abs()))).clamp(0.0, 1.0);
                }
                Some(method) => {
                    let row_mat = l_mat.slice(ndarray::s![i..i + 1, ..]).to_owned();
                    let test = fixed_effect_contrast_test(self, &row_mat, method, None)?;
                    den_df[i] = test.den_df;
                    p_value[i] = test.p_value;
                }
            }
        }

        let p_adjust = adjust_p_values(
            adjust,
            p_value.as_slice().unwrap(),
            n_groups,
            &statistic_values,
            &den_df,
        )?;
        let statistic = if ddf.is_some() { "t" } else { "z" }.to_string();

        Ok(GlhtResult {
            term: term.to_string(),
            mcp,
            adjust,
            statistic,
            comparisons,
            estimate,
            std_error,
            statistic_values,
            den_df,
            p_value,
            p_adjust: Array1::from(p_adjust),
        })
    }
}

/// Contrast matrix and labels for a Tukey or Dunnett family on `term`.
pub fn mcp_contrast_matrix(
    fit: &LmeFit,
    term: &str,
    mcp: &McpType,
) -> crate::Result<(Array2<f64>, Vec<String>, usize)> {
    let levels = fit
        .categorical_levels
        .as_ref()
        .and_then(|m| m.get(term))
        .cloned()
        .ok_or_else(|| LmeError::NotImplemented {
            feature: format!(
                "MCP requires a categorical term with stored levels; '{term}' was not found"
            ),
        })?;
    if levels.len() < 2 {
        return Err(LmeError::NotImplemented {
            feature: format!("MCP needs at least two levels for '{term}'"),
        });
    }

    let names = fit.fixed_names.clone().unwrap_or_default();
    let p = fit.coefficients.len();
    if names.len() != p {
        return Err(LmeError::NotImplemented {
            feature: "Fixed-effect names missing or mismatched".to_string(),
        });
    }

    let pairs: Vec<(usize, usize)> = match mcp {
        McpType::Tukey => (0..levels.len())
            .flat_map(|i| ((i + 1)..levels.len()).map(move |j| (i, j)))
            .collect(),
        McpType::Dunnett { control } => {
            let cidx = match control {
                Some(label) => levels.iter().position(|l| l == label).ok_or_else(|| {
                    LmeError::NotImplemented {
                        feature: format!("Dunnett control '{label}' is not a level of '{term}'"),
                    }
                })?,
                None => 0,
            };
            (0..levels.len())
                .filter(|&i| i != cidx)
                .map(|i| (cidx, i))
                .collect()
        }
    };

    let mut l_mat = Array2::<f64>::zeros((pairs.len(), p));
    let mut comparisons = Vec::with_capacity(pairs.len());
    for (row, &(i, j)) in pairs.iter().enumerate() {
        fill_pairwise_row(&mut l_mat, row, &names, term, &levels, i, j)?;
        comparisons.push(format!("{} - {}", levels[j], levels[i]));
    }
    Ok((l_mat, comparisons, levels.len()))
}

fn dummy_column(names: &[String], factor: &str, level: &str) -> Option<usize> {
    let want = format!("{factor}{level}");
    names.iter().position(|n| n == &want)
}

fn fill_pairwise_row(
    l_mat: &mut Array2<f64>,
    row: usize,
    names: &[String],
    factor: &str,
    levels: &[String],
    from: usize,
    to: usize,
) -> crate::Result<()> {
    let has_intercept = names.first().is_some_and(|n| n == "(Intercept)");
    let ref_dummy = dummy_column(names, factor, &levels[0]);
    if has_intercept && ref_dummy.is_some() {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Over-parameterized dummy coding for '{factor}'; cannot build MCP contrasts"
            ),
        });
    }

    apply_level_weight(l_mat, row, names, factor, &levels[to], has_intercept, 1.0)?;
    apply_level_weight(
        l_mat,
        row,
        names,
        factor,
        &levels[from],
        has_intercept,
        -1.0,
    )?;
    Ok(())
}

fn apply_level_weight(
    l_mat: &mut Array2<f64>,
    row: usize,
    names: &[String],
    factor: &str,
    level: &str,
    has_intercept: bool,
    weight: f64,
) -> crate::Result<()> {
    if let Some(j) = dummy_column(names, factor, level) {
        l_mat[[row, j]] += weight;
        return Ok(());
    }
    if has_intercept && names.first().is_some_and(|n| n == "(Intercept)") {
        // Treatment reference: absorbed in the intercept; pairwise diffs cancel it.
        return Ok(());
    }
    Err(LmeError::NotImplemented {
        feature: format!("No dummy column for {factor} level '{level}'"),
    })
}

fn adjust_p_values(
    adjust: McpAdjust,
    raw: &[f64],
    n_groups: usize,
    stats: &Array1<f64>,
    den_df: &Array1<f64>,
) -> crate::Result<Vec<f64>> {
    let m = raw.len();
    match adjust {
        McpAdjust::None => Ok(raw.to_vec()),
        McpAdjust::Bonferroni => Ok(raw.iter().map(|p| (m as f64 * p).min(1.0)).collect()),
        McpAdjust::Holm => Ok(holm_adjust(raw)),
        McpAdjust::Tukey => {
            let mut out = Vec::with_capacity(m);
            for i in 0..m {
                out.push(tukey_kramer_p(stats[i], n_groups, den_df[i]));
            }
            Ok(out)
        }
    }
}

fn holm_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; m];
    let mut running = 0.0_f64;
    for (rank, &i) in order.iter().enumerate() {
        let adj = ((m - rank) as f64 * p[i]).min(1.0);
        running = running.max(adj);
        out[i] = running;
    }
    out
}

impl fmt::Display for GlhtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let family = match &self.mcp {
            McpType::Tukey => "Tukey",
            McpType::Dunnett { .. } => "Dunnett",
        };
        let adj = match self.adjust {
            McpAdjust::None => "none",
            McpAdjust::Bonferroni => "Bonferroni",
            McpAdjust::Holm => "Holm",
            McpAdjust::Tukey => "Tukey-Kramer",
        };
        let inf = if self.statistic == "z" { "Wald z" } else { "t" };
        writeln!(f, "General linear hypotheses ({family} MCP, {inf}, {adj})")?;
        writeln!(f, "Term: {}", self.term)?;
        let stat_hdr = if self.statistic == "z" { "z" } else { "t" };
        writeln!(
            f,
            "{:<12} {:>10} {:>10} {:>10} {:>12} {:>12}",
            "Contrast", "Estimate", "SE", stat_hdr, "Pr(>|stat|)", "p adj"
        )?;
        for i in 0..self.comparisons.len() {
            writeln!(
                f,
                "{:<12} {:>10.4} {:>10.4} {:>10.4} {:>12.4e} {:>12.4e}",
                self.comparisons[i],
                self.estimate[i],
                self.std_error[i],
                self.statistic_values[i],
                self.p_value[i],
                self.p_adjust[i]
            )?;
        }
        Ok(())
    }
}
