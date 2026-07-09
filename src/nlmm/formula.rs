//! Parse `nlmer`-style three-part formulas.

use crate::LmeError;

/// Parsed nonlinear mixed-model formula.
///
/// Example: `circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree`
#[derive(Debug, Clone)]
pub struct NlmerFormula {
    /// Response column name.
    pub response: String,
    /// Covariate passed to the mean function (e.g. `age`).
    pub covariate: String,
    /// Names of fixed nonlinear parameters (`Asym`, `xmid`, `scal`, …).
    pub fixed_param_names: Vec<String>,
    /// Nonlinear parameters that carry random effects (subset of `fixed_param_names`).
    pub re_params: Vec<String>,
    /// Grouping factor column (e.g. `Tree`).
    pub re_group: String,
}

/// Supported nonlinear mean functions in the middle part of the formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlmmMeanKind {
    /// `SSlogis(covariate, Asym, xmid, scal)`
    Sslogis,
    /// `SSasymp(covariate, Asym, R0, lrc)`
    Ssasymp,
    /// `SSfol(covariate, Asym, R0, lrc)` — same mean as `SSasymp`
    Ssfol,
    /// `SSmicmen(covariate, Vmax, K)`
    Ssmicmen,
    /// `SSgompertz(covariate, Asym, b2, b3)` — `Asym * exp(-b2 * b3^x)`
    Ssgompertz,
    /// `SSpower(covariate, a, b, c)` — `a * x^b + c` (MATLAB Curve Fitter `power2`)
    Sspower,
}

impl NlmmMeanKind {
    pub(crate) fn n_params(self) -> usize {
        match self {
            Self::Ssmicmen => 2,
            Self::Sslogis | Self::Ssasymp | Self::Ssfol | Self::Ssgompertz | Self::Sspower => 3,
        }
    }

    pub(crate) fn expected_arg_count(self) -> usize {
        1 + self.n_params()
    }

    /// Asymptotic means (`SSasymp` / `SSfol`) use RSS-based σ² profiling in scalar-RE fits.
    pub(crate) fn uses_scalar_rss_sigma(self) -> bool {
        matches!(self, Self::Ssasymp | Self::Ssfol)
    }
}

/// Parse a three-part custom-mean formula: `response ~ covariate ~ re | group`.
///
/// The middle segment is the covariate column name (not an `SS*` call). Pass the
/// full list of fixed nonlinear parameter names via `fixed_param_names`.
pub fn parse_nlmer_custom_formula(
    formula: &str,
    fixed_param_names: &[String],
) -> crate::Result<NlmerFormula> {
    let parts: Vec<&str> = formula.split('~').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "custom nlmer formulas require three parts separated by '~' (got {} segments)",
                parts.len()
            ),
        });
    }
    if fixed_param_names.is_empty() {
        return Err(LmeError::NotImplemented {
            feature: "custom nlmer requires at least one fixed nonlinear parameter name"
                .to_string(),
        });
    }
    let response = parts[0].to_string();
    let covariate = parts[1].to_string();
    if covariate.is_empty() {
        return Err(LmeError::NotImplemented {
            feature: "custom nlmer covariate name is empty".to_string(),
        });
    }
    if covariate.contains('(') || covariate.contains(')') {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "custom nlmer middle part must be a covariate column name, not a function call (got '{covariate}')"
            ),
        });
    }
    let (re_params, re_group) = parse_random_part(parts[2], fixed_param_names)?;
    Ok(NlmerFormula {
        response,
        covariate,
        fixed_param_names: fixed_param_names.to_vec(),
        re_params,
        re_group,
    })
}

/// Parse a three-part `nlmer` formula string.
pub fn parse_nlmer_formula(formula: &str) -> crate::Result<(NlmerFormula, NlmmMeanKind)> {
    let parts: Vec<&str> = formula.split('~').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "nlmer formulas require three parts separated by '~' (got {} segments)",
                parts.len()
            ),
        });
    }
    let response = parts[0].to_string();
    let nonlinear = parts[1];
    let random = parts[2];

    let (mean_kind, covariate, param_names) = parse_nonlinear_part(nonlinear)?;
    let (re_params, re_group) = parse_random_part(random, &param_names)?;

    Ok((
        NlmerFormula {
            response,
            covariate,
            fixed_param_names: param_names,
            re_params,
            re_group,
        },
        mean_kind,
    ))
}

fn parse_nonlinear_part(s: &str) -> crate::Result<(NlmmMeanKind, String, Vec<String>)> {
    let open = s.find('(').ok_or_else(|| LmeError::NotImplemented {
        feature: format!("Nonlinear part must be a function call, got '{s}'"),
    })?;
    let fname = s[..open].trim();
    let close = s.rfind(')').ok_or_else(|| LmeError::NotImplemented {
        feature: format!("Missing ')' in nonlinear part '{s}'"),
    })?;
    let inner = &s[open + 1..close];
    let args: Vec<&str> = inner.split(',').map(str::trim).collect();
    let mean = match fname {
        "SSlogis" => NlmmMeanKind::Sslogis,
        "SSasymp" => NlmmMeanKind::Ssasymp,
        "SSfol" => NlmmMeanKind::Ssfol,
        "SSmicmen" => NlmmMeanKind::Ssmicmen,
        "SSgompertz" => NlmmMeanKind::Ssgompertz,
        "SSpower" => NlmmMeanKind::Sspower,
        other => {
            return Err(LmeError::NotImplemented {
                feature: format!(
                    "Unsupported nonlinear mean '{other}' (supported: SSlogis, SSasymp, SSfol, SSmicmen, SSgompertz, SSpower)"
                ),
            });
        }
    };
    if args.len() != mean.expected_arg_count() {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Nonlinear mean '{fname}' requires {} arguments (covariate + {} parameters), got {}",
                mean.expected_arg_count(),
                mean.n_params(),
                args.len()
            ),
        });
    }
    Ok((
        mean,
        args[0].to_string(),
        args[1..].iter().map(|s| s.to_string()).collect(),
    ))
}

fn parse_random_part(s: &str, fixed_names: &[String]) -> crate::Result<(Vec<String>, String)> {
    let parts: Vec<&str> = s.split('|').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(LmeError::NotImplemented {
            feature: format!("Random part must be 'param|group' or 'p1 + p2 | group', got '{s}'"),
        });
    }
    let re_params: Vec<String> = parts[0]
        .split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect();
    if re_params.is_empty() {
        return Err(LmeError::NotImplemented {
            feature: format!("Random part must name at least one parameter, got '{s}'"),
        });
    }
    for param in &re_params {
        if !fixed_names.iter().any(|n| n == param) {
            return Err(LmeError::NotImplemented {
                feature: format!(
                    "Random-effect parameter '{param}' is not a fixed nonlinear parameter ({fixed_names:?})"
                ),
            });
        }
    }
    Ok((re_params, parts[1].to_string()))
}

/// Map random-effect parameter names to indices in the fixed parameter vector.
pub(crate) fn re_param_indices(parsed: &NlmerFormula) -> crate::Result<Vec<usize>> {
    parsed
        .re_params
        .iter()
        .map(|name| {
            parsed
                .fixed_param_names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| LmeError::NotImplemented {
                    feature: format!("RE parameter '{name}' missing from fixed parameters"),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_orange_formula() {
        let (f, kind) =
            parse_nlmer_formula("circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree")
                .unwrap();
        assert_eq!(kind, NlmmMeanKind::Sslogis);
        assert_eq!(f.response, "circumference");
        assert_eq!(f.covariate, "age");
        assert_eq!(f.re_group, "Tree");
        assert_eq!(f.re_params, vec!["Asym".to_string()]);
    }

    #[test]
    fn parses_multi_re_formula() {
        let (f, _) = parse_nlmer_formula(
            "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree",
        )
        .unwrap();
        assert_eq!(f.re_params, vec!["Asym".to_string(), "xmid".to_string()]);
    }

    #[test]
    fn parses_ssasymp_formula() {
        let (f, kind) = parse_nlmer_formula("y ~ SSasymp(x, Asym, R0, lrc) ~ Asym|id").unwrap();
        assert_eq!(kind, NlmmMeanKind::Ssasymp);
        assert_eq!(f.fixed_param_names, vec!["Asym", "R0", "lrc"]);
    }

    #[test]
    fn parses_ssfol_formula() {
        let (f, kind) = parse_nlmer_formula("y ~ SSfol(x, Asym, R0, lrc) ~ Asym|id").unwrap();
        assert_eq!(kind, NlmmMeanKind::Ssfol);
        assert_eq!(f.fixed_param_names, vec!["Asym", "R0", "lrc"]);
    }

    #[test]
    fn parses_ssmicmen_formula() {
        let (f, kind) = parse_nlmer_formula("y ~ SSmicmen(x, Vmax, K) ~ Vmax|id").unwrap();
        assert_eq!(kind, NlmmMeanKind::Ssmicmen);
        assert_eq!(f.fixed_param_names, vec!["Vmax", "K"]);
    }

    #[test]
    fn parses_ssgompertz_formula() {
        let (f, kind) = parse_nlmer_formula("y ~ SSgompertz(x, Asym, b2, b3) ~ Asym|id").unwrap();
        assert_eq!(kind, NlmmMeanKind::Ssgompertz);
        assert_eq!(f.fixed_param_names, vec!["Asym", "b2", "b3"]);
    }

    #[test]
    fn parses_sspower_formula() {
        let (f, kind) = parse_nlmer_formula("y ~ SSpower(x, a, b, c) ~ c|id").unwrap();
        assert_eq!(kind, NlmmMeanKind::Sspower);
        assert_eq!(f.fixed_param_names, vec!["a", "b", "c"]);
        assert_eq!(f.re_params, vec!["c".to_string()]);
    }

    #[test]
    fn parses_custom_mean_formula() {
        let names = vec!["a".to_string(), "b".to_string()];
        let f = parse_nlmer_custom_formula("y ~ x ~ a | g", &names).unwrap();
        assert_eq!(f.response, "y");
        assert_eq!(f.covariate, "x");
        assert_eq!(f.fixed_param_names, names);
        assert_eq!(f.re_params, vec!["a".to_string()]);
        assert_eq!(f.re_group, "g");
    }
}
