//! Parse `nlmer`-style three-part formulas.

use crate::LmeError;

/// Parsed nonlinear mixed-model formula.
///
/// Example: `circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree`
#[derive(Debug, Clone)]
pub struct NlmerFormula {
    /// Response column name.
    pub response: String,
    /// Covariate passed to the mean function (e.g. `age`).
    pub covariate: String,
    /// Names of the three fixed nonlinear parameters (`Asym`, `xmid`, `scal` for `SSlogis`).
    pub fixed_param_names: [String; 3],
    /// Nonlinear parameter that carries the random effect (e.g. `Asym`).
    pub re_param: String,
    /// Grouping factor column (e.g. `Tree`).
    pub re_group: String,
}

/// Supported nonlinear mean functions in the middle part of the formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlmmMeanKind {
    /// `SSlogis(covariate, Asym, xmid, scal)`
    Sslogis,
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
    let (re_param, re_group) = parse_random_part(random)?;

    Ok((
        NlmerFormula {
            response,
            covariate,
            fixed_param_names: param_names,
            re_param,
            re_group,
        },
        mean_kind,
    ))
}

fn parse_nonlinear_part(s: &str) -> crate::Result<(NlmmMeanKind, String, [String; 3])> {
    let open = s.find('(').ok_or_else(|| LmeError::NotImplemented {
        feature: format!("Nonlinear part must be a function call, got '{s}'"),
    })?;
    let fname = s[..open].trim();
    let close = s.rfind(')').ok_or_else(|| LmeError::NotImplemented {
        feature: format!("Missing ')' in nonlinear part '{s}'"),
    })?;
    let inner = &s[open + 1..close];
    let args: Vec<&str> = inner.split(',').map(str::trim).collect();
    if args.len() != 4 {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "SSlogis requires four arguments (covariate, Asym, xmid, scal), got {}",
                args.len()
            ),
        });
    }
    let mean = match fname {
        "SSlogis" => NlmmMeanKind::Sslogis,
        other => {
            return Err(LmeError::NotImplemented {
                feature: format!(
                    "Unsupported nonlinear mean '{other}' (only SSlogis is implemented)"
                ),
            });
        }
    };
    Ok((
        mean,
        args[0].to_string(),
        [
            args[1].to_string(),
            args[2].to_string(),
            args[3].to_string(),
        ],
    ))
}

fn parse_random_part(s: &str) -> crate::Result<(String, String)> {
    let parts: Vec<&str> = s.split('|').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(LmeError::NotImplemented {
            feature: format!(
                "Random part must be 'param|group' (one random effect on one parameter), got '{s}'"
            ),
        });
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
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
    }
}
