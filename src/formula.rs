use fiasto::parse_formula;
use serde::Deserialize;
use std::collections::HashMap;

/// Root metadata structure holding the parsed Wilkinson AST.
#[derive(Debug, Deserialize)]
pub struct FiastoModel {
    pub all_generated_columns: Vec<String>,
    pub columns: HashMap<String, ColumnInfo>,
    pub metadata: FiastoMetadata,
    pub formula: String,
}

/// Captures the roles and random effect mappings for a specific parsed dataframe column.
#[derive(Debug, Deserialize)]
pub struct ColumnInfo {
    #[serde(default)]
    pub random_effects: Vec<RandomEffect>,
    #[serde(default)]
    pub roles: Vec<String>,
}

/// Represents a distinct Random Effect cluster mapped from `(expr | group)`.
#[derive(Debug, Deserialize)]
pub struct RandomEffect {
    pub correlated: bool,
    pub grouping_variable: String,
    pub has_intercept: bool,
    pub kind: String,
    #[serde(default)]
    pub variables: Option<Vec<String>>,
}

/// Stores top-level characteristics of the model structure.
#[derive(Debug, Deserialize)]
pub struct FiastoMetadata {
    pub has_intercept: bool,
    pub is_random_effects_model: bool,
    pub response_variable_count: usize,
}

/// Parses a Wilkinson's formula string into a structured AST metadata model.
///
/// Supports nested random effects: `(1|a/b)` is expanded to `(1|a) + (1|a:b)` before parsing.
pub fn parse(formula: &str) -> crate::Result<FiastoModel> {
    let expanded = expand_nested_re(formula);
    let json_val = parse_formula(&expanded)
        .map_err(|e| crate::LmeError::NotImplemented { feature: format!("Formula parsing error: {}", e) })?;
        
    let mut ast: FiastoModel = serde_json::from_value(json_val)
        .map_err(|e| crate::LmeError::NotImplemented { feature: format!("JSON parsing error: {}", e) })?;
    
    // Store the original formula (unexpanded) for display
    ast.formula = formula.to_string();
    
    Ok(ast)
}

/// Expand nested random effects: `(expr | a/b)` → `(expr | a) + (expr | a:b)`.
///
/// Also supports deeper nesting: `(1|a/b/c)` → `(1|a) + (1|a:b) + (1|a:b:c)`.
fn expand_nested_re(formula: &str) -> String {
    let mut result = formula.to_string();
    
    // Keep expanding until no more nested patterns remain
    loop {
        let mut expanded = String::new();
        let mut i = 0;
        let bytes = result.as_bytes();
        let mut changed = false;
        
        while i < bytes.len() {
            if bytes[i] == b'(' {
                // Find matching closing paren
                let start = i;
                let mut depth = 1;
                let mut j = i + 1;
                while j < bytes.len() && depth > 0 {
                    if bytes[j] == b'(' { depth += 1; }
                    if bytes[j] == b')' { depth -= 1; }
                    j += 1;
                }
                let re_term = &result[start..j]; // includes parens
                let inner = &result[start + 1..j - 1]; // without parens
                
                // Check if inner contains `|` and the grouping side contains `/`
                if let Some(bar_pos) = inner.find('|') {
                    let expr = inner[..bar_pos].trim();
                    let group = inner[bar_pos + 1..].trim();
                    
                    if group.contains('/') {
                        // Expand: (expr | a/b) → (expr | a) + (expr | a:b)
                        // Expand: (expr | a/b/c) → (expr | a) + (expr | a:b) + (expr | a:b:c)
                        let parts: Vec<&str> = group.split('/').map(|s| s.trim()).collect();
                        let mut terms = Vec::new();
                        let mut acc = String::new();
                        for (idx, part) in parts.iter().enumerate() {
                            if idx == 0 {
                                acc = part.to_string();
                            } else {
                                acc = format!("{}:{}", acc, part);
                            }
                            terms.push(format!("({} | {})", expr, acc));
                        }
                        expanded.push_str(&terms.join(" + "));
                        changed = true;
                        i = j;
                        continue;
                    }
                }
                
                expanded.push_str(re_term);
                i = j;
            } else {
                expanded.push(result.as_bytes()[i] as char);
                i += 1;
            }
        }
        
        if !changed {
            break;
        }
        result = expanded;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_intercept_only() {
        let ast = parse("Reaction ~ 1 + (1 | Subject)").unwrap();
        assert!(ast.metadata.is_random_effects_model);
        assert!(ast.metadata.has_intercept);
        
        let subject_col = ast.columns.get("Subject").unwrap();
        assert_eq!(subject_col.roles[0], "GroupingVariable");
        assert_eq!(subject_col.random_effects[0].grouping_variable, "Subject");
        assert_eq!(subject_col.random_effects[0].kind, "grouping");
    }

    #[test]
    fn test_parse_random_slope() {
        let ast = parse("Reaction ~ Days + (Days | Subject)").unwrap();
        
        let days_col = ast.columns.get("Days").unwrap();
        // Should be both a fixed effect (Identity) and random slope
        assert!(days_col.roles.contains(&"Identity".to_string()));
        assert!(days_col.roles.contains(&"RandomEffect".to_string()));
        
        let re = &days_col.random_effects[0];
        assert_eq!(re.kind, "slope");
        assert_eq!(re.grouping_variable, "Subject");
        assert!(re.correlated);
    }

    #[test]
    fn test_expand_nested_two_level() {
        let expanded = expand_nested_re("y ~ x + (1 | school/student)");
        assert_eq!(expanded, "y ~ x + (1 | school) + (1 | school:student)");
    }

    #[test]
    fn test_expand_nested_three_level() {
        let expanded = expand_nested_re("y ~ x + (1 | district/school/student)");
        assert_eq!(expanded, "y ~ x + (1 | district) + (1 | district:school) + (1 | district:school:student)");
    }

    #[test]
    fn test_expand_no_nesting() {
        let original = "y ~ x + (1 | school)";
        let expanded = expand_nested_re(original);
        assert_eq!(expanded, original);
    }

    #[test]
    fn test_expand_nested_with_slopes() {
        let expanded = expand_nested_re("y ~ x + (x | school/student)");
        assert_eq!(expanded, "y ~ x + (x | school) + (x | school:student)");
    }
}
