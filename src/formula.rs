use fiasto::parse_formula;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct FiastoModel {
    pub all_generated_columns: Vec<String>,
    pub columns: HashMap<String, ColumnInfo>,
    pub metadata: FiastoMetadata,
}

#[derive(Debug, Deserialize)]
pub struct ColumnInfo {
    #[serde(default)]
    pub random_effects: Vec<RandomEffect>,
    #[serde(default)]
    pub roles: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RandomEffect {
    pub correlated: bool,
    pub grouping_variable: String,
    pub has_intercept: bool,
    pub kind: String,
}

#[derive(Debug, Deserialize)]
pub struct FiastoMetadata {
    pub has_intercept: bool,
    pub is_random_effects_model: bool,
    pub response_variable_count: usize,
}

/// Parses a Wilkinson's formula string into a structured AST metadata model
pub fn parse(formula: &str) -> crate::Result<FiastoModel> {
    let json_val = parse_formula(formula)
        .map_err(|e| crate::LmeError::NotImplemented { feature: format!("Formula parsing error: {}", e) })?;
        
    let ast: FiastoModel = serde_json::from_value(json_val)
        .map_err(|e| crate::LmeError::NotImplemented { feature: format!("JSON parsing error: {}", e) })?;
        
    Ok(ast)
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
}
