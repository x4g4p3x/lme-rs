//! Panic-free Wilkinson formula parsing tailored to lme-rs.
//!
//! The parser intentionally produces the compact, variable-centric model that the
//! design-matrix builder consumes. It supports the formula surface exercised by
//! lme-rs: ordered fixed effects, two-way interactions, random intercepts and
//! slopes, crossed and nested grouping factors, `||`, and one `offset(...)` term.

use ahash::AHashMap;
use smallvec::SmallVec;

/// Root metadata structure holding a parsed Wilkinson formula.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormulaModel {
    /// Generated columns in formula order. `intercept` is metadata-only and has no [`ColumnInfo`].
    pub all_generated_columns: Vec<String>,
    /// Mapping of source/generated columns to their model roles and random effects.
    pub columns: AHashMap<String, ColumnInfo>,
    /// Global model properties.
    pub metadata: FormulaMetadata,
    /// The original, unmodified formula.
    pub formula: String,
    /// Optional offset expression. Simple column names are materialized by the matrix builder.
    pub offset: Option<String>,
}

/// Roles and random-effect mappings for one dataframe or generated column.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColumnInfo {
    /// Random-effect definitions for a grouping column.
    pub random_effects: Vec<RandomEffect>,
    /// Compact typed roles used by the design-matrix builder.
    pub roles: ColumnRoles,
    generated: bool,
}

impl ColumnInfo {
    /// Return whether this column has the requested model role.
    #[inline]
    pub fn has_role(&self, role: ColumnRole) -> bool {
        self.roles.contains(role)
    }
}

/// A column's role in the parsed formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnRole {
    /// The left-hand-side response column.
    Response,
    /// A main fixed-effect column.
    FixedEffect,
    /// A generated fixed-effect interaction term.
    Interaction,
    /// A source column used as a random slope.
    RandomEffect,
    /// A random-effect grouping factor.
    GroupingVariable,
}

/// Compact set of [`ColumnRole`] flags.
///
/// Formula columns usually carry one or two roles, so a bitset avoids a heap
/// allocation per column while retaining a typed public representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColumnRoles(u8);

impl ColumnRoles {
    /// Return whether this set contains `role`.
    #[inline]
    pub const fn contains(self, role: ColumnRole) -> bool {
        self.0 & (1 << role as u8) != 0
    }

    /// Insert `role` into this set.
    #[inline]
    fn insert(&mut self, role: ColumnRole) {
        self.0 |= 1 << role as u8;
    }
}

/// One `(expression | group)` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomEffect {
    /// Whether this declaration requests a correlated covariance block (`|`, not `||`).
    pub correlated: bool,
    /// Whether the declaration includes a random intercept.
    pub has_intercept: bool,
    /// Random-slope source columns. Empty for intercept-only declarations.
    pub variables: SmallVec<[String; 2]>,
}

/// Top-level characteristics of the parsed formula.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormulaMetadata {
    /// Whether the fixed-effects design includes an intercept.
    pub has_intercept: bool,
    /// Whether at least one random-effect declaration is present.
    pub is_random_effects_model: bool,
    /// Number of response variables. lme-rs currently supports one.
    pub response_variable_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token<'a> {
    Ident(&'a str),
    Number(usize),
    Tilde,
    Plus,
    Minus,
    Star,
    Colon,
    Slash,
    Pipe,
    DoublePipe,
    LParen,
    RParen,
    Comma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Add,
    Remove,
}

type Tokens<'source> = SmallVec<[Token<'source>; 16]>;
type RandomSlopes<'source> = SmallVec<[&'source str; 4]>;
type GroupNames = SmallVec<[String; 4]>;

struct Lexed<'source> {
    tokens: Tokens<'source>,
    tilde: Option<usize>,
    identifier_count: usize,
}

/// Parse a Wilkinson formula into lme-rs's typed formula model.
///
/// This function is total over UTF-8 input: malformed or unsupported syntax is
/// returned as a formula error and never handled with a parser panic.
pub fn parse(formula: &str) -> crate::Result<FormulaModel> {
    let Lexed {
        tokens,
        tilde,
        identifier_count,
    } = lex(formula)?;
    let tilde = tilde.ok_or_else(|| parse_error("formula must contain exactly one '~'"))?;
    let response = match &tokens[..tilde] {
        [Token::Ident(name)] => *name,
        [] => return Err(parse_error("formula is missing a response variable")),
        _ => {
            return Err(parse_error(
                "response must be one unquoted ASCII column name",
            ))
        }
    };

    let mut model = FormulaModel {
        all_generated_columns: Vec::with_capacity(identifier_count + 1),
        columns: AHashMap::with_capacity(identifier_count),
        metadata: FormulaMetadata {
            has_intercept: true,
            is_random_effects_model: false,
            response_variable_count: 1,
        },
        formula: formula.to_string(),
        offset: None,
    };
    add_role(&mut model, response, ColumnRole::Response, true);

    visit_additive_terms(&tokens[tilde + 1..], |sign, term| {
        if is_random_effect_term(term) {
            if sign == Sign::Remove {
                return Err(parse_error(
                    "subtracting random-effect terms is not supported",
                ));
            }
            parse_random_effect(term, &mut model)?;
        } else {
            parse_fixed_term(term, sign, &mut model)?;
        }
        Ok(())
    })?;

    if model.metadata.has_intercept {
        model.all_generated_columns.insert(
            1.min(model.all_generated_columns.len()),
            "intercept".to_string(),
        );
    }
    Ok(model)
}

fn lex(formula: &str) -> crate::Result<Lexed<'_>> {
    let bytes = formula.as_bytes();
    let mut tokens = if formula.len() > 128 {
        // Large generated formulas will exceed the inline token buffer. Reserve
        // directly so SmallVec does not first fill and copy its stack storage.
        Tokens::with_capacity(formula.len().div_ceil(2))
    } else {
        Tokens::new()
    };
    let mut tilde = None;
    let mut identifier_count = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token::Ident(&formula[start..i]));
            identifier_count += 1;
            continue;
        }
        if byte.is_ascii_digit() {
            let mut value = usize::from(byte - b'0');
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                value = value
                    .checked_mul(10)
                    .and_then(|current| current.checked_add(usize::from(bytes[i] - b'0')))
                    .ok_or_else(|| parse_error("numeric literal is too large"))?;
                i += 1;
            }
            tokens.push(Token::Number(value));
            continue;
        }
        let token = match byte {
            b'~' => {
                if tilde.replace(tokens.len()).is_some() {
                    return Err(parse_error("formula must contain exactly one '~'"));
                }
                Token::Tilde
            }
            b'+' => Token::Plus,
            b'-' => Token::Minus,
            b'*' => Token::Star,
            b':' => Token::Colon,
            b'/' => Token::Slash,
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b',' => Token::Comma,
            b'|' if bytes.get(i + 1) == Some(&b'|') => {
                i += 1;
                Token::DoublePipe
            }
            b'|' => Token::Pipe,
            _ if byte.is_ascii() => {
                return Err(parse_error(format!(
                    "unsupported character '{}' at byte {}",
                    byte as char, i
                )))
            }
            _ => return Err(parse_error(format!("non-ASCII identifier at byte {}", i))),
        };
        tokens.push(token);
        i += 1;
    }
    if tokens.is_empty() {
        return Err(parse_error("formula is empty"));
    }
    Ok(Lexed {
        tokens,
        tilde,
        identifier_count,
    })
}

fn visit_additive_terms<'source>(
    tokens: &[Token<'source>],
    mut visit: impl FnMut(Sign, &[Token<'source>]) -> crate::Result<()>,
) -> crate::Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut sign = Sign::Add;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            Token::LParen => depth += 1,
            Token::RParen => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| parse_error("unmatched ')'"))?;
            }
            Token::Plus | Token::Minus if depth == 0 => {
                if i == start {
                    if i == 0 && matches!(token, Token::Minus) {
                        sign = Sign::Remove;
                        start = 1;
                        continue;
                    }
                    return Err(parse_error("empty term between additive operators"));
                }
                visit(sign, &tokens[start..i])?;
                sign = if matches!(token, Token::Plus) {
                    Sign::Add
                } else {
                    Sign::Remove
                };
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(parse_error("unclosed '('"));
    }
    if start == tokens.len() {
        return Err(parse_error("formula cannot end with an additive operator"));
    }
    visit(sign, &tokens[start..])
}

fn is_random_effect_term(tokens: &[Token<'_>]) -> bool {
    if !matches!(tokens.first(), Some(Token::LParen))
        || !matches!(tokens.last(), Some(Token::RParen))
    {
        return false;
    }
    let mut depth = 0usize;
    for token in &tokens[1..tokens.len() - 1] {
        match token {
            Token::LParen => depth += 1,
            Token::RParen => depth = depth.saturating_sub(1),
            Token::Pipe | Token::DoublePipe if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn parse_fixed_term(
    tokens: &[Token<'_>],
    sign: Sign,
    model: &mut FormulaModel,
) -> crate::Result<()> {
    match tokens {
        [Token::Number(0)] => {
            model.metadata.has_intercept = false;
            return Ok(());
        }
        [Token::Number(1)] => {
            model.metadata.has_intercept = sign == Sign::Add;
            return Ok(());
        }
        [Token::Number(_)] => return Err(parse_error("only 0 and 1 are valid standalone terms")),
        [Token::Ident(name)] if sign == Sign::Add => {
            add_role(model, name, ColumnRole::FixedEffect, true);
            return Ok(());
        }
        [Token::Ident(_)] => {
            return Err(parse_error(
                "subtracting named fixed-effect terms is not supported",
            ))
        }
        _ => {}
    }

    if let Some((name, args)) = function_call(tokens) {
        if sign == Sign::Remove {
            return Err(parse_error("subtracting function terms is not supported"));
        }
        return parse_function(name, args, model);
    }

    parse_interaction(tokens, sign, model)
}

fn function_call<'tokens, 'source>(
    tokens: &'tokens [Token<'source>],
) -> Option<(&'source str, &'tokens [Token<'source>])> {
    match tokens {
        [Token::Ident(name), Token::LParen, middle @ .., Token::RParen]
            if outer_parentheses_are_balanced(middle) =>
        {
            Some((name, middle))
        }
        _ => None,
    }
}

fn outer_parentheses_are_balanced(tokens: &[Token<'_>]) -> bool {
    let mut depth = 0usize;
    for token in tokens {
        match token {
            Token::LParen => depth += 1,
            Token::RParen => {
                let Some(next) = depth.checked_sub(1) else {
                    return false;
                };
                depth = next;
            }
            _ => {}
        }
    }
    depth == 0
}

fn parse_function(name: &str, args: &[Token<'_>], model: &mut FormulaModel) -> crate::Result<()> {
    match (name, args) {
        ("offset", []) => Err(parse_error("offset() requires an expression")),
        ("offset", [Token::Ident(source)]) => {
            if model.offset.is_some() {
                return Err(parse_error("only one offset() term is supported"));
            }
            model.offset = Some((*source).to_string());
            Ok(())
        }
        ("offset", _) => Err(parse_error(
            "offset() currently requires one plain column name",
        )),
        _ => Err(parse_error(format!(
            "unsupported formula function '{}()'",
            name
        ))),
    }
}

fn parse_interaction(
    tokens: &[Token<'_>],
    sign: Sign,
    model: &mut FormulaModel,
) -> crate::Result<()> {
    if sign == Sign::Remove {
        return Err(parse_error(
            "subtracting interaction terms is not supported",
        ));
    }
    match tokens {
        [Token::Ident(left), Token::Star, Token::Ident(right)] => {
            add_role(model, left, ColumnRole::FixedEffect, true);
            add_role(model, right, ColumnRole::FixedEffect, true);
            add_role(
                model,
                &format!("{}_{}", left, right),
                ColumnRole::Interaction,
                true,
            );
            Ok(())
        }
        _ => Err(parse_error("malformed or unsupported fixed-effect term")),
    }
}

fn parse_random_effect(tokens: &[Token<'_>], model: &mut FormulaModel) -> crate::Result<()> {
    let inner = &tokens[1..tokens.len() - 1];
    let mut depth = 0usize;
    let mut separator = None;
    for (i, token) in inner.iter().enumerate() {
        match token {
            Token::LParen => depth += 1,
            Token::RParen => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| parse_error("unmatched ')' in random effect"))?;
            }
            Token::Pipe | Token::DoublePipe if depth == 0 => {
                if separator.is_some() {
                    return Err(parse_error("random effect contains more than one '|'"));
                }
                separator = Some((i, matches!(token, Token::Pipe)));
            }
            _ => {}
        }
    }
    let (bar, correlated) = separator.ok_or_else(|| parse_error("random effect is missing '|'"))?;
    let lhs = &inner[..bar];
    let rhs = &inner[bar + 1..];
    if lhs.is_empty() || rhs.is_empty() {
        return Err(parse_error(
            "random effect requires an expression and grouping factor",
        ));
    }

    let (has_intercept, slopes) = parse_random_lhs(lhs)?;
    if !has_intercept && slopes.is_empty() {
        return Err(parse_error("random effect cannot have zero columns"));
    }
    let groups = parse_grouping_expression(rhs)?;
    for group in groups {
        for slope in &slopes {
            add_role(model, slope, ColumnRole::RandomEffect, false);
        }
        let effect = RandomEffect {
            correlated,
            has_intercept,
            variables: slopes.iter().map(|slope| (*slope).to_string()).collect(),
        };
        add_role(model, &group, ColumnRole::GroupingVariable, true);
        model
            .columns
            .get_mut(&group)
            .expect("grouping column was inserted")
            .random_effects
            .push(effect);
    }
    model.metadata.is_random_effects_model = true;
    Ok(())
}

fn parse_random_lhs<'source>(
    tokens: &[Token<'source>],
) -> crate::Result<(bool, RandomSlopes<'source>)> {
    let mut has_intercept = true;
    let mut slopes = RandomSlopes::new();
    visit_additive_terms(tokens, |sign, term| {
        match term {
            [Token::Number(0)] => has_intercept = false,
            [Token::Number(1)] => has_intercept = sign == Sign::Add,
            [Token::Ident(name)] if sign == Sign::Add => {
                if !slopes.contains(name) {
                    slopes.push(*name);
                }
            }
            [Token::Ident(_)] => {
                return Err(parse_error("subtracting random slopes is not supported"))
            }
            _ => return Err(parse_error("malformed random-effect expression")),
        }
        Ok(())
    })?;
    Ok((has_intercept, slopes))
}

fn parse_grouping_expression(tokens: &[Token<'_>]) -> crate::Result<GroupNames> {
    if tokens.is_empty() {
        return Err(parse_error("empty grouping expression"));
    }
    let mut has_colon = false;
    let mut has_slash = false;
    let mut total_name_len = 0usize;
    for (i, token) in tokens.iter().enumerate() {
        if i % 2 == 0 {
            match token {
                Token::Ident(name) => total_name_len += name.len(),
                _ => return Err(parse_error("grouping factors must be ASCII column names")),
            }
        } else {
            match token {
                Token::Colon => has_colon = true,
                Token::Slash => has_slash = true,
                _ => return Err(parse_error("grouping factors use ':' or '/' separators")),
            }
        }
    }
    if tokens.len().is_multiple_of(2) {
        return Err(parse_error("grouping expression ends with a separator"));
    }
    if has_colon && has_slash {
        return Err(parse_error(
            "mixing ':' and '/' in one grouping expression is unsupported",
        ));
    }
    let name_count = tokens.len().div_ceil(2);
    if has_slash {
        let mut groups = GroupNames::with_capacity(name_count);
        let mut accumulated = String::with_capacity(total_name_len + name_count - 1);
        for token in tokens.iter().step_by(2) {
            let Token::Ident(name) = token else {
                unreachable!("grouping token shape was validated")
            };
            if !accumulated.is_empty() {
                accumulated.push(':');
            }
            accumulated.push_str(name);
            groups.push(accumulated.clone());
        }
        Ok(groups)
    } else {
        let mut group = String::with_capacity(total_name_len + name_count - 1);
        for token in tokens.iter().step_by(2) {
            let Token::Ident(name) = token else {
                unreachable!("grouping token shape was validated")
            };
            if !group.is_empty() {
                group.push(':');
            }
            group.push_str(name);
        }
        let mut groups = GroupNames::new();
        groups.push(group);
        Ok(groups)
    }
}

fn add_role(model: &mut FormulaModel, name: &str, role: ColumnRole, generated: bool) {
    let info = model.columns.entry(name.to_string()).or_default();
    info.roles.insert(role);
    if generated && !info.generated {
        info.generated = true;
        model.all_generated_columns.push(name.to_string());
    }
}

fn parse_error(message: impl Into<String>) -> crate::LmeError {
    crate::LmeError::NotImplemented {
        feature: format!("Formula parsing error: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_intercept_and_correlated_slope() {
        let ast = parse("Reaction ~ Days + (1 + Days | Subject)").unwrap();
        assert!(ast.metadata.has_intercept);
        assert!(ast.metadata.is_random_effects_model);
        assert!(ast.columns["Days"].has_role(ColumnRole::FixedEffect));
        let effect = &ast.columns["Subject"].random_effects[0];
        assert!(effect.has_intercept);
        assert!(effect.correlated);
        assert_eq!(effect.variables.as_slice(), &["Days".to_string()]);
    }

    #[test]
    fn parses_no_intercept_fixed_and_random_terms() {
        let ast = parse("y ~ 0 + x + (0 + x | g)").unwrap();
        assert!(!ast.metadata.has_intercept);
        assert!(!ast.columns["g"].random_effects[0].has_intercept);
    }

    #[test]
    fn expands_nested_grouping_to_every_level() {
        let ast = parse("y ~ x + (1 | district/school/student)").unwrap();
        for group in ["district", "district:school", "district:school:student"] {
            assert!(ast.columns[group].has_role(ColumnRole::GroupingVariable));
        }
    }

    #[test]
    fn parses_crossed_groups_and_offset() {
        let ast = parse("y ~ x + offset(log_exposure) + (1 | A) + (1 | B)").unwrap();
        assert_eq!(ast.offset.as_deref(), Some("log_exposure"));
        assert!(ast.columns.contains_key("A"));
        assert!(ast.columns.contains_key("B"));
    }

    #[test]
    fn malformed_inputs_are_errors_not_panics() {
        for formula in ["", "y", "~ y", "y ~ x + * z", "y ~ (1 |", "Rey ~ 1 (+ (1 |"] {
            assert!(parse(formula).is_err(), "{formula:?} should be rejected");
        }
    }
}
