//! Panic-free Wilkinson formula parsing tailored to lme-rs.
//!
//! The parser intentionally produces the compact, variable-centric model that the
//! design-matrix builder consumes. It supports the formula surface exercised by
//! lme-rs: ordered fixed effects, two-way `*` and colon `:` interactions, unary
//! column transforms (`log`, `sqrt`, `exp`), `I()` arithmetic, random intercepts
//! and slopes, crossed and nested grouping factors, `||`, and one `offset(...)`
//! term (plain column or unary transform).

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
    /// Optional offset expression. A plain column or a unary transform such as `log(w)`.
    pub offset: Option<NumericExpr>,
}

/// Roles and random-effect mappings for one dataframe or generated column.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColumnInfo {
    /// Random-effect definitions for a grouping column.
    pub random_effects: Vec<RandomEffect>,
    /// Compact typed roles used by the design-matrix builder.
    pub roles: ColumnRoles,
    generated: bool,
    /// Evaluated expression when this generated column is not a raw DataFrame column.
    pub expr: Option<NumericExpr>,
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

/// A numeric expression materialized from a formula term (`log(x)`, `I(x^2)`, …).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericExpr {
    /// A DataFrame column.
    Column(String),
    /// An integer literal from `I()` arithmetic.
    Literal(i64),
    /// `log`, `sqrt`, `exp`, or unary minus.
    Unary {
        /// Transform applied to [`arg`](Self::Unary::arg).
        op: UnaryOp,
        /// Inner expression.
        arg: Box<NumericExpr>,
    },
    /// `+`, `-`, `*`, `/`, or `^` inside `I()`.
    Binary {
        /// Operator.
        op: BinaryOp,
        /// Left operand.
        left: Box<NumericExpr>,
        /// Right operand.
        right: Box<NumericExpr>,
    },
}

/// Unary numeric transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Exponential.
    Exp,
    /// Arithmetic negation.
    Neg,
}

/// Binary arithmetic operator inside `I()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Exponentiation.
    Pow,
}

impl UnaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Neg => "-",
        }
    }
}

impl NumericExpr {
    /// Plain column name when this expression is a single DataFrame column.
    #[must_use]
    pub fn as_column_name(&self) -> Option<&str> {
        match self {
            Self::Column(name) => Some(name.as_str()),
            _ => None,
        }
    }

    /// Printable formula label (`log(x)`, `I(x^2)`, …).
    #[must_use]
    pub fn label(&self) -> String {
        self.fmt_prec(0)
    }

    fn fmt_prec(&self, parent_prec: u8) -> String {
        match self {
            Self::Column(name) => name.clone(),
            Self::Literal(value) => value.to_string(),
            Self::Unary {
                op: UnaryOp::Neg,
                arg,
            } => {
                let inner = arg.fmt_prec(3);
                wrap_prec(format!("-{inner}"), 3, parent_prec)
            }
            Self::Unary { op, arg } => format!("{}({})", op.name(), arg.fmt_prec(0)),
            Self::Binary { op, left, right } => {
                let prec = binary_prec(*op);
                let text = format!(
                    "{}{}{}",
                    left.fmt_prec(prec),
                    binary_symbol(*op),
                    right.fmt_prec(if *op == BinaryOp::Pow { prec } else { prec + 1 })
                );
                wrap_prec(text, prec, parent_prec)
            }
        }
    }

    /// Visit every DataFrame column referenced by this expression.
    pub fn for_each_column(&self, visit: &mut impl FnMut(&str)) {
        match self {
            Self::Column(name) => visit(name),
            Self::Literal(_) => {}
            Self::Unary { arg, .. } => arg.for_each_column(visit),
            Self::Binary { left, right, .. } => {
                left.for_each_column(visit);
                right.for_each_column(visit);
            }
        }
    }
}

impl FormulaModel {
    /// Offset column name when the offset is a plain DataFrame column.
    #[must_use]
    pub fn offset_column(&self) -> Option<&str> {
        self.offset.as_ref().and_then(NumericExpr::as_column_name)
    }
}

fn binary_prec(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add | BinaryOp::Sub => 1,
        BinaryOp::Mul | BinaryOp::Div => 2,
        BinaryOp::Pow => 4,
    }
}

fn binary_symbol(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => " + ",
        BinaryOp::Sub => " - ",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Pow => "^",
    }
}

fn wrap_prec(text: String, prec: u8, parent_prec: u8) -> String {
    if prec < parent_prec {
        format!("({text})")
    } else {
        text
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Token<'a> {
    Ident(&'a str),
    Number(usize),
    Tilde,
    Plus,
    Minus,
    Star,
    Caret,
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
            b'^' => Token::Caret,
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
    match name {
        "offset" => parse_offset(args, model),
        "log" => add_unary_transform(UnaryOp::Log, args, model),
        "sqrt" => add_unary_transform(UnaryOp::Sqrt, args, model),
        "exp" => add_unary_transform(UnaryOp::Exp, args, model),
        "I" => add_identity_term(args, model),
        _ => Err(parse_error(format!(
            "unsupported formula function '{}()'",
            name
        ))),
    }
}

fn parse_offset(args: &[Token<'_>], model: &mut FormulaModel) -> crate::Result<()> {
    if model.offset.is_some() {
        return Err(parse_error("only one offset() term is supported"));
    }
    let expr = match args {
        [] => return Err(parse_error("offset() requires an expression")),
        [Token::Ident(source)] => NumericExpr::Column((*source).to_string()),
        _ => match function_call(args) {
            Some((func, inner)) => unary_column_transform(func, inner)?,
            None => {
                return Err(parse_error(
                    "offset() requires a column name or a unary transform of one column",
                ))
            }
        },
    };
    expr.for_each_column(&mut |column| {
        ensure_column(model, column);
    });
    model.offset = Some(expr);
    Ok(())
}

fn add_unary_transform(
    op: UnaryOp,
    args: &[Token<'_>],
    model: &mut FormulaModel,
) -> crate::Result<()> {
    let expr = unary_column_transform(op.name(), args)?;
    add_generated_expr(model, expr, ColumnRole::FixedEffect)
}

fn unary_column_transform(name: &str, args: &[Token<'_>]) -> crate::Result<NumericExpr> {
    let op = match name {
        "log" => UnaryOp::Log,
        "sqrt" => UnaryOp::Sqrt,
        "exp" => UnaryOp::Exp,
        _ => {
            return Err(parse_error(format!(
                "offset() does not support '{name}()' transforms"
            )))
        }
    };
    match args {
        [Token::Ident(source)] => Ok(NumericExpr::Unary {
            op,
            arg: Box::new(NumericExpr::Column((*source).to_string())),
        }),
        _ => Err(parse_error(format!(
            "{name}() currently requires one plain column name"
        ))),
    }
}

fn add_identity_term(args: &[Token<'_>], model: &mut FormulaModel) -> crate::Result<()> {
    if args.is_empty() {
        return Err(parse_error("I() requires an arithmetic expression"));
    }
    let inner = parse_numeric_expr(args)?;
    if matches!(&inner, NumericExpr::Column(_)) {
        return Err(parse_error(
            "I() requires arithmetic; use the column name directly",
        ));
    }
    add_i_expr(model, inner)
}

fn add_i_expr(model: &mut FormulaModel, inner: NumericExpr) -> crate::Result<()> {
    inner.for_each_column(&mut |column| {
        ensure_column(model, column);
    });
    let name = format!("I({})", inner.label());
    add_role(model, &name, ColumnRole::FixedEffect, true);
    model
        .columns
        .get_mut(&name)
        .expect("generated column was inserted")
        .expr = Some(inner);
    Ok(())
}

fn add_generated_expr(
    model: &mut FormulaModel,
    expr: NumericExpr,
    role: ColumnRole,
) -> crate::Result<()> {
    expr.for_each_column(&mut |column| {
        ensure_column(model, column);
    });
    let name = expr.label();
    add_role(model, &name, role, true);
    model
        .columns
        .get_mut(&name)
        .expect("generated column was inserted")
        .expr = Some(expr);
    Ok(())
}

fn parse_numeric_expr(tokens: &[Token<'_>]) -> crate::Result<NumericExpr> {
    let (expr, end) = parse_sum(tokens, 0)?;
    if end != tokens.len() {
        return Err(parse_error("malformed I() arithmetic expression"));
    }
    Ok(expr)
}

fn parse_sum(tokens: &[Token<'_>], start: usize) -> crate::Result<(NumericExpr, usize)> {
    let (mut expr, mut i) = parse_product(tokens, start)?;
    while i < tokens.len() {
        let op = match tokens[i] {
            Token::Plus => BinaryOp::Add,
            Token::Minus => BinaryOp::Sub,
            _ => break,
        };
        let (right, next) = parse_product(tokens, i + 1)?;
        expr = NumericExpr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
        };
        i = next;
    }
    Ok((expr, i))
}

fn parse_product(tokens: &[Token<'_>], start: usize) -> crate::Result<(NumericExpr, usize)> {
    let (mut expr, mut i) = parse_power(tokens, start)?;
    while i < tokens.len() {
        let op = match tokens[i] {
            Token::Star => BinaryOp::Mul,
            Token::Slash => BinaryOp::Div,
            _ => break,
        };
        let (right, next) = parse_power(tokens, i + 1)?;
        expr = NumericExpr::Binary {
            op,
            left: Box::new(expr),
            right: Box::new(right),
        };
        i = next;
    }
    Ok((expr, i))
}

fn parse_power(tokens: &[Token<'_>], start: usize) -> crate::Result<(NumericExpr, usize)> {
    let (left, i) = parse_unary(tokens, start)?;
    if i < tokens.len() && matches!(tokens[i], Token::Caret) {
        let (right, next) = parse_power(tokens, i + 1)?;
        return Ok((
            NumericExpr::Binary {
                op: BinaryOp::Pow,
                left: Box::new(left),
                right: Box::new(right),
            },
            next,
        ));
    }
    Ok((left, i))
}

fn parse_unary(tokens: &[Token<'_>], start: usize) -> crate::Result<(NumericExpr, usize)> {
    if start >= tokens.len() {
        return Err(parse_error("I() arithmetic expression is incomplete"));
    }
    if matches!(tokens[start], Token::Minus) {
        let (arg, next) = parse_unary(tokens, start + 1)?;
        return Ok((
            NumericExpr::Unary {
                op: UnaryOp::Neg,
                arg: Box::new(arg),
            },
            next,
        ));
    }
    parse_primary(tokens, start)
}

fn parse_primary(tokens: &[Token<'_>], start: usize) -> crate::Result<(NumericExpr, usize)> {
    match tokens.get(start) {
        Some(Token::Ident(name)) => {
            if tokens.get(start + 1) == Some(&Token::LParen) {
                return Err(parse_error(
                    "I() does not support nested function calls; use log()/sqrt()/exp() terms",
                ));
            }
            Ok((NumericExpr::Column((*name).to_string()), start + 1))
        }
        Some(Token::Number(value)) => {
            let literal = i64::try_from(*value)
                .map_err(|_| parse_error("numeric literal in I() is too large"))?;
            Ok((NumericExpr::Literal(literal), start + 1))
        }
        Some(Token::LParen) => {
            let (expr, next) = parse_sum(tokens, start + 1)?;
            if tokens.get(next) != Some(&Token::RParen) {
                return Err(parse_error("unclosed '(' in I()"));
            }
            Ok((expr, next + 1))
        }
        _ => Err(parse_error("malformed I() arithmetic expression")),
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
        [Token::Ident(left), Token::Colon, Token::Ident(right)] => {
            ensure_column(model, left);
            ensure_column(model, right);
            add_role(
                model,
                &format!("{}:{}", left, right),
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

fn ensure_column(model: &mut FormulaModel, name: &str) {
    model.columns.entry(name.to_string()).or_default();
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
        assert_eq!(
            ast.offset.as_ref().and_then(NumericExpr::as_column_name),
            Some("log_exposure")
        );
        assert!(ast.columns.contains_key("A"));
        assert!(ast.columns.contains_key("B"));
    }

    #[test]
    fn parses_colon_interaction_without_mains() {
        let ast = parse("y ~ a:b + (1 | g)").unwrap();
        assert!(ast.columns["a:b"].has_role(ColumnRole::Interaction));
        assert!(!ast.columns["a"].has_role(ColumnRole::FixedEffect));
        assert!(!ast.columns["b"].has_role(ColumnRole::FixedEffect));
    }

    #[test]
    fn parses_log_and_identity_terms() {
        let ast = parse("y ~ log(x) + I(x^2) + (1 | g)").unwrap();
        assert!(ast.columns["log(x)"].expr.is_some());
        assert!(ast.columns["I(x^2)"].expr.is_some());
        assert!(ast.columns.contains_key("x"));
    }

    #[test]
    fn parses_transformed_offset() {
        let ast = parse("y ~ x + offset(log(w)) + (1 | g)").unwrap();
        assert!(matches!(
            ast.offset,
            Some(NumericExpr::Unary {
                op: UnaryOp::Log,
                ..
            })
        ));
        assert!(ast.columns.contains_key("w"));
    }

    #[test]
    fn malformed_inputs_are_errors_not_panics() {
        for formula in ["", "y", "~ y", "y ~ x + * z", "y ~ (1 |", "Rey ~ 1 (+ (1 |"] {
            assert!(parse(formula).is_err(), "{formula:?} should be rejected");
        }
    }
}
