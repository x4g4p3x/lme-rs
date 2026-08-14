//! Standalone Wilkinson AST probe for the native formula parser.
//!
//! Prints a stable dump of intercept flags, generated columns, and random-effect
//! blocks for a catalog of edge-case formulas. Replaces the old fiasto JSON probes.

use lme_rs::formula::{parse, ColumnRole, FormulaModel};

struct CatalogEntry {
    formula: &'static str,
    expect_intercept: bool,
    expect_re: bool,
    expect_uncorrelated: bool,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        formula: "Reaction ~ Days + (Days | Subject)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "Reaction ~ Days + (Days || Subject)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: true,
    },
    CatalogEntry {
        formula: "Reaction ~ Days + I(Days^2) + (1 | Subject)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "Reaction ~ sqrt(Days) + (1 | Subject)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "Reaction ~ poly(Days, 2) + (1 | Subject)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "Reaction ~ ns(Days, 3) + (1 | Subject)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "strength ~ 0 + cask + (1 | batch)",
        expect_intercept: false,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "strength ~ 1 + (1 | batch/cask)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "y ~ log(x) + offset(log(w)) + (1 | g)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
    CatalogEntry {
        formula: "y ~ a:b + (1 | g)",
        expect_intercept: true,
        expect_re: true,
        expect_uncorrelated: false,
    },
];

fn dump(model: &FormulaModel) {
    println!("  intercept: {}", model.metadata.has_intercept);
    println!(
        "  random_effects_model: {}",
        model.metadata.is_random_effects_model
    );
    println!("  generated: {:?}", model.all_generated_columns);
    if let Some(offset) = &model.offset {
        println!("  offset: {}", offset.label());
    }

    let mut grouping: Vec<_> = model
        .columns
        .iter()
        .filter(|(_, info)| info.has_role(ColumnRole::GroupingVariable))
        .map(|(name, info)| (name.clone(), info.random_effects.clone()))
        .collect();
    grouping.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, effects) in grouping {
        for effect in effects {
            println!(
                "  RE {name}: correlated={} intercept={} slopes={:?}",
                effect.correlated, effect.has_intercept, effect.variables
            );
        }
    }
}

fn main() {
    println!("native formula AST catalog ({} formulas)", CATALOG.len());
    for entry in CATALOG {
        println!("\nformula: {}", entry.formula);
        let model = parse(entry.formula).unwrap_or_else(|err| {
            panic!("parse failed for {}: {err}", entry.formula);
        });
        assert_eq!(
            model.metadata.has_intercept, entry.expect_intercept,
            "{}",
            entry.formula
        );
        assert_eq!(
            model.metadata.is_random_effects_model, entry.expect_re,
            "{}",
            entry.formula
        );
        if entry.expect_uncorrelated {
            let uncorrelated = model
                .columns
                .values()
                .any(|info| info.random_effects.iter().any(|effect| !effect.correlated));
            assert!(
                uncorrelated,
                "{} should declare an uncorrelated RE block",
                entry.formula
            );
        }
        dump(&model);
    }
    println!("\nOK");
}
