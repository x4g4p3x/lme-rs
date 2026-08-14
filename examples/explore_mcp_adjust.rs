//! Compare unadjusted, Bonferroni, Holm, and Tukey–Kramer MCP p-values.
//!
//! Fits pastes `strength ~ cask + (1 | batch)` and dumps the `cask` Tukey family.

use lme_rs::lmer;
use lme_rs::mcp::{McpAdjust, McpType};
use polars::prelude::*;
use std::fs::File;
use std::path::PathBuf;

fn load_pastes() -> DataFrame {
    let path = PathBuf::from("tests/data/pastes.csv");
    if !path.exists() {
        eprintln!("Could not find {}", path.display());
        eprintln!("Run this example from the repository root.");
        std::process::exit(1);
    }
    let file = File::open(&path).expect("open pastes.csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .expect("read pastes.csv")
}

fn main() {
    let df = load_pastes();
    let fit = lmer("strength ~ cask + (1 | batch)", &df, true).expect("lmer");
    let none = fit
        .glht("cask", McpType::Tukey, McpAdjust::None, None)
        .expect("glht none");
    let bonf = fit
        .glht("cask", McpType::Tukey, McpAdjust::Bonferroni, None)
        .expect("glht bonferroni");
    let holm = fit
        .glht("cask", McpType::Tukey, McpAdjust::Holm, None)
        .expect("glht holm");
    let tukey = fit
        .glht("cask", McpType::Tukey, McpAdjust::Tukey, None)
        .expect("glht tukey");

    println!(
        "term={}  mcp=Tukey  statistic={}",
        none.term, none.statistic
    );
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "contrast", "estimate", "raw_p", "bonferroni", "holm", "tukey"
    );
    for i in 0..none.comparisons.len() {
        println!(
            "{:<12} {:>10.4} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            none.comparisons[i],
            none.estimate[i],
            none.p_value[i],
            bonf.p_adjust[i],
            holm.p_adjust[i],
            tukey.p_adjust[i]
        );
        assert!(
            bonf.p_adjust[i] + 1e-12 >= none.p_value[i],
            "Bonferroni must not be anti-conservative vs raw p"
        );
        assert!(
            holm.p_adjust[i] + 1e-12 >= none.p_value[i],
            "Holm must not be anti-conservative vs raw p"
        );
        assert!(
            holm.p_adjust[i] <= bonf.p_adjust[i] + 1e-12,
            "Holm must be no harsher than Bonferroni"
        );
        assert!(
            tukey.p_adjust[i].is_finite() && tukey.p_adjust[i] >= 0.0,
            "Tukey–Kramer p-value must be finite"
        );
    }
    println!("OK");
}
