use lme_rs::{lmer, DdfMethod};
use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Rust (lme-rs) ===");
    let mut file = File::open("tests/data/pastes.csv")?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;

    // Evaluate lmer and Satterthwaite implicitly
    let mut fit = lmer("strength ~ cask + (1 | batch)", &df, true)?;
    fit.with_satterthwaite(&df)?;

    println!("{}", fit);

    println!("\n=== Type III ANOVA (Satterthwaite 1-DoF / Joint Wald) ===");
    let anova_res = fit.anova(DdfMethod::Satterthwaite)?;
    println!("{}", anova_res);

    Ok(())
}
