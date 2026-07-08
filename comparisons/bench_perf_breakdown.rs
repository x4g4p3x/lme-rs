//! Performance breakdown for fair-harness fixtures (`LME_PERF_DIAG=1`).
//!
//! ```text
//! cargo run --release --example bench_perf_breakdown -- \
//!   --case crossed_20k \
//!   --data benchmark-results/fair-rust-julia-data/crossed_20k.csv \
//!   --formula "y ~ x + (1 | plate) + (1 | sample)" \
//!   --reml false
//! ```

use std::path::PathBuf;
use std::time::Instant;

use polars::prelude::*;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct BreakdownReport {
    case: String,
    formula: String,
    reml: bool,
    n_obs: usize,
    optimizer_iterations: u64,
    fit_wall_seconds: f64,
    #[serde(flatten)]
    perf: lme_rs::perf_diag::PerfReport,
}

fn parse_bool(s: &str) -> bool {
    matches!(s, "1" | "true" | "TRUE" | "True")
}

fn usage() -> &'static str {
    "Usage:\n  \
     bench_perf_breakdown --case NAME --data PATH --formula F --reml true|false [--warmups N]\n\n\
     Sets LME_PERF_DIAG=1 and prints a JSON breakdown of where fit time went."
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("LME_PERF_DIAG", "1");

    let args: Vec<String> = std::env::args().collect();
    let mut case = String::from("crossed_20k");
    let mut data_path: Option<PathBuf> = None;
    let mut formula: Option<String> = None;
    let mut reml = false;
    let mut warmups = 1usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--case" => {
                i += 1;
                case = args.get(i).cloned().ok_or("missing --case value")?;
            }
            "--data" => {
                i += 1;
                data_path = Some(PathBuf::from(args.get(i).ok_or("missing --data value")?));
            }
            "--formula" => {
                i += 1;
                formula = Some(args.get(i).cloned().ok_or("missing --formula value")?);
            }
            "--reml" => {
                i += 1;
                reml = parse_bool(args.get(i).ok_or("missing --reml value")?);
            }
            "--warmups" => {
                i += 1;
                warmups = args.get(i).ok_or("missing --warmups value")?.parse()?;
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage()).into()),
        }
        i += 1;
    }

    let data_path = data_path.ok_or_else(|| format!("--data is required\n{}", usage()))?;
    let formula = formula.ok_or_else(|| format!("--formula is required\n{}", usage()))?;

    let mut file = std::fs::File::open(&data_path)?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;
    let n_obs = df.height();

    for _ in 0..warmups {
        let _ = lme_rs::lmer(&formula, &df, reml)?;
        lme_rs::perf_diag::reset();
    }

    lme_rs::perf_diag::reset();
    let started = Instant::now();
    let fit = lme_rs::lmer(&formula, &df, reml)?;
    let fit_wall = started.elapsed();
    lme_rs::perf_diag::set_fit_wall(fit_wall);

    let perf = lme_rs::perf_diag::take_report().ok_or("perf diagnostics produced no report")?;
    let report = BreakdownReport {
        case,
        formula,
        reml,
        n_obs,
        optimizer_iterations: fit.iterations.unwrap_or(0),
        fit_wall_seconds: fit_wall.as_secs_f64(),
        perf,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
