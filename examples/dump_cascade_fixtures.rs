//! Provisional cascade fixture dump (tooling — not a golden source of truth).
//!
//! **Prefer R** when available:
//! `Rscript tests/generate_nlmm_fixtures.R`
//!
//! This binary fits with **lme-rs** and writes provisional CSV/JSON. Those outputs
//! must **not** replace lme4 goldens under `tests/data/` unless you are deliberately
//! bootstrapping fixtures without R.
//!
//! Default (safe): write under `target/cascade_fixture_dump/`.
//!
//! ```text
//! cargo run --example dump_cascade_fixtures --locked --release
//! cargo run --example dump_cascade_fixtures --locked --release -- --out-dir path/to/dir
//! cargo run --example dump_cascade_fixtures --locked --release -- --write-tests-data --force
//! ```
//!
//! `--write-tests-data` targets `tests/data/` and refuses to overwrite existing files
//! unless `--force` is also passed.

use lme_rs::family::Family;
use lme_rs::nlmm::{ssbiexp_eval, ssfpl_eval, ssweibull_eval, NlmmStart};
use lme_rs::{glmer, lmer, nlmer, ConfintMethod};
use polars::prelude::*;
use serde_json::json;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROVISIONAL_NOTE: &str =
    "Provisional lme-rs fixture; regenerate with R tests/generate_nlmm_fixtures.R for lme4 goldens";

struct Args {
    out_dir: PathBuf,
    write_tests_data: bool,
    force: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut out_dir = PathBuf::from("target/cascade_fixture_dump");
    let mut write_tests_data = false;
    let mut force = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--out-dir requires a path".to_string())?;
                out_dir = PathBuf::from(value);
            }
            "--write-tests-data" => write_tests_data = true,
            "--force" => force = true,
            "--help" | "-h" => {
                eprintln!(
                    "Usage: dump_cascade_fixtures [--out-dir DIR] [--write-tests-data] [--force]\n\
                     \n\
                     Default: write provisional fixtures under target/cascade_fixture_dump/.\n\
                     Prefer Rscript tests/generate_nlmm_fixtures.R for lme4 goldens.\n\
                     --write-tests-data writes under tests/data/ (requires --force to overwrite)."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if write_tests_data {
        out_dir = PathBuf::from("tests/data");
    }
    Ok(Args {
        out_dir,
        write_tests_data,
        force,
    })
}

fn create_writer(path: &Path, force: bool) -> Result<File, String> {
    if path.exists() && !force {
        return Err(format!(
            "refusing to overwrite existing file {} (pass --force to replace)",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    File::create(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_csv(path: &Path, df: &DataFrame, force: bool) -> Result<(), String> {
    let mut file = create_writer(path, force)?;
    CsvWriter::new(&mut file)
        .finish(&mut df.clone())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

fn write_json(path: &Path, value: serde_json::Value, force: bool) -> Result<(), String> {
    let mut file = create_writer(path, force)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&value).unwrap())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

fn fpl_df() -> DataFrame {
    let n_g = 5usize;
    let n_per = 12usize;
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut id = Vec::new();
    for g in 0..n_g {
        let b = if g % 2 == 0 { 1.2 } else { -0.8 };
        for j in 0..n_per {
            let xi = (j as f64) * 0.5;
            let (mu2, _) = ssfpl_eval(10.0, 50.0, 6.0, 2.0, xi);
            y.push(mu2 + b * 0.15 + 0.05 * ((j % 3) as f64 - 1.0));
            x.push(xi);
            id.push(format!("{}", g + 1));
        }
    }
    DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("id".into(), id),
    ])
    .unwrap()
}

fn biexp_df() -> DataFrame {
    let n_g = 5usize;
    let n_per = 12usize;
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut id = Vec::new();
    for g in 0..n_g {
        let b = if g % 2 == 0 { 0.4 } else { -0.3 };
        for j in 0..n_per {
            let xi = 0.1 + (j as f64) * 0.4;
            let (mu, _) = ssbiexp_eval(5.0 + b, 1.2_f64.ln(), 3.0, 0.3_f64.ln(), xi);
            y.push(mu + 0.02 * ((j % 4) as f64 - 1.5));
            x.push(xi);
            id.push(format!("{}", g + 1));
        }
    }
    DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("id".into(), id),
    ])
    .unwrap()
}

fn weibull_df() -> DataFrame {
    let n_g = 5usize;
    let n_per = 12usize;
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut id = Vec::new();
    for g in 0..n_g {
        let b = if g % 2 == 0 { 0.8 } else { -0.6 };
        for j in 0..n_per {
            let xi = 0.2 + (j as f64) * 0.35;
            let (mu, _) = ssweibull_eval(100.0 + b, 80.0, -1.0, 1.5, xi);
            y.push(mu + 0.25 * ((j % 3) as f64 - 1.0));
            x.push(xi);
            id.push(format!("{}", g + 1));
        }
    }
    DataFrame::new(vec![
        Column::new("y".into(), y),
        Column::new("x".into(), x),
        Column::new("id".into(), id),
    ])
    .unwrap()
}

fn dump_nlmm(
    out_dir: &Path,
    force: bool,
    name: &str,
    formula: &str,
    df: &DataFrame,
) -> Result<(), String> {
    let csv = out_dir.join(format!("{name}_synthetic.csv"));
    let json_path = out_dir.join(format!("{name}_nlmer.json"));
    write_csv(&csv, df, force)?;
    let fit = nlmer(formula, df, NlmmStart::new(), false)
        .map_err(|e| format!("{name}: nlmer failed: {e}"))?;
    let re_sd = fit
        .theta
        .as_ref()
        .and_then(|t| t.first().copied())
        .unwrap_or(f64::NAN);
    let sigma = fit.sigma2.unwrap_or(1.0).sqrt();
    let vc = re_sd * sigma;
    let value = json!({
        "model": formula,
        "provisional": true,
        "outputs": {
            "beta": fit.coefficients.to_vec(),
            "theta": fit.theta.as_ref().map(|t| t.to_vec()).unwrap_or_default(),
            "re_sd": vc,
            "sigma2": fit.sigma2,
            "logLik": fit.log_likelihood,
            "note": PROVISIONAL_NOTE
        }
    });
    write_json(&json_path, value, force)?;
    println!(
        "{name}: wrote {} and {} (beta={:?})",
        csv.display(),
        json_path.display(),
        fit.coefficients.to_vec()
    );
    Ok(())
}

fn run(args: &Args) -> Result<(), String> {
    if !Path::new("tests/data").exists() {
        return Err("run from repository root (tests/data missing)".into());
    }
    if args.write_tests_data {
        eprintln!(
            "WARNING: --write-tests-data targets tests/data/. Prefer Rscript tests/generate_nlmm_fixtures.R for lme4 goldens."
        );
        if !args.force {
            eprintln!("Existing files will not be overwritten unless --force is passed.");
        }
    } else {
        eprintln!(
            "Writing provisional fixtures under {} (not lme4 goldens).",
            args.out_dir.display()
        );
    }
    fs::create_dir_all(&args.out_dir).map_err(|e| format!("{}: {e}", args.out_dir.display()))?;

    dump_nlmm(
        &args.out_dir,
        args.force,
        "ssfpl",
        "y ~ SSfpl(x, A, B, xmid, scal) ~ A|id",
        &fpl_df(),
    )?;
    dump_nlmm(
        &args.out_dir,
        args.force,
        "ssbiexp",
        "y ~ SSbiexp(x, A1, lrc1, A2, lrc2) ~ A1|id",
        &biexp_df(),
    )?;
    dump_nlmm(
        &args.out_dir,
        args.force,
        "ssweibull",
        "y ~ SSweibull(x, Asym, Drop, lrc, pwr) ~ Asym|id",
        &weibull_df(),
    )?;

    let mut file = File::open("tests/data/cbpp_binary.csv").map_err(|e| e.to_string())?;
    let cbpp = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .map_err(|e| e.to_string())?;
    let fit_agq = glmer(
        "y ~ period2 + period3 + period4 + (1 | herd)",
        &cbpp,
        Family::Binomial,
        7,
    )
    .map_err(|e| format!("cbpp agq: {e}"))?;
    let agq_path = args.out_dir.join("glmm_binomial_agq.json");
    write_json(
        &agq_path,
        json!({
            "model": "y ~ period2 + period3 + period4 + (1 | herd) [Binomial n_agq=7]",
            "provisional": true,
            "outputs": {
                "beta": fit_agq.coefficients.to_vec(),
                "theta": fit_agq.theta.as_ref().map(|t| t.to_vec()).unwrap_or_default(),
                "deviance": fit_agq.deviance,
                "note": PROVISIONAL_NOTE
            }
        }),
        args.force,
    )?;
    println!("cbpp agq: wrote {}", agq_path.display());

    let mut ss = File::open("tests/data/sleepstudy.csv").map_err(|e| e.to_string())?;
    let sleep = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut ss)
        .finish()
        .map_err(|e| e.to_string())?;
    let fit = lmer("Reaction ~ Days + (1 | Subject)", &sleep, false)
        .map_err(|e| format!("sleepstudy: {e}"))?;
    let ci = fit
        .confint_with(0.95, ConfintMethod::Profile, Some(&sleep))
        .map_err(|e| format!("sleepstudy profile: {e}"))?;
    let ci_path = args.out_dir.join("sleepstudy_confint_profile.json");
    write_json(
        &ci_path,
        json!({
            "model": "Reaction ~ Days + (1 | Subject) ML",
            "level": 0.95,
            "method": "profile",
            "provisional": true,
            "outputs": {
                "names": ci.names,
                "estimate": fit.coefficients.to_vec(),
                "lower": ci.lower.to_vec(),
                "upper": ci.upper.to_vec(),
                "note": PROVISIONAL_NOTE
            }
        }),
        args.force,
    )?;
    println!("sleepstudy profile: wrote {}", ci_path.display());
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
