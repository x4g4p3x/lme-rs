//! Fair Rust vs Julia timing: shared CSV fixtures; fit-only samples (plus optional Rust phases).
//!
//! Data generation matches [`benches/bench_math.rs`]. The `time` subcommand loads data once,
//! runs warmup fits, then records wall-clock samples. With `--with-phases`, Rust also reports
//! `prepare_lmer` and `fit_prepared` medians (LMM / weighted LMM only).

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use lme_rs::family::Family;
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FitModel {
    Lmm,
    LmmWeighted,
    GlmmBinomial,
    GlmmPoisson,
}

impl FitModel {
    fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "lmm" => Ok(Self::Lmm),
            "lmm_weighted" => Ok(Self::LmmWeighted),
            "glmm_binomial" => Ok(Self::GlmmBinomial),
            "glmm_poisson" => Ok(Self::GlmmPoisson),
            other => anyhow::bail!("unknown --model {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Lmm => "lmm",
            Self::LmmWeighted => "lmm_weighted",
            Self::GlmmBinomial => "glmm_binomial",
            Self::GlmmPoisson => "glmm_poisson",
        }
    }

    fn supports_phases(self) -> bool {
        matches!(self, Self::Lmm | Self::LmmWeighted)
    }
}

#[derive(Debug, Clone, Serialize)]
struct TimingSummary {
    min_seconds: f64,
    max_seconds: f64,
    mean_seconds: f64,
    median_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    stdev_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MetricSamples {
    samples_seconds: Vec<f64>,
    summary: TimingSummary,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    implementation: &'static str,
    case: String,
    formula: String,
    model: &'static str,
    reml: bool,
    n_obs: usize,
    warmups: usize,
    repeats: usize,
    /// Cold end-to-end fit (`lmer` / `glmer` / weighted `lmer`).
    cold_fit: MetricSamples,
    #[serde(skip_serializing_if = "Option::is_none")]
    prepare_lmer: Option<MetricSamples>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fit_prepared: Option<MetricSamples>,
}

fn generate_large_synthetic_df(n_obs: usize, n_groups: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut y = Vec::with_capacity(n_obs);
    let mut x1 = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);

    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let current_x = normal.sample(&mut rng);
        let current_y = 1.0 + 2.0 * current_x + normal.sample(&mut rng);

        y.push(current_y);
        x1.push(current_x);
        group.push(format!("G{g}"));
    }

    df!(
        "y" => &y,
        "x" => &x1,
        "group" => &group
    )
    .unwrap()
}

/// Large correlated random-intercept/random-slope fixture for the fair harness.
fn generate_large_random_slopes_df(n_obs: usize, n_groups: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(20260709);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let random_intercepts: Vec<f64> = (0..n_groups).map(|_| normal.sample(&mut rng)).collect();
    let random_slopes: Vec<f64> = random_intercepts
        .iter()
        .map(|&intercept| 0.35 * intercept + 0.65 * normal.sample(&mut rng))
        .collect();

    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let x_i = normal.sample(&mut rng);
        y.push(
            1.0 + 1.25 * x_i
                + random_intercepts[g]
                + random_slopes[g] * x_i
                + 0.25 * normal.sample(&mut rng),
        );
        x.push(x_i);
        group.push(format!("G{g}"));
    }
    df!("y" => y, "x" => x, "group" => group).unwrap()
}

fn generate_large_crossed_df(n_obs: usize, n_plates: usize, n_samples: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(1337);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let plate_effects: Vec<f64> = (0..n_plates).map(|_| normal.sample(&mut rng)).collect();
    let sample_effects: Vec<f64> = (0..n_samples).map(|_| normal.sample(&mut rng)).collect();

    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut plate = Vec::with_capacity(n_obs);
    let mut sample = Vec::with_capacity(n_obs);

    for _ in 0..n_obs {
        let plate_idx = rng.random_range(0..n_plates);
        let sample_idx = rng.random_range(0..n_samples);
        let x_i = normal.sample(&mut rng);
        let noise = 0.25 * normal.sample(&mut rng);
        let y_i = 1.5 + 0.75 * x_i + plate_effects[plate_idx] + sample_effects[sample_idx] + noise;

        y.push(y_i);
        x.push(x_i);
        plate.push(format!("P{plate_idx}"));
        sample.push(format!("S{sample_idx}"));
    }

    df!(
        "y" => y,
        "x" => x,
        "plate" => plate,
        "sample" => sample
    )
    .unwrap()
}

fn generate_large_nested_df(
    n_batches: usize,
    casks_per_batch: usize,
    reps_per_cask: usize,
) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(2026);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let batch_effects: Vec<f64> = (0..n_batches).map(|_| normal.sample(&mut rng)).collect();
    let cask_effects: Vec<Vec<f64>> = (0..n_batches)
        .map(|_| {
            (0..casks_per_batch)
                .map(|_| normal.sample(&mut rng))
                .collect()
        })
        .collect();

    let total = n_batches * casks_per_batch * reps_per_cask;
    let mut y = Vec::with_capacity(total);
    let mut x = Vec::with_capacity(total);
    let mut batch = Vec::with_capacity(total);
    let mut cask = Vec::with_capacity(total);

    for (batch_idx, casks) in cask_effects.iter().enumerate().take(n_batches) {
        for (cask_idx, &cask_effect) in casks.iter().enumerate().take(casks_per_batch) {
            for _ in 0..reps_per_cask {
                let x_i = normal.sample(&mut rng);
                let noise = 0.2 * normal.sample(&mut rng);
                let y_i = 2.0 + 1.25 * x_i + batch_effects[batch_idx] + cask_effect + noise;

                y.push(y_i);
                x.push(x_i);
                batch.push(format!("B{batch_idx}"));
                cask.push(format!("C{cask_idx}"));
            }
        }
    }

    df!(
        "y" => y,
        "x" => x,
        "batch" => batch,
        "cask" => cask
    )
    .unwrap()
}

fn sleepstudy_weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + (i % 5) as f64 * 0.1).collect()
}

fn write_csv(df: &DataFrame, path: &PathBuf) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = df.clone();
    let mut file = File::create(path)?;
    CsvWriter::new(&mut file).finish(&mut out)?;
    Ok(())
}

fn load_csv(path: &PathBuf) -> anyhow::Result<DataFrame> {
    let mut file = File::open(path)?;
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()?;
    normalize_fixture_df(df)
}

/// Cast common grouping columns to strings (matches comparison examples).
fn normalize_fixture_df(df: DataFrame) -> anyhow::Result<DataFrame> {
    let string_cols = [
        "group", "plate", "sample", "batch", "cask", "Subject", "herd", "BROOD",
    ];
    let cols_to_cast: Vec<&str> = string_cols
        .iter()
        .copied()
        .filter(|name| df.column(name).is_ok())
        .collect();
    if cols_to_cast.is_empty() {
        return Ok(df);
    }
    let mut lazy = df.lazy();
    for name in cols_to_cast {
        lazy = lazy.with_column(col(name).cast(DataType::String));
    }
    Ok(lazy.collect()?)
}

fn summarize(samples: &[f64]) -> TimingSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let median_seconds = if n == 0 {
        0.0
    } else if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let mean_seconds = if n == 0 {
        0.0
    } else {
        sorted.iter().sum::<f64>() / n as f64
    };
    let stdev_seconds = if n > 1 {
        let var = sorted
            .iter()
            .map(|v| {
                let d = v - mean_seconds;
                d * d
            })
            .sum::<f64>()
            / (n - 1) as f64;
        Some(var.sqrt())
    } else {
        None
    };
    TimingSummary {
        min_seconds: *sorted.first().unwrap_or(&0.0),
        max_seconds: *sorted.last().unwrap_or(&0.0),
        mean_seconds,
        median_seconds,
        stdev_seconds,
    }
}

fn metric_from_samples(samples: Vec<f64>) -> MetricSamples {
    let summary = summarize(&samples);
    MetricSamples {
        samples_seconds: samples,
        summary,
    }
}

fn run_cold_fit(model: FitModel, formula: &str, df: &DataFrame, reml: bool) -> anyhow::Result<()> {
    match model {
        FitModel::Lmm => {
            lme_rs::lmer(formula, df, reml)?;
        }
        FitModel::LmmWeighted => {
            let w = ndarray::Array1::from_vec(sleepstudy_weights(df.height()));
            lme_rs::lmer_weighted(formula, df, reml, Some(w))?;
        }
        FitModel::GlmmBinomial => {
            lme_rs::glmer(formula, df, Family::Binomial, 1)?;
        }
        FitModel::GlmmPoisson => {
            lme_rs::glmer(formula, df, Family::Poisson, 1)?;
        }
    }
    Ok(())
}

fn run_prepare(
    model: FitModel,
    formula: &str,
    df: &DataFrame,
) -> anyhow::Result<lme_rs::LmerPrepared> {
    match model {
        FitModel::Lmm => Ok(lme_rs::prepare_lmer(formula, df)?),
        FitModel::LmmWeighted => {
            let w = ndarray::Array1::from_vec(sleepstudy_weights(df.height()));
            Ok(lme_rs::prepare_lmer_weighted(formula, df, Some(w))?)
        }
        _ => anyhow::bail!("prepare phases are only supported for LMM models"),
    }
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> anyhow::Result<&'a str> {
    let idx = args
        .iter()
        .position(|a| a == flag)
        .ok_or_else(|| anyhow::anyhow!("missing required flag {flag}"))?;
    args.get(idx + 1)
        .map(|s| s.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

fn arg_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_usize(args: &[String], flag: &str) -> anyhow::Result<usize> {
    Ok(arg_value(args, flag)?.parse()?)
}

fn parse_bool(args: &[String], flag: &str) -> anyhow::Result<bool> {
    match arg_value(args, flag)?.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => anyhow::bail!("invalid boolean for {flag}: {other}"),
    }
}

fn cmd_generate(args: &[String]) -> anyhow::Result<()> {
    let output = PathBuf::from(arg_value(args, "--output")?);
    let kind = arg_value(args, "--kind")?;
    let df = match kind {
        "random_intercept" => {
            let n_obs = parse_usize(args, "--n-obs")?;
            let n_groups = parse_usize(args, "--n-groups")?;
            generate_large_synthetic_df(n_obs, n_groups)
        }
        "random_slopes" => {
            let n_obs = parse_usize(args, "--n-obs")?;
            let n_groups = parse_usize(args, "--n-groups")?;
            generate_large_random_slopes_df(n_obs, n_groups)
        }
        "crossed" => {
            let n_obs = parse_usize(args, "--n-obs")?;
            let n_plates = parse_usize(args, "--n-plates")?;
            let n_samples = parse_usize(args, "--n-samples")?;
            generate_large_crossed_df(n_obs, n_plates, n_samples)
        }
        "nested" => {
            let n_batches = parse_usize(args, "--n-batches")?;
            let casks_per_batch = parse_usize(args, "--casks-per-batch")?;
            let reps_per_cask = parse_usize(args, "--reps-per-cask")?;
            generate_large_nested_df(n_batches, casks_per_batch, reps_per_cask)
        }
        other => anyhow::bail!("unknown --kind {other}"),
    };
    write_csv(&df, &output)?;
    eprintln!("wrote {} rows to {}", df.height(), output.display());
    Ok(())
}

fn cmd_time(args: &[String]) -> anyhow::Result<()> {
    let data_path = PathBuf::from(arg_value(args, "--data")?);
    let case = arg_value(args, "--case")?.to_string();
    let formula = arg_value(args, "--formula")?.to_string();
    let reml = parse_bool(args, "--reml")?;
    let warmups = parse_usize(args, "--warmups")?;
    let repeats = parse_usize(args, "--repeats")?;
    let model = if arg_flag(args, "--model") {
        FitModel::parse(arg_value(args, "--model")?)?
    } else {
        FitModel::Lmm
    };
    let with_phases = arg_flag(args, "--with-phases");

    let df = load_csv(&data_path)?;
    let n_obs = df.height();

    for _ in 0..warmups {
        run_cold_fit(model, &formula, &df, reml)?;
    }

    let mut cold_samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        run_cold_fit(model, &formula, &df, reml)?;
        cold_samples.push(started.elapsed().as_secs_f64());
    }

    let (prepare_lmer, fit_prepared) = if with_phases && model.supports_phases() {
        let mut prepare_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = Instant::now();
            let _ = run_prepare(model, &formula, &df)?;
            prepare_samples.push(started.elapsed().as_secs_f64());
        }

        let prepared = run_prepare(model, &formula, &df)?;
        for _ in 0..warmups {
            let _ = lme_rs::fit_prepared(&prepared, reml)?;
        }
        let mut hot_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = Instant::now();
            let _ = lme_rs::fit_prepared(&prepared, reml)?;
            hot_samples.push(started.elapsed().as_secs_f64());
        }
        (
            Some(metric_from_samples(prepare_samples)),
            Some(metric_from_samples(hot_samples)),
        )
    } else {
        (None, None)
    };

    let report = TimingReport {
        implementation: "rust",
        case,
        formula,
        model: model.as_str(),
        reml,
        n_obs,
        warmups,
        repeats,
        cold_fit: metric_from_samples(cold_samples),
        prepare_lmer,
        fit_prepared,
    };
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  \
     bench_fair_rust_julia generate --kind random_intercept --n-obs N --n-groups G --output PATH\n  \
     bench_fair_rust_julia generate --kind crossed --n-obs N --n-plates P --n-samples S --output PATH\n  \
     bench_fair_rust_julia generate --kind nested --n-batches B --casks-per-batch C --reps-per-cask R --output PATH\n  \
     bench_fair_rust_julia time --case NAME --data PATH --formula F --reml true|false \\\n    \
       [--model lmm|lmm_weighted|glmm_binomial|glmm_poisson] [--with-phases] --warmups N --repeats N"
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || arg_flag(&args, "--help") || arg_flag(&args, "-h") {
        eprintln!("{}", usage());
        return Ok(());
    }
    match args[0].as_str() {
        "generate" => cmd_generate(&args[1..]),
        "time" => cmd_time(&args[1..]),
        other => anyhow::bail!("unknown subcommand {other}\n{}", usage()),
    }
}
