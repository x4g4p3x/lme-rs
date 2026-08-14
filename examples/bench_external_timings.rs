//! Fit-only timings for workloads the Julia fair harness does not cover:
//! `nlmer`, post-fit Satterthwaite / Kenward–Roger ANOVA, and a Rust `lmer` baseline
//! for Python FFI overhead comparisons.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use lme_rs::anova::DdfMethod;
use lme_rs::lmer;
use lme_rs::nlmm::{nlmer, NlmmStart};
use polars::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy)]
enum Case {
    SleepstudyLmer,
    SleepstudySatterthwaite,
    SleepstudyKenwardRoger,
    OrangeNlmer,
}

impl Case {
    fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "sleepstudy_lmer" => Ok(Self::SleepstudyLmer),
            "sleepstudy_satterthwaite" => Ok(Self::SleepstudySatterthwaite),
            "sleepstudy_kenward_roger" => Ok(Self::SleepstudyKenwardRoger),
            "orange_nlmer" => Ok(Self::OrangeNlmer),
            other => anyhow::bail!("unknown case {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SleepstudyLmer => "sleepstudy_lmer",
            Self::SleepstudySatterthwaite => "sleepstudy_satterthwaite",
            Self::SleepstudyKenwardRoger => "sleepstudy_kenward_roger",
            Self::OrangeNlmer => "orange_nlmer",
        }
    }

    fn family(self) -> &'static str {
        match self {
            Self::SleepstudyLmer => "lmm_fit",
            Self::SleepstudySatterthwaite | Self::SleepstudyKenwardRoger => "post_fit_inference",
            Self::OrangeNlmer => "nlmm_fit",
        }
    }

    fn formula(self) -> &'static str {
        match self {
            Self::SleepstudyLmer | Self::SleepstudySatterthwaite | Self::SleepstudyKenwardRoger => {
                "Reaction ~ Days + (Days | Subject)"
            }
            Self::OrangeNlmer => "circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree",
        }
    }
}

#[derive(Serialize)]
struct TimingSummary {
    min_seconds: f64,
    max_seconds: f64,
    mean_seconds: f64,
    median_seconds: f64,
}

#[derive(Serialize)]
struct TimingReport {
    implementation: &'static str,
    case: &'static str,
    family: &'static str,
    formula: &'static str,
    n_obs: usize,
    warmups: usize,
    repeats: usize,
    samples_seconds: Vec<f64>,
    summary: TimingSummary,
}

fn load_csv(path: &str) -> DataFrame {
    let path = PathBuf::from(path);
    let file = File::open(&path).unwrap_or_else(|err| panic!("open {}: {err}", path.display()));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn summarize(samples: &[f64]) -> TimingSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let median = if sorted.len().is_multiple_of(2) {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    TimingSummary {
        min_seconds: sorted[0],
        max_seconds: *sorted.last().unwrap(),
        mean_seconds: mean,
        median_seconds: median,
    }
}

fn time_samples<F: FnMut()>(warmups: usize, repeats: usize, mut body: F) -> Vec<f64> {
    for _ in 0..warmups {
        body();
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        body();
        samples.push(started.elapsed().as_secs_f64());
    }
    samples
}

fn orange_start() -> NlmmStart {
    let mut start = NlmmStart::new();
    start.insert("Asym".to_string(), 200.0);
    start.insert("xmid".to_string(), 725.0);
    start.insert("scal".to_string(), 350.0);
    start
}

fn run_case(case: Case, warmups: usize, repeats: usize) -> TimingReport {
    match case {
        Case::SleepstudyLmer => {
            let df = load_csv("tests/data/sleepstudy.csv");
            let formula = case.formula();
            let samples = time_samples(warmups, repeats, || {
                lmer(formula, &df, true).expect("lmer");
            });
            TimingReport {
                implementation: "rust",
                case: case.as_str(),
                family: case.family(),
                formula,
                n_obs: df.height(),
                warmups,
                repeats,
                summary: summarize(&samples),
                samples_seconds: samples,
            }
        }
        Case::SleepstudySatterthwaite => {
            let df = load_csv("tests/data/sleepstudy.csv");
            let base = lmer(case.formula(), &df, true).expect("lmer");
            let samples = time_samples(warmups, repeats, || {
                let mut fit = base.clone();
                fit.with_satterthwaite(&df).expect("satterthwaite");
                fit.anova(DdfMethod::Satterthwaite).expect("anova");
            });
            TimingReport {
                implementation: "rust",
                case: case.as_str(),
                family: case.family(),
                formula: case.formula(),
                n_obs: df.height(),
                warmups,
                repeats,
                summary: summarize(&samples),
                samples_seconds: samples,
            }
        }
        Case::SleepstudyKenwardRoger => {
            let df = load_csv("tests/data/sleepstudy.csv");
            let base = lmer(case.formula(), &df, true).expect("lmer");
            let samples = time_samples(warmups, repeats, || {
                let mut fit = base.clone();
                fit.with_kenward_roger(&df).expect("kenward-roger");
                fit.anova(DdfMethod::KenwardRoger).expect("anova");
            });
            TimingReport {
                implementation: "rust",
                case: case.as_str(),
                family: case.family(),
                formula: case.formula(),
                n_obs: df.height(),
                warmups,
                repeats,
                summary: summarize(&samples),
                samples_seconds: samples,
            }
        }
        Case::OrangeNlmer => {
            let df = load_csv("tests/data/orange.csv");
            let formula = case.formula();
            let samples = time_samples(warmups, repeats, || {
                nlmer(formula, &df, orange_start(), false).expect("nlmer");
            });
            TimingReport {
                implementation: "rust",
                case: case.as_str(),
                family: case.family(),
                formula,
                n_obs: df.height(),
                warmups,
                repeats,
                summary: summarize(&samples),
                samples_seconds: samples,
            }
        }
    }
}

fn print_help() {
    eprintln!(
        "Usage: bench_external_timings --case <name> [--warmups N] [--repeats N]\n\
         Cases: sleepstudy_lmer, sleepstudy_satterthwaite, sleepstudy_kenward_roger, orange_nlmer"
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut case = None;
    let mut warmups = 1usize;
    let mut repeats = 5usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--case" => {
                case =
                    Some(Case::parse(&args.next().expect("--case needs a value")).expect("case"));
            }
            "--warmups" => {
                warmups = args
                    .next()
                    .expect("--warmups needs a value")
                    .parse()
                    .expect("warmups");
            }
            "--repeats" => {
                repeats = args
                    .next()
                    .expect("--repeats needs a value")
                    .parse()
                    .expect("repeats");
            }
            other => {
                eprintln!("unknown argument {other}");
                print_help();
                std::process::exit(2);
            }
        }
    }
    let case = case.unwrap_or_else(|| {
        print_help();
        std::process::exit(2);
    });
    let report = run_case(case, warmups, repeats);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
