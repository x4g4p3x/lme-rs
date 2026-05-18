//! Criterion benchmarks for production-shaped loads (complement `tests/test_production_load.rs`).
//! Run (gated so default `cargo bench` stays fast):
//! `cargo bench --features slow-production-benches --bench bench_load_production`
//!
//! For a rough debug vs release comparison on the same machine, contrast this command with
//! `cargo build --benches` (debug) plus running the produced test binary, against the default
//! `cargo bench` optimized build (see `[profile.bench]` in the crate manifest).

use criterion::{criterion_group, criterion_main, Criterion};
use lme_rs::lmer;
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::hint::black_box;
fn synthetic_random_intercept(n_obs: usize, n_groups: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let xi = normal.sample(&mut rng);
        y.push(1.0 + 1.5 * xi + normal.sample(&mut rng));
        x.push(xi);
        group.push(format!("G{}", g));
    }
    df!("y" => y, "x" => x, "group" => group).unwrap()
}

fn synthetic_random_slopes(n_obs: usize, n_groups: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let xi = normal.sample(&mut rng);
        y.push(1.0 + 1.25 * xi + normal.sample(&mut rng));
        x.push(xi);
        group.push(format!("G{}", g));
    }
    df!("y" => y, "x" => x, "group" => group).unwrap()
}

fn load_sleepstudy() -> DataFrame {
    let mut file = File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv");
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("csv")
}

fn prediction_grid_sleepstudy(base: &DataFrame, repeats: usize) -> DataFrame {
    let days = base
        .column("Days")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    let days = days.f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
    let subjects = base
        .column("Subject")
        .unwrap()
        .cast(&DataType::String)
        .unwrap();
    let subjects = subjects
        .str()
        .unwrap()
        .into_no_null_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let total = days.len() * repeats;
    let mut out_days = Vec::with_capacity(total);
    let mut out_subjects = Vec::with_capacity(total);
    for _ in 0..repeats {
        out_days.extend(days.iter().copied());
        out_subjects.extend(subjects.iter().cloned());
    }
    df!("Days" => out_days, "Subject" => out_subjects).unwrap()
}

fn bench_production_largest_rows(c: &mut Criterion) {
    let df = synthetic_random_intercept(250_000, 4_000, 101);
    let mut g = c.benchmark_group("production_largest_rows");
    g.sample_size(10);
    g.bench_function("lmer_250k_rows_4k_groups", |b| {
        b.iter(|| {
            black_box(
                lmer(
                    black_box("y ~ x + (1 | group)"),
                    black_box(&df),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });
    g.finish();
}

fn bench_production_high_group_count(c: &mut Criterion) {
    let df = synthetic_random_intercept(120_000, 20_000, 103);
    let mut g = c.benchmark_group("production_high_group_count");
    g.sample_size(10);
    g.bench_function("lmer_120k_rows_20k_groups", |b| {
        b.iter(|| {
            black_box(
                lmer(
                    black_box("y ~ x + (1 | group)"),
                    black_box(&df),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });
    g.finish();
}

fn bench_production_random_slopes_wide(c: &mut Criterion) {
    let df = synthetic_random_slopes(80_000, 3_000, 105);
    let mut g = c.benchmark_group("production_random_slopes");
    g.sample_size(10);
    g.bench_function("lmer_80k_rows_3k_groups_x_slope", |b| {
        b.iter(|| {
            black_box(
                lmer(
                    black_box("y ~ x + (x | group)"),
                    black_box(&df),
                    black_box(false),
                )
                .unwrap(),
            );
        })
    });
    g.finish();
}

fn bench_production_service_burst(c: &mut Criterion) {
    let mut g = c.benchmark_group("production_service_burst");
    g.sample_size(10);
    g.bench_function("service_burst_12_sequential_fits_8k", |b| {
        b.iter(|| {
            for i in 0..12 {
                let df = synthetic_random_intercept(8_000, 600, 200 + i);
                black_box(lmer("y ~ x + (1 | group)", black_box(&df), false).unwrap());
            }
        })
    });
    g.finish();
}

fn bench_production_prediction_throughput(c: &mut Criterion) {
    let df = load_sleepstudy();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).unwrap();
    let pred_df = prediction_grid_sleepstudy(&df, 500);
    let mut g = c.benchmark_group("production_prediction");
    g.bench_function("predict_population_sleepstudy_500x_grid", |b| {
        b.iter(|| black_box(fit.predict(black_box(&pred_df)).unwrap()))
    });
    g.bench_function("predict_conditional_sleepstudy_500x_grid", |b| {
        b.iter(|| {
            black_box(
                fit.predict_conditional(black_box(&pred_df), black_box(false))
                    .unwrap(),
            )
        })
    });
    g.finish();
    eprintln!(
        "prediction grid rows per predict call: {}",
        pred_df.height()
    );
}

criterion_group!(
    benches,
    bench_production_largest_rows,
    bench_production_high_group_count,
    bench_production_random_slopes_wide,
    bench_production_service_burst,
    bench_production_prediction_throughput
);
criterion_main!(benches);
