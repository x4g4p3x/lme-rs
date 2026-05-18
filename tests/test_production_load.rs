//! Production-shaped performance, throughput, and memory smoke tests.
//!
//! - Default `cargo test` runs **smoke** workloads (correctness + convergence).
//! - `cargo test --release --test test_production_load` also enforces **wall-clock gates**
//!   on key workloads (`#[cfg(not(debug_assertions))]`).
//! - `cargo test --release --test test_production_load -- --ignored --test-threads=1` runs
//!   **heavy** shapes (largest rows, many groups, high random-effect rank, service loop).
//!
//! **Debug vs release:** timing gates apply only in release builds. For a local comparison,
//! run the same command with and without `--release` and compare printed timings.
//!
//! **Memory peak (Linux):** set `LME_MEM_LIMIT_MB` to assert `VmHWM` after a large fit
//! stays below the budget (best-effort; CI does not set this by default).
//!
//! **Criterion microbenches** for the same production shapes (optional, large):
//! `cargo bench --features slow-production-benches --bench bench_load_production`

use lme_rs::lmer;
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::hint::black_box;
use std::time::Instant;

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

/// Two correlated predictors plus a random vector of dimension 3 (intercept + 2 slopes) per group.
fn synthetic_mid_rank_re(n_obs: usize, n_groups: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut y = Vec::with_capacity(n_obs);
    let mut x1 = Vec::with_capacity(n_obs);
    let mut x2 = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let a = normal.sample(&mut rng);
        let b = normal.sample(&mut rng);
        y.push(0.5 + 0.4 * a + 0.35 * b + 0.12 * normal.sample(&mut rng));
        x1.push(a);
        x2.push(b);
        group.push(format!("G{}", g));
    }
    df!("y" => y, "x1" => x1, "x2" => x2, "group" => group).unwrap()
}

/// Three slopes + intercept in the random vector (k = 4) for stress workloads.
fn synthetic_high_rank_re(n_obs: usize, n_groups: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut y = Vec::with_capacity(n_obs);
    let mut x1 = Vec::with_capacity(n_obs);
    let mut x2 = Vec::with_capacity(n_obs);
    let mut x3 = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        let g = rng.random_range(0..n_groups);
        let a = normal.sample(&mut rng);
        let b = normal.sample(&mut rng);
        let c = normal.sample(&mut rng);
        y.push(0.5 + 0.4 * a + 0.3 * b - 0.2 * c + 0.15 * normal.sample(&mut rng));
        x1.push(a);
        x2.push(b);
        x3.push(c);
        group.push(format!("G{}", g));
    }
    df!(
        "y" => y,
        "x1" => x1,
        "x2" => x2,
        "x3" => x3,
        "group" => group
    )
    .unwrap()
}

#[cfg(not(debug_assertions))]
fn gate_sleepstudy_ms() -> u64 {
    std::env::var("LME_GATE_SLEEPSTUDY_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120_000)
}

#[cfg(not(debug_assertions))]
fn gate_smoke_lmer_ms() -> u64 {
    std::env::var("LME_GATE_SMOKE_LMER_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(180_000)
}

#[cfg(not(debug_assertions))]
fn gate_prediction_rows_per_sec() -> f64 {
    std::env::var("LME_GATE_PRED_ROWS_PER_SEC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25_000.0)
}

#[cfg(target_os = "linux")]
fn vm_hwm_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn vm_hwm_kb() -> Option<u64> {
    None
}

#[test]
fn smoke_random_intercept_converges() {
    let df = synthetic_random_intercept(6_000, 400, 11);
    let fit = lmer("y ~ x + (1 | group)", &df, false).expect("fit");
    assert!(fit.converged.unwrap_or(false), "expected convergence");
    assert!(fit.theta.as_ref().unwrap()[0].is_finite());
}

#[test]
fn smoke_random_slopes_converges() {
    let df = synthetic_random_slopes(6_000, 300, 17);
    let fit = lmer("y ~ x + (x | group)", &df, false).expect("fit");
    assert!(fit.converged.unwrap_or(false));
    assert_eq!(fit.theta.as_ref().unwrap().len(), 3);
}

#[test]
fn smoke_high_random_effect_rank_converges() {
    let df = synthetic_mid_rank_re(8_000, 350, 23);
    let fit = lmer("y ~ x1 + x2 + (x1 + x2 | group)", &df, false).expect("vector-valued RE fit");
    assert!(fit.converged.unwrap_or(false));
    let nt = fit.theta.as_ref().unwrap().len();
    assert_eq!(nt, 6, "3x3 lower Cholesky factor has 6 free parameters");
}

#[test]
fn release_gate_sleepstudy_lmer_wall_clock() {
    let mut file = std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("csv");
    let t0 = Instant::now();
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).expect("lmer");
    let elapsed = t0.elapsed();
    println!(
        "sleepstudy lmer: {:?} (debug_assertions={})",
        elapsed,
        cfg!(debug_assertions)
    );
    assert!(fit.converged.unwrap_or(false));
    #[cfg(not(debug_assertions))]
    {
        let limit_ms = gate_sleepstudy_ms();
        assert!(
            elapsed.as_millis() <= u128::from(limit_ms),
            "sleepstudy lmer exceeded gate: {:?} > {} ms (override with LME_GATE_SLEEPSTUDY_MS)",
            elapsed,
            limit_ms
        );
    }
}

#[test]
fn release_gate_smoke_scale_lmer_wall_clock() {
    let df = synthetic_random_slopes(25_000, 800, 29);
    let t0 = Instant::now();
    let fit = lmer("y ~ x + (x | group)", &df, false).expect("lmer");
    let elapsed = t0.elapsed();
    println!(
        "25k random-slopes lmer: {:?} (debug_assertions={})",
        elapsed,
        cfg!(debug_assertions)
    );
    assert!(fit.converged.unwrap_or(false));
    #[cfg(not(debug_assertions))]
    {
        let limit_ms = gate_smoke_lmer_ms();
        assert!(
            elapsed.as_millis() <= u128::from(limit_ms),
            "smoke-scale lmer exceeded gate: {:?} > {} ms (override with LME_GATE_SMOKE_LMER_MS)",
            elapsed,
            limit_ms
        );
    }
}

#[test]
fn release_gate_prediction_throughput_sleepstudy() {
    let mut file = std::fs::File::open("tests/data/sleepstudy.csv").expect("sleepstudy.csv");
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .expect("csv");
    let fit = lmer("Reaction ~ Days + (Days | Subject)", &df, true).expect("lmer");

    let days = df.column("Days").unwrap().cast(&DataType::Float64).unwrap();
    let days = days.f64().unwrap().into_no_null_iter().collect::<Vec<_>>();
    let subjects = df
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

    let repeats = 400usize;
    let n = days.len() * repeats;
    let mut out_days = Vec::with_capacity(n);
    let mut out_subjects = Vec::with_capacity(n);
    for _ in 0..repeats {
        out_days.extend(days.iter().copied());
        out_subjects.extend(subjects.iter().cloned());
    }
    let pred_df = df!("Days" => out_days, "Subject" => out_subjects).unwrap();

    let t0 = Instant::now();
    for _ in 0..3 {
        black_box(fit.predict(&pred_df).expect("predict"));
    }
    let elapsed = t0.elapsed();
    let rows_per_sec = (pred_df.height() * 3) as f64 / elapsed.as_secs_f64().max(1e-9);
    println!(
        "prediction throughput (~population): {:.0} rows/s over 3 iters (debug_assertions={})",
        rows_per_sec,
        cfg!(debug_assertions)
    );

    #[cfg(not(debug_assertions))]
    {
        let min_rps = gate_prediction_rows_per_sec();
        assert!(
            rows_per_sec >= min_rps,
            "prediction throughput {:.0} rows/s < gate {:.0} (set LME_GATE_PRED_ROWS_PER_SEC)",
            rows_per_sec,
            min_rps
        );
    }
}

#[test]
fn linux_memory_budget_optional() {
    let limit_mb: u64 = match std::env::var("LME_MEM_LIMIT_MB") {
        Ok(s) => s.parse().expect("LME_MEM_LIMIT_MB must be integer"),
        Err(_) => return,
    };
    let df = synthetic_random_intercept(80_000, 4_000, 41);
    let _fit = lmer("y ~ x + (1 | group)", &df, false).expect("fit");
    let Some(hwm_kb) = vm_hwm_kb() else {
        eprintln!("skip memory gate: VmHWM not available on this platform");
        return;
    };
    let peak_mb = hwm_kb / 1024;
    println!("VmHWM peak (resident high water) ~ {} MiB", peak_mb);
    assert!(
        peak_mb <= limit_mb,
        "memory gate: VmHWM peak {} MiB > {} MiB (raise limit or shrink workload)",
        peak_mb,
        limit_mb
    );
}

// --- Heavy workloads (`--ignored`) ---

#[test]
#[ignore = "heavy: largest expected row count (run with --release --ignored)"]
fn heavy_largest_expected_row_count() {
    let df = synthetic_random_intercept(250_000, 4_000, 101);
    let fit = lmer("y ~ x + (1 | group)", &df, false).expect("fit");
    assert!(fit.converged.unwrap_or(false));
}

#[test]
#[ignore = "heavy: high group count (run with --release --ignored)"]
fn heavy_highest_expected_group_count() {
    let df = synthetic_random_intercept(120_000, 20_000, 103);
    let fit = lmer("y ~ x + (1 | group)", &df, false).expect("fit");
    assert!(fit.converged.unwrap_or(false));
}

#[test]
#[ignore = "heavy: high random-effect dimension (run with --release --ignored)"]
fn heavy_highest_random_effect_dimension() {
    let df = synthetic_high_rank_re(35_000, 600, 107);
    let fit = lmer("y ~ x1 + x2 + x3 + (x1 + x2 + x3 | group)", &df, false).expect("fit");
    let theta = fit.theta.as_ref().expect("theta");
    assert_eq!(theta.len(), 10, "k=4 RE block => 10 Cholesky parameters");
    assert!(
        theta.iter().all(|t| t.is_finite()),
        "all theta elements should be finite"
    );
    let s2 = fit.sigma2.expect("sigma2");
    assert!(
        s2.is_finite() && s2 > 0.0,
        "sigma2 should be positive finite"
    );
    if !fit.converged.unwrap_or(false) {
        eprintln!(
            "note: k=4 vector RE may exhaust Nelder-Mead iteration budget; iterations={:?}",
            fit.iterations
        );
    }
}

#[test]
#[ignore = "heavy: repeated fitting service-like workload"]
fn heavy_service_like_repeated_fits() {
    for i in 0..40 {
        let df = synthetic_random_intercept(8_000, 600, 109 + i as u64);
        black_box(
            lmer("y ~ x + (1 | group)", black_box(&df), false)
                .expect("fit")
                .coefficients,
        );
    }
}
