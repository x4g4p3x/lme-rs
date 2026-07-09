//! Independent `power2` (SSpower) fits per sensor — CPU parallel batch NLS.
//!
//! Demonstrates the **MATLAB / lmfit lane** (one curve per sensor) vs pooled
//! **`nlmer`** in `lme-rs`. See [docs/CALO_CALIBRATION.md](../docs/CALO_CALIBRATION.md).
//!
//! ```bash
//! cargo run --example batch_sspower_cpu --release
//! ```

use lme_rs::nlmm::sspower_eval;
use lme_rs::{nlmer, NlmmStart};
use polars::prelude::*;
use std::time::Instant;

const A_TRUE: f64 = 2.0;
const B_TRUE: f64 = 0.45;
const C_TRUE: f64 = 0.5;

fn power2(x: f64, a: f64, b: f64, c: f64) -> f64 {
    sspower_eval(a, b, c, x).0
}

/// Log-linear start for y ≈ a * x^b + c on positive x (same idea as nlmer selfStart).
fn start_power2(xy: &[(f64, f64)]) -> [f64; 3] {
    let pos: Vec<(f64, f64)> = xy
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite() && *x > 0.0)
        .collect();
    if pos.len() < 3 {
        return [1.0, 1.0, 0.0];
    }
    let y_min = pos.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let c = y_min - 0.05 * y_min.abs().max(1.0);
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut n = 0.0;
    for (x, y) in &pos {
        let resid = y - c;
        if resid <= 0.0 {
            continue;
        }
        let lx = x.ln();
        let ly = resid.ln();
        sum_x += lx;
        sum_y += ly;
        sum_xx += lx * lx;
        sum_xy += lx * ly;
        n += 1.0;
    }
    if n < 2.0 {
        return [1.0, 1.0, c];
    }
    let denom = n * sum_xx - sum_x * sum_x;
    let b = if denom.abs() < 1e-12 {
        1.0
    } else {
        (n * sum_xy - sum_x * sum_y) / denom
    };
    let log_a = (sum_y - b * sum_x) / n;
    [log_a.exp().max(1e-8), b, c]
}

/// Gauss–Newton on weighted least squares (unit weights; σ = 0.02 in synthetic data).
fn fit_power2_gn(xy: &[(f64, f64)], max_iter: usize) -> [f64; 3] {
    let mut p = start_power2(xy);
    let inv_var = 1.0 / (0.02_f64 * 0.02);
    for _ in 0..max_iter {
        let mut jtj = [[0.0; 3]; 3];
        let mut jtr = [0.0; 3];
        for &(x, y) in xy {
            let (mu, da, db, dc) = sspower_eval(p[0], p[1], p[2], x);
            if !mu.is_finite() {
                continue;
            }
            let r = y - mu;
            let g = [da, db, dc];
            for i in 0..3 {
                jtr[i] += inv_var * g[i] * r;
                for j in 0..3 {
                    jtj[i][j] += inv_var * g[i] * g[j];
                }
            }
        }
        let delta = solve_3x3(jtj, jtr);
        if delta.iter().all(|d| d.abs() < 1e-10) {
            break;
        }
        for i in 0..3 {
            p[i] += delta[i];
        }
        if p[0] <= 0.0 {
            p[0] = 1e-6;
        }
    }
    p
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];
    for col in 0..3 {
        let mut pivot = col;
        for row in (col + 1)..3 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        m.swap(col, pivot);
        let div = m[col][col];
        if div.abs() < 1e-14 {
            return [0.0; 3];
        }
        for j in col..4 {
            m[col][j] /= div;
        }
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            for j in col..4 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}

fn synthetic_sensor_curve(sensor: usize, n_pts: usize) -> Vec<(f64, f64)> {
    let c_offset = (sensor as f64) * 0.02;
    (0..n_pts)
        .map(|i| {
            let t = i as f64 + 1.0;
            let x = 0.5 + t * 0.15;
            let noise = ((sensor * 17 + i * 31) % 100) as f64 / 100.0 - 0.5;
            let y = power2(x, A_TRUE, B_TRUE, C_TRUE + c_offset) + 0.02 * noise;
            (x, y)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    // Independent batch NLS scales with sensor count; pooled nlmer cost grows quickly
    // with the number of random-effect levels — keep N modest for a quick demo run.
    let n_sensors = 25usize;
    let n_pts = 30usize;

    let curves: Vec<Vec<(f64, f64)>> = (0..n_sensors)
        .map(|s| synthetic_sensor_curve(s, n_pts))
        .collect();

    // --- Lane A: independent batch NLS (CPU parallel) ---
    let t0 = Instant::now();
    let coeffs: Vec<[f64; 3]> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_sensors);
        for (idx, curve) in curves.iter().enumerate() {
            let curve = curve.clone();
            handles.push(s.spawn(move || (idx, fit_power2_gn(&curve, 25))));
        }
        let mut out = vec![[0.0; 3]; n_sensors];
        for handle in handles {
            let (idx, p) = handle.join().unwrap();
            out[idx] = p;
        }
        out
    });
    let batch_wall = t0.elapsed();

    let mean_a: f64 = coeffs.iter().map(|p| p[0]).sum::<f64>() / n_sensors as f64;
    let mean_b: f64 = coeffs.iter().map(|p| p[1]).sum::<f64>() / n_sensors as f64;
    let mean_c: f64 = coeffs.iter().map(|p| p[2]).sum::<f64>() / n_sensors as f64;

    // --- Lane B: one pooled nlmer on long data ---
    let mut sensor_id = Vec::with_capacity(n_sensors * n_pts);
    let mut x_col = Vec::with_capacity(n_sensors * n_pts);
    let mut y_col = Vec::with_capacity(n_sensors * n_pts);
    for (s, curve) in curves.iter().enumerate() {
        for &(x, y) in curve {
            sensor_id.push(format!("s{s}"));
            x_col.push(x);
            y_col.push(y);
        }
    }
    let df = DataFrame::new(vec![
        Series::new("sensor_id".into(), sensor_id).into(),
        Series::new("x".into(), x_col).into(),
        Series::new("y".into(), y_col).into(),
    ])?;

    let t1 = Instant::now();
    let pooled = nlmer(
        "y ~ SSpower(x, a, b, c) ~ c|sensor_id",
        &df,
        NlmmStart::new(),
        false,
    )?;
    let nlmer_wall = t1.elapsed();

    println!("=== batch_sspower_cpu (synthetic) ===");
    println!("sensors: {n_sensors}, points/sensor: {n_pts}");
    println!();
    println!("Lane A — independent Gauss–Newton (std::thread::scope):");
    println!("  wall: {:.2?}", batch_wall);
    println!(
        "  mean fitted (a, b, c): ({mean_a:.4}, {mean_b:.4}, {mean_c:.4})"
    );
    println!();
    println!("Lane B — pooled nlmer SSpower ~ c|sensor_id:");
    println!("  wall: {:.2?}", nlmer_wall);
    println!("  deviance: {:.4}", pooled.deviance.unwrap_or(f64::NAN));
    println!();
    println!("See docs/CALO_CALIBRATION.md for when to use each lane.");
    println!("CUDA batch fitters target Lane A at much larger N, not Lane B.");

    Ok(())
}
