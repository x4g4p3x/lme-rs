//! Parametric response simulation from fitted models.
//!
//! Draws new response vectors from fitted conditional means (and residual dispersion
//! for Gaussian models). This is **not** R's full `bootMer` loop (resample → refit);
//! use [`crate::boot_lmer`] for parametric/residual bootstrap refits.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Bernoulli, Gamma, Poisson, StandardNormal};
use rayon::prelude::*;

use crate::LmeFit;
use crate::SimulateResult;

/// Simulate `nsim` response vectors, optionally in parallel.
///
/// * `n_jobs` — `None` uses all logical CPUs (capped at `nsim`); `Some(1)` is sequential.
/// * `seed` — when set, draw `i` uses `seed.wrapping_add(i)` so results are reproducible
///   regardless of `n_jobs`.
pub fn simulate_fit(
    fit: &LmeFit,
    nsim: usize,
    n_jobs: Option<usize>,
    seed: Option<u64>,
) -> anyhow::Result<SimulateResult> {
    simulate_range(fit, 0, nsim, n_jobs, seed).map(|simulations| SimulateResult { simulations })
}

/// Simulate `count` draws starting at global index `start_index`.
pub fn simulate_range(
    fit: &LmeFit,
    start_index: usize,
    count: usize,
    n_jobs: Option<usize>,
    seed: Option<u64>,
) -> anyhow::Result<Vec<Array1<f64>>> {
    if matches!(n_jobs, Some(0)) {
        return Err(anyhow::anyhow!("simulate requires n_jobs >= 1"));
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    let workers = resolve_n_jobs(n_jobs, count);
    if workers == 1 {
        simulate_sequential(fit, start_index, count, seed)
    } else {
        simulate_parallel(fit, start_index, count, workers, seed)
    }
}

/// Invoke `on_batch` for each chunk of at most `batch_size` simulations without
/// materializing all draws at once.
pub fn simulate_batched<F>(
    fit: &LmeFit,
    nsim: usize,
    batch_size: usize,
    n_jobs: Option<usize>,
    seed: Option<u64>,
    mut on_batch: F,
) -> anyhow::Result<()>
where
    F: FnMut(usize, &[Array1<f64>]) -> anyhow::Result<()>,
{
    if batch_size == 0 {
        return Err(anyhow::anyhow!("simulate_batch requires batch_size >= 1"));
    }
    if matches!(n_jobs, Some(0)) {
        return Err(anyhow::anyhow!("simulate requires n_jobs >= 1"));
    }

    let mut offset = 0usize;
    let mut batch_idx = 0usize;
    while offset < nsim {
        let count = (nsim - offset).min(batch_size);
        let batch = simulate_range(fit, offset, count, n_jobs, seed)?;
        on_batch(batch_idx, &batch)?;
        offset += count;
        batch_idx += 1;
    }
    Ok(())
}

fn simulate_sequential(
    fit: &LmeFit,
    start_index: usize,
    count: usize,
    seed: Option<u64>,
) -> anyhow::Result<Vec<Array1<f64>>> {
    let sigma2 = fit.sigma2.unwrap_or(1.0);
    let mut out = Vec::with_capacity(count);

    if let Some(base) = seed {
        for i in 0..count {
            let mut rng = StdRng::seed_from_u64(base.wrapping_add((start_index + i) as u64));
            out.push(draw_one(&fit.fitted, fit.family, sigma2, &mut rng)?);
        }
    } else {
        let mut rng = rand::rng();
        for _ in 0..count {
            out.push(draw_one(&fit.fitted, fit.family, sigma2, &mut rng)?);
        }
    }

    Ok(out)
}

fn simulate_parallel(
    fit: &LmeFit,
    start_index: usize,
    count: usize,
    workers: usize,
    seed: Option<u64>,
) -> anyhow::Result<Vec<Array1<f64>>> {
    let fitted = fit.fitted.clone();
    let family = fit.family;
    let sigma2 = fit.sigma2.unwrap_or(1.0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| anyhow::anyhow!("simulate failed to build thread pool: {e}"))?;

    pool.install(|| {
        (0..count)
            .into_par_iter()
            .map(|i| {
                let global = start_index + i;
                let mut rng = match seed {
                    Some(base) => StdRng::seed_from_u64(base.wrapping_add(global as u64)),
                    None => StdRng::from_os_rng(),
                };
                draw_one(&fitted, family, sigma2, &mut rng)
            })
            .collect()
    })
}

fn draw_one<R: Rng + ?Sized>(
    fitted: &Array1<f64>,
    family: Option<crate::family::Family>,
    sigma2: f64,
    rng: &mut R,
) -> anyhow::Result<Array1<f64>> {
    let n = fitted.len();
    let sigma = sigma2.sqrt();
    let mut y_sim = fitted.to_owned();

    match family {
        None | Some(crate::family::Family::Gaussian) => {
            for i in 0..n {
                let eps: f64 = rng.sample(StandardNormal);
                y_sim[i] += sigma * eps;
            }
        }
        Some(crate::family::Family::Binomial) => {
            for i in 0..n {
                let p = y_sim[i].clamp(f64::EPSILON, 1.0 - f64::EPSILON);
                let bern = Bernoulli::new(p)
                    .map_err(|e| anyhow::anyhow!("Invalid binomial probability: {e}"))?;
                y_sim[i] = if rng.sample(bern) { 1.0 } else { 0.0 };
            }
        }
        Some(crate::family::Family::Poisson) => {
            for i in 0..n {
                let lambda = y_sim[i].max(f64::EPSILON);
                let pois = Poisson::new(lambda)
                    .map_err(|e| anyhow::anyhow!("Invalid Poisson mean: {e}"))?;
                y_sim[i] = rng.sample(pois);
            }
        }
        Some(crate::family::Family::Gamma) => {
            let dispersion = sigma2.max(f64::EPSILON);
            let shape = (1.0 / dispersion).max(f64::EPSILON);
            for i in 0..n {
                let mu = y_sim[i].max(f64::EPSILON);
                let scale = (mu * dispersion).max(f64::EPSILON);
                let gamma = Gamma::new(shape, scale)
                    .map_err(|e| anyhow::anyhow!("Invalid Gamma parameters: {e}"))?;
                y_sim[i] = rng.sample(gamma);
            }
        }
    }

    Ok(y_sim)
}

fn resolve_n_jobs(n_jobs: Option<usize>, n_tasks: usize) -> usize {
    let requested = n_jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    requested.max(1).min(n_tasks.max(1))
}
