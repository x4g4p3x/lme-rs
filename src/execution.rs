//! Explicit parallel execution without process-wide BLAS/OpenMP environment changes.
use crate::{LmeError, Result};

/// A reusable Rayon pool owned by the caller. Backend thread settings stay caller-owned.
pub struct ExecutionContext {
    pool: rayon::ThreadPool,
}

impl ExecutionContext {
    /// Create a reusable context with exactly `workers` worker threads.
    pub fn new(workers: usize) -> Result<Self> {
        if workers == 0 {
            return Err(LmeError::InvalidInput {
                message: "workers must be positive".into(),
            });
        }
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|e| LmeError::InvalidInput {
                    message: format!("cannot create worker pool: {e}"),
                })?,
        })
    }
    /// Run work in this context; nested operations use this pool when requested.
    pub fn install<R: Send>(&self, work: impl FnOnce() -> R + Send) -> R {
        self.pool.install(work)
    }
    /// Number of workers in this context.
    pub fn workers(&self) -> usize {
        self.pool.current_num_threads()
    }
}

pub(crate) fn resolve_workers(requested: Option<usize>, tasks: usize) -> usize {
    requested
        .unwrap_or_else(rayon::current_num_threads)
        .max(1)
        .min(tasks.max(1))
}

pub(crate) fn run<R: Send>(workers: Option<usize>, work: impl FnOnce() -> R + Send) -> Result<R> {
    match workers {
        Some(n) => Ok(ExecutionContext::new(n)?.install(work)),
        None => Ok(work()),
    }
}
