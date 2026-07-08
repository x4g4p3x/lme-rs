//! Optional fit/deviance performance diagnostics (zero overhead when disabled).
//!
//! Enable with environment variable `LME_PERF_DIAG=1` before calling [`crate::lmer`] or
//! [`crate::math::LmmData::log_reml_deviance`]. After a fit, call [`take_report`] for a JSON
//! breakdown of where wall time went.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// High-level phases recorded when diagnostics are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Formula parse + design-matrix build in [`crate::lmer`].
    LmerSetup,
    /// Wilkinson formula parse in [`crate::prepare_lmer`].
    SetupFormula,
    /// Polars → X/Z design-matrix build in [`crate::prepare_lmer`].
    SetupDesignMatrix,
    /// [`crate::math::LmmData::new_weighted`] cross-products and intercept cache.
    SetupLmmData,
    /// θ optimization (grid / Nelder–Mead).
    LmerOptimize,
    /// Post-fit coefficient extraction and DataFrame assembly.
    LmerPostFit,
    /// One optimizer deviance evaluation (outer wrapper).
    DevianceEval,
    /// Blocked intercept `updateL!` + deviance tail.
    DevianceBlocked,
    /// Reset / scale Gram blocks at the start of `update_l_and_factor`.
    BlockedReset,
    /// RE rank updates and Cholesky on diagonal blocks.
    BlockedRankChol,
    /// Schur complement on RE cross blocks.
    BlockedSchurRe,
    /// Schur on `Xy` / `Xy'` blocks and `chol_xy_block`.
    BlockedSchurXy,
    /// Triangular solves on cross / `Xy` blocks.
    BlockedTrisolve,
    /// Scalar deviance assembly from factored `Xy` block.
    BlockedDevianceTail,
    /// Reused sparse LDL intercept path (nested / gated-off blocked).
    DevianceSparseLdl,
    /// Small-q dense Cholesky intercept fallback.
    DevianceDenseChol,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::LmerSetup => "lmer_setup",
            Phase::SetupFormula => "setup_formula",
            Phase::SetupDesignMatrix => "setup_design_matrix",
            Phase::SetupLmmData => "setup_lmm_data",
            Phase::LmerOptimize => "lmer_optimize",
            Phase::LmerPostFit => "lmer_post_fit",
            Phase::DevianceEval => "deviance_eval",
            Phase::DevianceBlocked => "deviance_blocked",
            Phase::BlockedReset => "blocked_reset",
            Phase::BlockedRankChol => "blocked_rank_chol",
            Phase::BlockedSchurRe => "blocked_schur_re",
            Phase::BlockedSchurXy => "blocked_schur_xy",
            Phase::BlockedTrisolve => "blocked_trisolve",
            Phase::BlockedDevianceTail => "blocked_deviance_tail",
            Phase::DevianceSparseLdl => "deviance_sparse_ldl",
            Phase::DevianceDenseChol => "deviance_dense_chol",
        }
    }
}

#[derive(Debug, Default)]
struct Accum {
    count: u64,
    total: Duration,
}

#[derive(Debug, Default)]
struct State {
    phases: HashMap<Phase, Accum>,
    deviance_evals: u64,
    kernel: Option<&'static str>,
    kernel_detail: Option<&'static str>,
    fit_wall: Option<Duration>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether `LME_PERF_DIAG` requested diagnostics for this process.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("LME_PERF_DIAG")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Clear accumulated counters (call before a timed fit).
pub fn reset() {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        *s.borrow_mut() = State::default();
    });
}

/// Record why the blocked kernel was or was not selected at cache build time.
pub fn set_kernel_detail(detail: &'static str) {
    if !enabled() {
        return;
    }
    STATE.with(|s| s.borrow_mut().kernel_detail = Some(detail));
}

/// Record which deviance kernel handled the hot path (`blocked`, `sparse_ldl`, `dense_chol`).
pub fn set_kernel(kernel: &'static str) {
    if !enabled() {
        return;
    }
    STATE.with(|s| s.borrow_mut().kernel = Some(kernel));
}

/// Total wall time for the outer fit (set by breakdown tooling).
pub fn set_fit_wall(duration: Duration) {
    if !enabled() {
        return;
    }
    STATE.with(|s| s.borrow_mut().fit_wall = Some(duration));
}

/// Count one optimizer deviance evaluation.
pub fn inc_deviance_eval() {
    if !enabled() {
        return;
    }
    STATE.with(|s| s.borrow_mut().deviance_evals += 1);
}

/// Add elapsed time to a phase bucket.
pub fn record_duration(phase: Phase, duration: Duration) {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let acc = state.phases.entry(phase).or_default();
        acc.count += 1;
        acc.total += duration;
    });
}

/// Run `f` and record its wall time under `phase` when diagnostics are enabled.
pub fn scope<R>(phase: Phase, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let out = f();
    record_duration(phase, started.elapsed());
    out
}

/// RAII helper: records elapsed time on drop.
pub struct PhaseGuard {
    phase: Phase,
    started: Option<Instant>,
}

impl PhaseGuard {
    /// Start timing `phase` (no-op when diagnostics are disabled).
    pub fn new(phase: Phase) -> Self {
        Self {
            phase,
            started: enabled().then(Instant::now),
        }
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if let Some(started) = self.started.take() {
            record_duration(self.phase, started.elapsed());
        }
    }
}

/// Per-phase timing row in a [`PerfReport`].
#[derive(Debug, Serialize)]
pub struct PhaseRow {
    /// Phase name (stable snake_case identifier).
    pub phase: &'static str,
    /// How many times this phase ran.
    pub count: u64,
    /// Total seconds across all invocations.
    pub total_seconds: f64,
    /// Mean seconds per invocation.
    pub mean_seconds: f64,
    /// Share of total `deviance_eval` time (when applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction_of_deviance: Option<f64>,
}

/// Aggregated diagnostics from the most recent instrumented fit.
#[derive(Debug, Serialize)]
pub struct PerfReport {
    /// Deviance kernel used on the hot path.
    pub kernel: Option<String>,
    /// Why blocked Cholesky was or was not installed (`blocked_active`, `blocked_unavailable`, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_detail: Option<String>,
    /// Number of `log_reml_deviance` calls during optimization (+ any post-fit eval).
    pub deviance_eval_count: u64,
    /// Mean seconds per `deviance_eval` wrapper.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_deviance_eval_seconds: Option<f64>,
    /// Phases sorted by total time (descending).
    pub phases: Vec<PhaseRow>,
    /// Outer fit wall time when set by breakdown tooling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_wall_seconds: Option<f64>,
    /// `lmer_optimize` as a fraction of `fit_wall` when both are set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize_fraction_of_fit: Option<f64>,
    /// Sum of `deviance_eval` as a fraction of `fit_wall` when both are set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deviance_fraction_of_fit: Option<f64>,
}

/// Take and clear the accumulated report (returns `None` if diagnostics were disabled or empty).
pub fn take_report() -> Option<PerfReport> {
    if !enabled() {
        return None;
    }
    STATE.with(|s| {
        let state = std::mem::take(&mut *s.borrow_mut());
        if state.deviance_evals == 0 && state.phases.is_empty() {
            return None;
        }

        let deviance_total = state
            .phases
            .get(&Phase::DevianceEval)
            .map(|a| a.total)
            .unwrap_or_default();
        let optimize_total = state
            .phases
            .get(&Phase::LmerOptimize)
            .map(|a| a.total)
            .unwrap_or_default();

        let mut phases: Vec<PhaseRow> = state
            .phases
            .iter()
            .map(|(&phase, acc)| PhaseRow {
                phase: phase.name(),
                count: acc.count,
                total_seconds: acc.total.as_secs_f64(),
                mean_seconds: if acc.count > 0 {
                    acc.total.as_secs_f64() / acc.count as f64
                } else {
                    0.0
                },
                fraction_of_deviance: if deviance_total > Duration::ZERO {
                    Some(acc.total.as_secs_f64() / deviance_total.as_secs_f64())
                } else {
                    None
                },
            })
            .collect();
        phases.sort_by(|a, b| {
            b.total_seconds
                .partial_cmp(&a.total_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let fit_wall = state.fit_wall;
        Some(PerfReport {
            kernel: state.kernel.map(str::to_string),
            kernel_detail: state.kernel_detail.map(str::to_string),
            deviance_eval_count: state.deviance_evals,
            mean_deviance_eval_seconds: if state.deviance_evals > 0 {
                Some(deviance_total.as_secs_f64() / state.deviance_evals as f64)
            } else {
                None
            },
            phases,
            fit_wall_seconds: fit_wall.map(|d| d.as_secs_f64()),
            optimize_fraction_of_fit: fit_wall
                .filter(|w| *w > Duration::ZERO)
                .map(|w| optimize_total.as_secs_f64() / w.as_secs_f64()),
            deviance_fraction_of_fit: fit_wall
                .filter(|w| *w > Duration::ZERO)
                .map(|w| deviance_total.as_secs_f64() / w.as_secs_f64()),
        })
    })
}
