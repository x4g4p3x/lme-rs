//! Nonlinear mean evaluation trait (built-ins and user-defined).

use std::sync::Arc;

use crate::nlmm::fit::NlmmStart;
use crate::nlmm::formula::NlmmMeanKind;
use crate::nlmm::self_start;
use crate::nlmm::ssasymp::ssasymp_eval;
use crate::nlmm::ssgompertz::ssgompertz_eval;
use crate::nlmm::sslogis::sslogis_eval;
use crate::nlmm::ssmicmen::ssmicmen_eval;
use ndarray::Array1;

type NlmmMeanClosure = dyn Fn(f64, &[f64]) -> (f64, Vec<f64>) + Send + Sync;

/// Evaluate μ and ∂μ/∂(each parameter) at effective parameter values (RE offsets applied).
pub trait NlmmMeanEval: Send + Sync + std::fmt::Debug {
    /// Number of fixed nonlinear parameters.
    fn n_params(&self) -> usize;
    /// Mean and parameter gradients at `(x, params)`.
    fn eval(&self, x: f64, params: &[f64]) -> (f64, Vec<f64>);
    /// Static default starting values by parameter name.
    fn default_start_values(&self, names: &[String]) -> Vec<f64>;
    /// Data-driven starting values (`selfStart` style).
    fn self_start_values(&self, x: &Array1<f64>, y: &Array1<f64>, names: &[String]) -> NlmmStart;
    /// Use RSS-based σ² profiling for scalar-RE fits (`SSasymp` / `SSfol`).
    fn uses_scalar_rss_sigma(&self) -> bool {
        false
    }
    /// Keep `scal > 0` during optimization (`SSlogis`).
    fn needs_positive_scal(&self) -> bool {
        false
    }
}

impl NlmmMeanEval for NlmmMeanKind {
    fn n_params(&self) -> usize {
        NlmmMeanKind::n_params(*self)
    }

    fn eval(&self, x: f64, params: &[f64]) -> (f64, Vec<f64>) {
        match self {
            NlmmMeanKind::Sslogis => {
                let (mu, da, dx, ds) = sslogis_eval(params[0], params[1], params[2], x);
                (mu, vec![da, dx, ds])
            }
            NlmmMeanKind::Ssasymp | NlmmMeanKind::Ssfol => {
                let (mu, da, dr, dl) = ssasymp_eval(params[0], params[1], params[2], x);
                (mu, vec![da, dr, dl])
            }
            NlmmMeanKind::Ssmicmen => {
                let (mu, dv, dk) = ssmicmen_eval(params[0], params[1], x);
                (mu, vec![dv, dk])
            }
            NlmmMeanKind::Ssgompertz => {
                let (mu, grads) = ssgompertz_eval(params[0], params[1], params[2], x);
                (mu, grads)
            }
        }
    }

    fn default_start_values(&self, names: &[String]) -> Vec<f64> {
        crate::nlmm::mean::default_start(*self, names)
    }

    fn self_start_values(&self, x: &Array1<f64>, y: &Array1<f64>, names: &[String]) -> NlmmStart {
        self_start::self_start(*self, x, y, names)
    }

    fn uses_scalar_rss_sigma(&self) -> bool {
        NlmmMeanKind::uses_scalar_rss_sigma(*self)
    }

    fn needs_positive_scal(&self) -> bool {
        matches!(self, NlmmMeanKind::Sslogis)
    }
}

/// Wrap a built-in mean kind for fitting / prediction.
pub fn builtin_mean(kind: NlmmMeanKind) -> Arc<dyn NlmmMeanEval> {
    Arc::new(kind)
}

/// User-defined nonlinear mean from a Rust closure.
#[derive(Clone)]
pub struct CustomNlmmMean {
    n_params: usize,
    eval_fn: Arc<NlmmMeanClosure>,
}

impl CustomNlmmMean {
    /// Build a custom mean from a closure returning `(μ, ∂μ/∂params)`.
    pub fn new<F>(n_params: usize, eval_fn: F) -> Self
    where
        F: Fn(f64, &[f64]) -> (f64, Vec<f64>) + Send + Sync + 'static,
    {
        Self {
            n_params,
            eval_fn: Arc::new(eval_fn),
        }
    }

    /// Wrap as `Arc<dyn NlmmMeanEval>` for fitting.
    pub fn into_arc(self) -> Arc<dyn NlmmMeanEval> {
        Arc::new(self)
    }
}

impl std::fmt::Debug for CustomNlmmMean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomNlmmMean")
            .field("n_params", &self.n_params)
            .finish()
    }
}

impl NlmmMeanEval for CustomNlmmMean {
    fn n_params(&self) -> usize {
        self.n_params
    }

    fn eval(&self, x: f64, params: &[f64]) -> (f64, Vec<f64>) {
        (self.eval_fn)(x, params)
    }

    fn default_start_values(&self, names: &[String]) -> Vec<f64> {
        names.iter().map(|_| 1.0).collect()
    }

    fn self_start_values(&self, _x: &Array1<f64>, _y: &Array1<f64>, names: &[String]) -> NlmmStart {
        self.default_start_values(names)
            .into_iter()
            .zip(names.iter())
            .map(|(v, n)| (n.clone(), v))
            .collect()
    }
}

pub(crate) fn eval_mean_with_re(
    mean: &dyn NlmmMeanEval,
    x: f64,
    params: &[f64],
    re_param_indices: &[usize],
    re_offsets: &[f64],
) -> (f64, Vec<f64>) {
    let mut effective = params.to_vec();
    for (idx, off) in re_param_indices.iter().zip(re_offsets.iter()) {
        effective[*idx] += *off;
    }
    mean.eval(x, &effective)
}
