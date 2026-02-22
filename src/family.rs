//! Family/Link abstractions for Generalized Linear Mixed Models (GLMMs).
//!
//! This module implements the GLM family/link system matching R's `stats::family` interface.
//! Each family specifies a variance function, deviance residuals, and a canonical link function.
//! Each link function provides forward/inverse transformations and the derivative of the inverse.

use ndarray::Array1;

// ─── Link Trait ────────────────────────────────────────────────────────────────

/// A link function mapping between the linear predictor η and the mean μ.
///
/// Every link must provide three operations:
/// - `link_fun`: μ → η  
/// - `link_inv`: η → μ (the inverse link / mean function)
/// - `mu_eta`:   η → dμ/dη (derivative of inverse link, for IRLS weights)
pub trait GlmLink: std::fmt::Debug + Send + Sync {
    /// Map mean to linear predictor: η = g(μ)
    fn link_fun(&self, mu: &Array1<f64>) -> Array1<f64>;
    /// Map linear predictor to mean: μ = g⁻¹(η)
    fn link_inv(&self, eta: &Array1<f64>) -> Array1<f64>;
    /// Derivative of inverse link: dμ/dη
    fn mu_eta(&self, eta: &Array1<f64>) -> Array1<f64>;
    /// Name of the link for display purposes
    fn name(&self) -> &str;
}

// ─── Family Trait ──────────────────────────────────────────────────────────────

/// A GLM distribution family providing variance function and deviance residuals.
pub trait GlmFamily: std::fmt::Debug + Send + Sync {
    /// Variance function V(μ); relates variance to the mean.
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    /// Unit deviance residual contributions: d(y, μ) for each observation.
    fn dev_resid(&self, y: &Array1<f64>, mu: &Array1<f64>, wt: &Array1<f64>) -> Array1<f64>;
    /// The canonical link for this family.
    fn link(&self) -> &dyn GlmLink;
    /// Name of the family for display purposes
    fn name(&self) -> &str;
    /// Whether this family uses a dispersion parameter (sigma²).  
    /// Gaussian: true, Binomial/Poisson: false.
    fn uses_dispersion(&self) -> bool;
    /// Initialize mu from y (provides sensible starting values).
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64>;
    /// Create a new boxed clone of this family (needed for optimizer cloning).
    fn build_clone(&self) -> Box<dyn GlmFamily>;
}

// ─── Link Implementations ──────────────────────────────────────────────────────

/// Logit link: η = log(μ / (1 - μ)), canonical for Binomial.
#[derive(Debug, Clone)]
pub struct LogitLink;

impl GlmLink for LogitLink {
    fn link_fun(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            // Clamp to avoid log(0)
            let m = m.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            (m / (1.0 - m)).ln()
        })
    }

    fn link_inv(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| {
            let p = 1.0 / (1.0 + (-e).exp());
            // Clamp to valid probability range
            p.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
        })
    }

    fn mu_eta(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| {
            let p = 1.0 / (1.0 + (-e).exp());
            let d = p * (1.0 - p);
            // Clamp derivative away from zero for numerical stability
            d.max(f64::EPSILON)
        })
    }

    fn name(&self) -> &str { "logit" }
}

/// Log link: η = log(μ), canonical for Poisson.
#[derive(Debug, Clone)]
pub struct LogLink;

impl GlmLink for LogLink {
    fn link_fun(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.max(f64::EPSILON).ln())
    }

    fn link_inv(&self, eta: &Array1<f64>) -> Array1<f64> {
        // Clamp eta to avoid overflow in exp
        eta.mapv(|e| e.clamp(-30.0, 30.0).exp())
    }

    fn mu_eta(&self, eta: &Array1<f64>) -> Array1<f64> {
        // dμ/dη = exp(η) (same as link_inv for log link)
        eta.mapv(|e| e.clamp(-30.0, 30.0).exp().max(f64::EPSILON))
    }

    fn name(&self) -> &str { "log" }
}

/// Identity link: η = μ, canonical for Gaussian.
#[derive(Debug, Clone)]
pub struct IdentityLink;

impl GlmLink for IdentityLink {
    fn link_fun(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()
    }

    fn link_inv(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.clone()
    }

    fn mu_eta(&self, eta: &Array1<f64>) -> Array1<f64> {
        Array1::ones(eta.len())
    }

    fn name(&self) -> &str { "identity" }
}

// ─── Family Implementations ───────────────────────────────────────────────────

/// Binomial family with logit link (default).
#[derive(Debug)]
pub struct BinomialFamily {
    link: LogitLink,
}

impl BinomialFamily {
    pub fn new() -> Self {
        BinomialFamily { link: LogitLink }
    }
}

impl GlmFamily for BinomialFamily {
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            let m = m.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            m * (1.0 - m)
        })
    }

    fn dev_resid(&self, y: &Array1<f64>, mu: &Array1<f64>, wt: &Array1<f64>) -> Array1<f64> {
        // 2 * wt * [ y*ln(y/mu) + (1-y)*ln((1-y)/(1-mu)) ]
        // with boundary handling for y=0 and y=1
        let n = y.len();
        let mut d = Array1::zeros(n);
        for i in 0..n {
            let yi = y[i];
            let mi = mu[i].clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            let wi = wt[i];
            let mut dev_i = 0.0;
            if yi > 0.0 {
                dev_i += yi * (yi / mi).ln();
            }
            if yi < 1.0 {
                dev_i += (1.0 - yi) * ((1.0 - yi) / (1.0 - mi)).ln();
            }
            d[i] = 2.0 * wi * dev_i;
        }
        d
    }

    fn link(&self) -> &dyn GlmLink { &self.link }
    fn name(&self) -> &str { "binomial" }
    fn uses_dispersion(&self) -> bool { false }

    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        // R's binomial()$initialize: mu = (y + 0.5) / 2
        y.mapv(|yi| (yi + 0.5) / 2.0)
    }

    fn build_clone(&self) -> Box<dyn GlmFamily> {
        Box::new(BinomialFamily::new())
    }
}

/// Poisson family with log link (default).
#[derive(Debug)]
pub struct PoissonFamily {
    link: LogLink,
}

impl PoissonFamily {
    pub fn new() -> Self {
        PoissonFamily { link: LogLink }
    }
}

impl GlmFamily for PoissonFamily {
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        // V(mu) = mu for Poisson
        mu.mapv(|m| m.max(f64::EPSILON))
    }

    fn dev_resid(&self, y: &Array1<f64>, mu: &Array1<f64>, wt: &Array1<f64>) -> Array1<f64> {
        // 2 * wt * [ y*ln(y/mu) - (y - mu) ]
        let n = y.len();
        let mut d = Array1::zeros(n);
        for i in 0..n {
            let yi = y[i];
            let mi = mu[i].max(f64::EPSILON);
            let wi = wt[i];
            let mut dev_i = -(yi - mi);
            if yi > 0.0 {
                dev_i += yi * (yi / mi).ln();
            }
            d[i] = 2.0 * wi * dev_i;
        }
        d
    }

    fn link(&self) -> &dyn GlmLink { &self.link }
    fn name(&self) -> &str { "poisson" }
    fn uses_dispersion(&self) -> bool { false }

    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        // R's poisson()$initialize: mu = y + 0.1
        y.mapv(|yi| yi + 0.1)
    }

    fn build_clone(&self) -> Box<dyn GlmFamily> {
        Box::new(PoissonFamily::new())
    }
}

/// Gaussian family with identity link (default).
#[derive(Debug)]
pub struct GaussianFamily {
    link: IdentityLink,
}

impl GaussianFamily {
    pub fn new() -> Self {
        GaussianFamily { link: IdentityLink }
    }
}

impl GlmFamily for GaussianFamily {
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        Array1::ones(mu.len())
    }

    fn dev_resid(&self, y: &Array1<f64>, mu: &Array1<f64>, wt: &Array1<f64>) -> Array1<f64> {
        // wt * (y - mu)^2
        let diff = y - mu;
        wt * &(&diff * &diff)
    }

    fn link(&self) -> &dyn GlmLink { &self.link }
    fn name(&self) -> &str { "gaussian" }
    fn uses_dispersion(&self) -> bool { true }

    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.clone()
    }

    fn build_clone(&self) -> Box<dyn GlmFamily> {
        Box::new(GaussianFamily::new())
    }
}

// ─── Public Dispatch Enum ──────────────────────────────────────────────────────

/// User-facing family selector for `glmer()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Binomial response with logit link (for binary / proportion data).
    Binomial,
    /// Poisson response with log link (for count data).
    Poisson,
    /// Gaussian response with identity link (equivalent to `lmer()`).
    Gaussian,
}

impl Family {
    /// Create the concrete family implementation.
    pub fn build(&self) -> Box<dyn GlmFamily> {
        match self {
            Family::Binomial => Box::new(BinomialFamily::new()),
            Family::Poisson => Box::new(PoissonFamily::new()),
            Family::Gaussian => Box::new(GaussianFamily::new()),
        }
    }
}

impl std::fmt::Display for Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Family::Binomial => write!(f, "binomial"),
            Family::Poisson => write!(f, "poisson"),
            Family::Gaussian => write!(f, "gaussian"),
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn logit_roundtrip() {
        let mu = array![0.1, 0.5, 0.9];
        let link = LogitLink;
        let eta = link.link_fun(&mu);
        let mu2 = link.link_inv(&eta);
        for i in 0..mu.len() {
            assert!((mu[i] - mu2[i]).abs() < 1e-10, "logit roundtrip failed at {}", i);
        }
    }

    #[test]
    fn logit_known_values() {
        let link = LogitLink;
        let mu = array![0.5];
        let eta = link.link_fun(&mu);
        assert!((eta[0] - 0.0).abs() < 1e-10, "logit(0.5) should be 0");

        let eta2 = array![0.0];
        let mu2 = link.link_inv(&eta2);
        assert!((mu2[0] - 0.5).abs() < 1e-10, "logit_inv(0) should be 0.5");
    }

    #[test]
    fn logit_mu_eta_values() {
        let link = LogitLink;
        let eta = array![0.0];
        let d = link.mu_eta(&eta);
        assert!((d[0] - 0.25).abs() < 1e-10, "mu_eta at eta=0 should be 0.25");
    }

    #[test]
    fn log_roundtrip() {
        let mu = array![0.5, 1.0, 5.0];
        let link = LogLink;
        let eta = link.link_fun(&mu);
        let mu2 = link.link_inv(&eta);
        for i in 0..mu.len() {
            assert!((mu[i] - mu2[i]).abs() < 1e-10, "log roundtrip failed at {}", i);
        }
    }

    #[test]
    fn log_known_values() {
        let link = LogLink;
        let mu = array![1.0];
        let eta = link.link_fun(&mu);
        assert!((eta[0] - 0.0).abs() < 1e-10, "log(1) should be 0");

        let eta2 = array![0.0];
        let mu2 = link.link_inv(&eta2);
        assert!((mu2[0] - 1.0).abs() < 1e-10, "exp(0) should be 1");
    }

    #[test]
    fn identity_roundtrip() {
        let mu = array![-1.0, 0.0, 3.14];
        let link = IdentityLink;
        let eta = link.link_fun(&mu);
        let mu2 = link.link_inv(&eta);
        for i in 0..mu.len() {
            assert!((mu[i] - mu2[i]).abs() < 1e-15, "identity roundtrip failed at {}", i);
        }
        let d = link.mu_eta(&eta);
        for i in 0..d.len() {
            assert!((d[i] - 1.0).abs() < 1e-15, "identity mu_eta should be 1");
        }
    }

    #[test]
    fn binomial_variance() {
        let fam = BinomialFamily::new();
        let mu = array![0.5];
        let v = fam.variance(&mu);
        assert!((v[0] - 0.25).abs() < 1e-10, "binomial variance at 0.5 should be 0.25");
    }

    #[test]
    fn binomial_dev_resid_boundaries() {
        let fam = BinomialFamily::new();
        // y=mu should give zero deviance
        let y = array![0.5];
        let mu = array![0.5];
        let wt = array![1.0];
        let d = fam.dev_resid(&y, &mu, &wt);
        assert!(d[0].abs() < 1e-10, "dev_resid(y=mu) should be ~0");
    }

    #[test]
    fn poisson_variance() {
        let fam = PoissonFamily::new();
        let mu = array![3.0];
        let v = fam.variance(&mu);
        assert!((v[0] - 3.0).abs() < 1e-10, "poisson variance should equal mu");
    }

    #[test]
    fn poisson_dev_resid_zero_y() {
        let fam = PoissonFamily::new();
        let y = array![0.0];
        let mu = array![1.0];
        let wt = array![1.0];
        let d = fam.dev_resid(&y, &mu, &wt);
        // 2 * [0*ln(0/1) - (0-1)] = 2 * 1 = 2
        assert!((d[0] - 2.0).abs() < 1e-10, "poisson dev_resid(y=0,mu=1) should be 2");
    }

    #[test]
    fn gaussian_dev_resid() {
        let fam = GaussianFamily::new();
        let y = array![3.0];
        let mu = array![1.0];
        let wt = array![1.0];
        let d = fam.dev_resid(&y, &mu, &wt);
        assert!((d[0] - 4.0).abs() < 1e-10, "gaussian dev_resid(y=3,mu=1) should be 4");
    }

    #[test]
    fn family_enum_build() {
        let fam = Family::Binomial.build();
        assert_eq!(fam.name(), "binomial");
        assert!(!fam.uses_dispersion());

        let fam = Family::Poisson.build();
        assert_eq!(fam.name(), "poisson");

        let fam = Family::Gaussian.build();
        assert_eq!(fam.name(), "gaussian");
        assert!(fam.uses_dispersion());
    }
}
