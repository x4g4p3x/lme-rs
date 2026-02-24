# lme-rs - Rust implementation of R's lme4

`lme-rs` is a fast, robust, production-grade Rust implementation of R's `lme4` regression suite, providing the same statistical behavior and numerics used by `lmer()` and `lm()`.

## Core Philosophy

**Core Philosophy**: `lme-rs` provides 100% numerical parity with base `lme4` for fitting core models, while integrating the most essential downstream diagnostics (like `lmerTest` p-values and `car` ANOVAs) directly into a single, high-performance API.

## Capabilities

After extensive porting and architectural alignment with `RcppEigen` structures, `lme-rs` currently supports:

- **R-style Wilkinson Formulas**: Native parsing of mixed-effect bounds via `fiasto` natively supporting formulas like `Reaction ~ Days + (Days | Subject) + (1 | Item)`.
- **Sparse Matrix Performance**: Native implementations of Compressed Sparse Column/Row (CSC/CSR) $Z$ boundaries evaluated against `sprs-ldl` Sparse Cholesky Decompositions to completely prevent `O(N^2)` matrix blowouts.
- **Complex Random Effects**: Full native resolution of independent slopes, intercepts, and **crossed** random effects mapped to multi-dimensional optimization spaces.
- **Advanced Optimizers**: Employs robust derivative-free N-Dimensional solvers (`argmin::Nelder-Mead`) to iterate relative covariance matrices ($\Lambda_\theta$).
- **Maximum Likelihood Toggles**: Computes pure evaluations over Restricted Maximum Likelihood (`REML=TRUE`) and strict Maximum Likelihood boundaries (`REML=FALSE`), accurately tracking explicit log-determinant $L_x$ variance penalties.
- **Diagnostics & Formatting**: Safely exposes structured `std::fmt::Display` diagnostics mirroring the comprehensive output format of `summary(model)` in R (Fixed Effects, Variance Components, Scaled Residuals, Standard Errors).
- **Out-of-Bag Forecasting**: Leverages internal model parameters out to a `.predict()` method forecasting expectations against new input DataFrames mathematically evaluating population-level distributions.

## Installation

```bash
cargo add lme-rs
```

Or in `Cargo.toml`:

```toml
[dependencies]
lme-rs = "0.1.0"
```

## Quick Example (lmer workflow)

You can easily instantiate and resolve a Linear Mixed-Effects model out of a `Polars` DataFrame exactly like you would inside the R Console:

```rust
use polars::prelude::*;
use lme_rs::lmer;

fn main() -> anyhow::Result<()> {
    // 1. Load your Polars DataFrame
    let mut file = std::fs::File::open("sleepstudy.csv")?;
    let df = CsvReadOptions::default().with_has_header(true).into_reader_with_file_handle(&mut file).finish()?;

    // 2. Define Wilkinson formula and Evaluate
    let formula = "Reaction ~ Days + (Days | Subject)";
    
    // Evaluate standard REML model
    let fit = lmer(formula, &df, true)?;

    // 3. Print R-like Console Summary
    println!("{}", fit);

    // 4. Generate Population-Level Predictions
    let new_days = Series::new("Days".into(), &[0.0, 1.0, 5.0, 10.0]);
    let new_subject = Series::new("Subject".into(), &["308", "308", "308", "308"]);
    let newdata = DataFrame::new(vec![new_days.into(), new_subject.into()])?;
    
    let preds = fit.predict(&newdata)?;
    println!("Predictions: {:?}", preds);

    Ok(())
}
```

## Testing & Validation

`lme-rs` is subjected to heavy numeric integration parity testing against `lme4`. The test suite strictly loads raw regression matrix outputs formally constrained directly from R to ensure precision accuracy across parameters:

- Deviance objective matching
- Variance component bounds stabilization ($\theta$)
- Matrix Cholesky deterministic factorizations ($A$ blocks)
- Fixed Effect Coefficient estimates ($\beta$)

## Roadmap

- Core LMM Architecture implementation - **Complete**
- GLMM: `glmer` family/link support mapping (Binomial / Poisson link bounds) - **Complete**
- Optimizer: Theta bound enforcement (diagonal ≥ 0, matching `lme4`) - **Complete**
- Gamma family + additional link functions (Probit, Cloglog, Inverse, Sqrt) - **Complete**
- GLMM predict on response scale (`predict_response` / `predict_conditional_response`) - **Complete**
- `anova()`: Likelihood ratio tests for comparing nested models - **Complete**
- Observation weights (`lmer_weighted`) for prior weights on observations - **Complete**
- `confint()`: Wald confidence intervals for fixed effects - **Complete**
- `simulate()`: Parametric bootstrap from fitted models - **Complete**
- Nested random effects: `(1|a/b)` → `(1|a) + (1|a:b)` expansion - **Complete**
- Satterthwaite approximate degrees of freedom and p-values for fixed effects - **Complete**
- Kenward-Roger degrees of freedom and p-values (`pbkrtest` / `lmerTest`) - **In Progress**
- Type II and Type III ANOVA tables (`car` package) - **Planned**
- Robust Standard Errors - **Planned**
- Predicting with new levels (`allow.new.levels`) - **Planned**
