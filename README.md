# lme-rs - Rust implementation of R's lme4

`lme-rs` aims to provide a fast, production-grade Rust implementation of R's `lme4` with the same statistical behavior and numerics used by `lmer()` and `lm()`.

## Goal

**100% numerical match to `lme4::lmer()` for validation against R**

This project prioritizes numerical parity first, performance second, and API ergonomics third.

## Installation

```bash
cargo add lme-rs
```

Or in `Cargo.toml`:

```toml
[dependencies]
lme-rs = "0.1.0"
```

## Quick Example (mtcars-style workflow)

```rust
use ndarray::array;
use lme_rs::lm;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Toy "mtcars-like" example: mpg ~ 1 + wt
    let y = array![21.0, 21.0, 22.8, 21.4, 18.7];
    let x = array![
        [1.0, 2.620],
        [1.0, 2.875],
        [1.0, 2.320],
        [1.0, 3.215],
        [1.0, 3.440],
    ];

    let fit = lm(&y, &x)?;
    println!("coefficients = {:?}", fit.coefficients);
    Ok(())
}
```

For mixed models, the target interface is:

```rust,ignore
let fit = lme_rs::lmer("Reaction ~ Days + (Days | Subject)", &data)?;
```

## Roadmap

- MVP: OLS `lm` (dense QR path, deterministic outputs)
- Full LMM: `lmer` random effects, REML/ML, `VarCorr`, ranef extraction
- GLMM: `glmer` family/link support with robust convergence and diagnostics

## Validation

- Tested against R 4.4+ test suite
- Numerical diffs tracked against `lme4` reference outputs
- Reproducibility checks for deterministic fitting paths
