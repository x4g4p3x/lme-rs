# lme-python

Native Python bindings for [`lme-rs`](https://github.com/x4g4p3x/lme-rs), a Rust implementation of linear, generalized linear, and nonlinear mixed-effects models modeled after R's `lme4` workflow.

## Install

```bash
pip install lme-python
```

The latest published version is **[0.2.2](https://pypi.org/project/lme-python/0.2.2/)**. Published wheels are available for CPython 3.10 on Windows, macOS, and Linux; CI also builds and tests wheels from source on Python 3.10–3.13. The package imports as `lme_python` and uses Polars DataFrames. See the [changelog](https://github.com/x4g4p3x/lme-rs/blob/master/CHANGELOG.md) for what 0.2.2 ships.

## Quick start

```python
import lme_python
import polars as pl

data = pl.DataFrame(
    {
        "y": [10.0, 12.0, 13.0, 15.0, 9.0, 11.0, 14.0, 17.0, 8.0, 10.0, 12.0, 14.0],
        "x": [0.0, 1.0, 2.0, 3.0] * 3,
        "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
    }
)

fit = lme_python.lmer("y ~ x + (1 | group)", data=data, reml=True)
print(fit.summary())
print(fit.predict(data))
```

The bindings also expose GLMMs (`glmer`), nonlinear mixed models (`nlmer`), prediction and simulation, profile and bootstrap confidence intervals, grouped cross-validation, contrasts, Type I/II/III ANOVA, and LMM estimated marginal means with reference-grid pairwise comparisons.

## Documentation

- [Python guide](https://github.com/x4g4p3x/lme-rs/blob/master/python/PYTHON_GUIDE.md)
- [Rust and Python workflow guide](https://github.com/x4g4p3x/lme-rs/blob/master/GUIDE.md)
- [Supported workflows and limitations](https://github.com/x4g4p3x/lme-rs/blob/master/USABILITY.md)
- [Numerical comparisons](https://github.com/x4g4p3x/lme-rs/blob/master/comparisons/COMPARISONS.md)
- [Release history](https://github.com/x4g4p3x/lme-rs/blob/master/CHANGELOG.md)

## Development

Contributor setup, validation commands, and release policy live in the repository's [CONTRIBUTING.md](https://github.com/x4g4p3x/lme-rs/blob/master/CONTRIBUTING.md). The automated `task consumer:smoke` gate builds an isolated wheel, installs it into a clean environment, and runs the portable examples against that installed artifact.
