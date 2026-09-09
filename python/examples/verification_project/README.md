# Python verification project

[Example catalog](../../../docs/EXAMPLES.md) · [Python guide](../../PYTHON_GUIDE.md)

This project fits models through `lme_python` and asserts reference values from
[the repository fixtures](../../../tests/data/). It covers LMMs and GLMMs,
weighted REML, nested likelihood-ratio tests, Satterthwaite Type III ANOVA,
intervals, and simulation.

These are end-to-end regression checks of the bindings. Agreement applies to
the fixtures and tolerances used here; it is not a general parity guarantee.

## Setup

From the repository root:

```bash
mise install
cd python
uv sync --extra dev --no-install-project
uv run --no-sync maturin develop --release
```

Use `--no-sync` after the build so environment synchronization does not remove
the extension. See [contributor setup](../../../CONTRIBUTING.md#python-bindings).

## Run

From **`python/`**, choose either the summary runner or pytest:

```bash
uv run --no-sync python examples/verification_project/run.py
uv run --no-sync pytest examples/verification_project/test_parity.py -v
```

The runner prints an outcome for each verification. Inspect failures against
the corresponding fixture and fit options; do not relax tolerances solely
to make a changed fit pass.

For clean installed-wheel coverage, run `task consumer:smoke` from the
repository root. It also runs this project's summary runner.

## Source map

| File | Role |
|:-----|:-----|
| [paths.py](paths.py) | Resolves repository and fixture paths |
| [parity.py](parity.py) | Verification functions and tolerances |
| [run.py](run.py) | Summary runner |
| [test_parity.py](test_parity.py) | Pytest wrappers |
| [conftest.py](conftest.py) | Local test imports |
