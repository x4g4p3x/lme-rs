# Python verification mini-project

This folder is a **self-check** for the `lme_python` bindings: it loads the same CSV/JSON fixtures as the Rust integration tests under `tests/data/`, fits LMMs and GLMMs through the Python API, and **asserts** reference numbers (R/lme4-style scalars, CBPP β/θ from `glmm_binomial.json`, nested LRTs, Satterthwaite Type III ANOVA, weighted REML, confint/simulate).

It is more than a usage demo: if these checks pass, the extension is exercising the library end-to-end with tolerances aligned to `tests/test_numerical_parity.rs` and `tests/test_glmm.rs`.

## Setup

From the repository root:

```bash
cd python
maturin develop --release
```

Use the same virtual environment where `polars` and `lme_python` are installed.

## Run (CLI)

From `python/`:

```bash
python examples/verification_project/run.py
```

## Run (pytest)

From `python/`:

```bash
pytest examples/verification_project/test_parity.py -v
```

## Layout

| File | Role |
|------|------|
| `paths.py` | Resolves repo root and `tests/data/*.csv` |
| `parity.py` | All `verify_*` checks |
| `run.py` | Prints OK/FAIL for each check |
| `test_parity.py` | Pytest wrappers |
| `conftest.py` | Puts this directory on `sys.path` for imports |
