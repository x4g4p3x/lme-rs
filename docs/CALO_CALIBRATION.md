# Sensor calibration with nonlinear mixed models

[Documentation](README.md) · [Rust nonlinear models](../GUIDE.md#nlmer-for-nonlinear-mixed-models) · [Python nonlinear models](../python/PYTHON_GUIDE.md#nonlinear-mixed-models)

Choose the statistical model before choosing a parallel or GPU implementation.
Independent sensor curves and a pooled mixed model estimate different things.

## Two different problems

| Requirement | Model | Starting point |
|:------------|:------|:---------------|
| Separate coefficients for every sensor | Independent nonlinear least squares | [CPU batch demonstration](../examples/batch_sspower_cpu.rs) or a dedicated curve-fitting tool |
| Shared population curve with sensor deviations | Nonlinear mixed-effects model | `nlmer` with `SSpower` |

The built-in power mean is:

```text
mean(x) = a * x^b + c
```

It has the same functional form as MATLAB's `power2`; this does not establish
identical estimation or bounds behavior across tools.
The `SSpower` implementation requires **x > 0**.

For pooled sensor offsets, use:

```text
y ~ SSpower(x, a, b, c) ~ c|sensor_id
```

Here, `a`, `b`, and `c` are population parameters; each sensor has a random
deviation on `c`. This is one joint fit, not an independent curve per sensor.

## Starts and bounds

The Rust API uses [NlmerOptions](../src/nlmm/fit.rs);
Python exposes corresponding keyword arguments.

| Option | Meaning |
|:-------|:--------|
| `start` | Named starting values for population parameters; omitted/empty values use the built-in self-start path |
| `lower`, `upper` | Bounds on population coefficients |
| `group_lower`, `group_upper` | Bounds on each group's `β + b` for parameters present in the random-effects formula |
| `n_agq` | Laplace at 1; adaptive quadrature where the grid size permits |

Group-level bounds are distinct from population bounds. For the formula above,
group bounds can constrain `c`; they do not turn the pooled model into
independent bounded fits of all three parameters.
See [bounds tests](../tests/test_nlmm_bounds.rs) and the
[Rust recipe](../GUIDE.md#nlmer-for-nonlinear-mixed-models).

## Evidence and limitations

The `sspower_synthetic_self_start` case in
[the golden manifest](../tests/data/golden_parity_manifest.json) compares the
implemented mean and fit against a custom R `selfStart` reference in
[nlmm_sspower.R](../comparisons/nlmm_sspower.R).

That fixture is not a general MATLAB compatibility test and does not establish
parity for every bound configuration. Additional tests cover
[SSpower behavior](../tests/test_nlmm_sspower.rs) and
[parameter bounds](../tests/test_nlmm_bounds.rs).

The nonlinear API uses one grouping factor. Robust-loss fitting and independent
per-sensor fitting inside `nlmer` are outside its scope. Inspect convergence,
starts, parameter identifiability, and prediction behavior on your data.

## CPU batch demo

From the repository root:

```bash
cargo run --release --locked --example batch_sspower_cpu
```

The example compares a small independent Gauss–Newton batch solver with a
pooled `nlmer` call on synthetic data. Read it as a demonstration of two
different workflows. Its timings do not compare equivalent statistical models,
and the independent solver is not the core mixed-model API.

## When to investigate GPU fitting

Independent curves can provide parallel work across sensors. A pooled mixed
model couples groups through shared parameters and covariance estimation.
A speedup from an independent GPU batch fitter cannot be transferred directly
to `nlmer`.

If throughput is a requirement, first measure the intended model, batch size,
data transfer, and end-to-end cost. A separate accelerator integration needs
its own numerical validation and maintenance. No GPU backend is provided by
the current `lme-rs` fitting API.

## Related documentation

- [Workflow assessment](../USABILITY.md): where nonlinear modeling fits in the supported scope.
- [Numerical comparisons](../comparisons/COMPARISONS.md): reference methodology and outputs.
- [Benchmarks](../BENCHMARKS.md): how to make scoped timing claims.
- [Optimization notes](../OPTIMIZATION.md): existing CPU fitting architecture.
