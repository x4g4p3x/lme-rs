# Exploratory Codex workflow skill evaluation

These records support the shorter [bug-hunt](../../../.agents/skills/bug-hunt/SKILL.md)
and [perf-tune](../../../.agents/skills/perf-tune/SKILL.md) workflows. They describe
disposable fixtures, not lme-rs statistical-runtime performance or scientific
validation. The publication work record is
[here](../../work/2026-10-26-codex-skill-optimization.md).

## Changes and evidence

- Bug-hunt entrypoint: 5,920 to 2,574 normalized text characters, a 56.5% reduction.
- Perf-tune entrypoint: 6,368 to 2,935 characters, a 53.9% reduction.
- Supporting references and numerical/validation requirements remain intact.
- Independent review restored durable correctness regressions and expected-head
  merge enforcement before the final instructions were tested.
- Sixteen fresh native Codex task contexts exercised weighted-mean fixing,
  review-only behavior, a correct supported-domain implementation, and complete
  JSON-aggregation optimization. All accepted fixture outcomes passed.

The host configured a Sol model, but the collaboration interface did not expose an
independently attested exact runtime model ID. No model overrides were used.
There was one no-skill/original agent run per scenario, with adaptive candidate
revisions. This is an exploratory engineering comparison, not a randomized,
preregistered, cross-model effectiveness study. Baseline Codex also succeeded.

## Produced-code measurements

The frozen outputs were measured serially: five alternating-order blocks, two
warmups and eleven measured samples per block, or 55 samples per implementation.
All passed a separate 506-case differential suite. That stress suite was added
after initial discovery runs; it was not a preregistered primary endpoint.

For 6,000 records and 45 requested categories:

| Produced implementation | Median complete operation |
|---|---:|
| Original slow fixture | 58.965 ms |
| No-skill baseline | 1.709 ms |
| Original skill | 8.302 ms |
| First compact skill | 8.452 ms |
| Revised compact skill | 1.791 ms |
| Final reviewed skill | 1.706 ms |

The final produced implementation is about 4.87 times faster than the original
skill's produced implementation on this fixture, and matches the no-skill result.
These are code timings, not agent latency or a causal estimate of universal
prompt effectiveness. Repeated code samples are not repeated model trials.

The final implementation's traced Python allocation peak was approximately
1.400 MB versus 1.345 MB for the original skill's output, about 4.1% higher.
The original slow fixture peaked at 2.718 MB. These are separate tracemalloc
measurements, not process RSS. Empty/single-category adjacent medians were
effectively unchanged.

## Reproduce

Use Python 3.14 for the closest historical comparison; other versions can differ
in floating summation and timing. No third-party packages or model calls are
required. From this directory:

```sh
python -B reproduce.py --output /path/to/new/receipt.json
```

The output path must not exist: the runner preserves earlier receipts rather
than overwriting them. It checks frozen implementations against 506 differential
cases, measures three workloads, and records every sample and source/input hash.
Absolute timings are machine-dependent. This reproduces produced-code checks,
not the original native model sessions. The local attributes preserve exact
historical source bytes for the recorded hashes; do not reformat frozen sources.

## Functional boundaries and limits

Additional fixture checks covered analysis-only scope, inconclusive measurements,
migration status, a bounded Haskell/Rust framing port, a corrupt comparator,
serial/stale-review boundaries, actual two-lane review/fix/integration, and a fake
Grok transport. These checks do not validate real BioBit science or a live Grok
provider. Two Grok-negative cases are host-policy checks, not model evaluations.

The parallel reviewers independently found and fixed a supported large-integer
boundary failure. The native migration checker exercised all 65,536 valid values
and six malformed/precedence cases. Original code, unrelated edits, and Git state
were independently checked where preservation was required.

An analysis-only scratch output was overwritten by an import-triggered rerun.
The agent disclosed it; baseline sessions and reported comparison samples remain
preserved. Retain this provenance lapse when interpreting the evaluation. Critical
scientific runs require immutable evidence enforcement in their actual harness.

Live provider quality/cost, other models, total agent token/latency efficiency, and
bundled artifact workflows remain unmeasured. No global-optimality claim follows.
Machine-local configuration and memory changes are excluded from this publication.

## Records

- [Raw timing, memory, input and source identities](measurements.json).
- [Trial outcomes and review decisions](trial-outcomes.json).
- [Functional guardrails and external acceptance](guardrail-results.json).
- [Fake transport and host-policy checks](grok-transport-checks.json).
- [Per-scenario evaluation status](evaluation-plan.json).
- [Normalized maintained-source identities](instruction-identities.json).
- [Original bug-hunt instructions](original-instructions/bug-hunt/SKILL.md) and
  [original perf-tune instructions](original-instructions/perf-tune/SKILL.md).
