# Codex workflow skill optimization delivery

## Status and context

- Created: 26 October 2026. Prospective for repository delivery; retrospective for
  the preceding disposable-fixture experiments.
- Last updated: 26 October 2026, Europe/Berlin.
- Status: implemented and validated for the instruction/evidence scope; push
  blocked by pre-existing dependency advisories in the repository preflight.
- Starting revision: `93dc06acec14671348432a1155d3efc80b62261c`.
- Scope is isolated from unrelated numerical/experimental work in the primary
  checkout. No production statistical code is included.
- Delivery target: `codex/optimize-workflow-skills`; no default-branch update or
  dependency remediation is included in this commit.

## Scope and acceptance criteria

Publish the validated shorter bug-hunt and perf-tune entrypoints together with a
portable, privacy-reviewed evidence record. Preserve the supporting references,
scientific/numerical safeguards, review-only and analysis-only scope, durable
regressions, and inspected-head merge enforcement. Keep machine-local Codex
configuration, memory notes, environment details, and unrelated work out of Git.

## Intended implementation and verification

- Copy the tested entrypoints into `.agents/skills/` without rewriting them.
- Publish anonymized measurements and frozen fixture outputs with a standalone
  reproducer. Distinguish exploratory model runs from repeated code timings and
  retain the disclosed scratch-data provenance lapse.
- Run skill validation, `task docs:check`, the portable reproducer, Python format
  checks for the added evidence runner, and privacy/diff checks. Keep normal Git
  hooks enabled and require the repository pre-push gate before remote delivery.
- Verify the delivery checkout, tracking ref, and live remote SHA after pushing.

## Material findings and decisions

The experiments precede this repository record and are not a preregistered
scientific study. Shorter instructions preserved tested outcomes, but the no-skill
baseline also succeeded. Fixture execution speed is not total agent latency or a
causal estimate of universal skill effectiveness. Detailed observations and raw
samples will be linked from the evidence directory.

## Implementation and observed validation

The two maintained entrypoints now match the tested personal versions. The
[evaluation record](../agent-workflows/2026-10-26-skill-optimization/README.md)
contains anonymized raw samples, outcomes, source identities, original instruction
snapshots, and a standalone receipt-preserving reproducer. Frozen fixture source
bytes are retained with local Git attributes. Machine-local configuration and
memory files are excluded.

Observed checks:

- Both bundled skill validations passed.
- The published reproducer passed all 506 differential cases for every frozen
  implementation; historical source hashes were independently checked.
- Ruff 0.17.0 lint and format checks passed for the publication runner.
- `task docs:check` passed: local links, 161 harness tests, examples, six doctests,
  dashboard checks and generated Rust documentation.
- Privacy/credential-pattern and diff checks passed.
- `task preflight` passed its earlier lint, toolchain, workflow and compile stages
  but failed at `cargo audit --deny warnings --ignore RUSTSEC-2024-0436`: six
  vulnerabilities and eleven denied unsoundness warnings. These are present in
  the unchanged base lockfile. Later legal/metadata stages did not run after the
  failure. The pre-push gate must remain enabled.

The failures include rustls-webpki, webpki-roots, tokio, and the transitive zstd
family. The advertised zstd fixes require a newer compatibility line, so resolving
the full gate is a separate dependency-graph task rather than an instruction-only
lockfile refresh. No advisory exception or hook bypass was added.

## Limitations and next action

Retain the scoped commit and resolve the dependency gate separately before a normal
push can succeed. Do not publish local configuration or claim that all models,
providers, or bundled skills have been optimized. The primary checkout's unrelated
28 pending paths were checked byte-for-byte unchanged.
