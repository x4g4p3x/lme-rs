---
name: perf-tune
description: Measure bottlenecks and improve performance while preserving correctness. Use for optimization, latency, throughput, allocation or memory investigations; preserve analysis-only scope.
---

# Performance Tuning

Improve the requested complete workload with reproducible evidence. A faster kernel alone does not establish a faster application. Run directly; this skill does not request delegation.

1. Inspect Git state and relevant changes; preserve unrelated work. Read applicable AGENTS.md and performance contracts, including [the lme-rs guide](references/lme-rs.md) when applicable. Analysis-only requests produce measurements and recommendations without production edits.
2. Define the metric, representative workload, timing boundary and acceptance. Measure the baseline before edits. Preserve source/dirty-state, binary, input/seed, toolchain, build, hardware and thread identities plus commands and raw samples. Profile separately when instrumentation changes throughput; distinguish preparation, kernel, IO and reuse.
3. State the measured bottleneck, proposed mechanism and invariants. Change the cost supported by evidence. Preserve outputs, precision, convergence, failure behavior, memory/shape gates and supported inputs. Do not skip work or loosen validation. For caches, test the complete validity key, changed inputs/controls, lifetime and concurrency. Use independent correctness checks, including adverse inputs and fallback paths. Retain meaningful regression tests when correctness-sensitive behavior changes.
4. Compare equivalent complete operations with the same inputs, settings, build and resource limits. Separate startup, cold work and prepared reuse. Serialize timings; warm up, retain repeated samples and alternate order or repeat sessions when drift matters. Follow the repository's acceptance/sampling rules; inspect variability, not only a best run or median. Keep slow and failed cases visible. Missing runtimes are skips; noise is inconclusive.
5. After the first improvement, check whether the dominant cost moved. If substantial repeated work remains, evaluate a better algorithm within scope and verify its compatibility and memory tradeoffs; a measured speedup alone is not a stopping condition. Test adjacent workloads and report absolute results, units, sample counts, variability, tradeoffs and the measured scope. Revise or remove failed speculative edits while preserving unrelated work. Complete repository-required tests, changelog and evidence records, review the diff, and report failures/skips honestly. Avoid wall-clock assertions in ordinary unit tests.

Publication requires existing user authorization. Keep hooks enabled, inspect hook edits, and verify local/tracking/live SHAs after a push. This skill does not authorize optimizer-default changes, delegation, external benchmark publication, releases or scope reduction to earn completion credit.
