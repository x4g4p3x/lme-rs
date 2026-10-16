# Internal AI review: capped scaling and native interruption

16 October 2026. A separate AI reviewer inspected the new capped runner, resource
guard, streamed report, control driver, completion audit, tests and protocol.
Review was read-only, without fits, builds or artifact changes. This is internal
independent scrutiny, not human statistical peer review or certification.

Before the first freeze, review identified five issues, all corrected before
generation:

- Publish the native marker atomically after flushing/fsyncing its contents.
- Query whole-job active process accounting; a zero-exit root with a live child
  must be rejected and its owned descendants terminated.
- Bind analysis/R receipts to actual commands, executable/scripts, frozen inputs,
  completed outputs and enforced thread limits. Audit them after both jobs finish.
- Reject contradictory cap receipts: active-clock elapsed, final free/disk
  measurements and retained memory events. Record actual sampling gaps.
- Freeze/index all three roles before input generation, enforce disjoint
  role-bound seed namespaces and verify their common source identity.

Reviewer reinspection found no remaining freeze blocker. Microsoft's documented
job limits apply to committed bytes and nested descendants; ordinary completion
notifications are not guaranteed. The raw OS peak may include denied allocations
on this host, so the design retains separate current-commit/charge/resident
metrics and conservative rejection rather than claiming a resident-memory quota.

Actual bank A validation then rejected a false root/native PID equality
assumption in the control audit. Preserve bank A unchanged: its scaling inputs
were never generated and its interrupted native control was not resumed. Before
bank B generation, source/protocol revision added the complete queried job PID
inventory and membership check. The reviewer checked the Windows class-3 layout,
nested-job semantics, marker engine/case/key/input binding, unchanged failed bank
and fresh replacement namespaces, finding no new blocker. Its final suggestion
to fold refreshed marker-time memory measurements/events into the guard was also
implemented and focused tests passed before bank B froze.

After the runs, independent raw-fit inspection confirmed two converged but
incorrect candidate rank-zero fits, with objective gaps `0.06063436058` and
`0.00534295505` against rank-one sample optima. The reviewer identified the
whole-column recovery threshold in `src/optimizer.rs` as a plausible mechanism;
the tiny off-diagonal survives that guard. This remains a hypothesis pending
public-API regressions and a principled repair. A zero generating slope does not
justify a rank-zero sample optimum. Both failed inferences must remain retained,
and confirmation stays blocked.

Independent AI data/oracle/count audit regenerated all 2,820 responses and matched
14,100 starts and 14,100 outcomes to retained audits/journals using read-only
immutable SQLite connections. Independent spectral active-set recomputation
found exactly two candidate misses, zero selected-R misses and 11/23 unselected
variant misses. All 564 count and secondary-moment rows plus 72 bounds in each
run agreed; eligible pairs were 939/1,879 with zero conditional discrepancies.
Review corrected a draft attribution: the R checker verifies attempted/failed/
reject/cover counts, not bias/RMSE/mean SE/width. This separate AI recomputation
provides the secondary-moment check.

Final evidence review found no delivery/provenance blocker. All eight bank B
candidate/reference/analysis/R-verification jobs satisfy the declared resource
limits, recorded sequential execution and zero-active-descendant completion.
Fit source/native identities and the analysis/R executable/script/argument,
input/output and thread-limit bindings agree with retained evidence.

The native B marker belongs to an owned child distinct from the launcher;
rep1 durable hashes match, rep0 compressed outcome/start is unchanged, and resume
adds only rep2. Bank A remains rejected and unresumed with three inputs, two starts,
one outcome and no scaling cases. Both public counterexamples match raw designs,
responses, fits, selected R results, oracles and provenance hashes.

Public original-file copies are byte-identical. The 28,801,948-byte archive has
1,508 entries; SHA/CRC and every member's retained file bytes agree. Resource
receipt copies and all nine descriptive secants independently reproduce,
including the negative R-time intercept. This accepts engineering evidence and
publication integrity, while the two material numerical counterexamples continue
to block confirmation. The repair/new-binary/fresh-pilot plan is appropriate.

See the [result report](../scientific/RANDOM_SLOPE_SCALING_V1.md)
and [next-step plan](LMM_RANDOM_SLOPE_OPTIMIZER_NEXT_STEP.md).
