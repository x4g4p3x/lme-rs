# Fresh covariance-precision repair pilot v1

Declared before inputs, 16 October 2026. Exploratory numerical validation only;
no release, confirmation, or nominal-rate claim. Preserve the three original
stopping-control failures and aborted analysis. Diagnostic replays do not replace
those outcomes. Follow the [repair record](work/2026-10-16-covariance-precision-recovery.md).

Use the unchanged 94-case registry, 20 replicates per case and four fixed-order
R variants on every input: 1,880 candidates and 7,520 reference fits. Controls
remain REML, absolute simplex cost-SD tolerance `1e-10`, 5,000 shared outer
iterations, default starts and explicit convergence eligibility. No retries,
replacement fits, adaptive starts/controls or outcome extensions.

Identify a newly built isolated wheel. Record the actual base commit, dirty
engine status, compiler and 64 engine-input hashes before building; verify them
again before freeze. Match installed native bytes to the wheel and reject reuse
of the predecessor's native/wheel. Keep Python/Polars/platform/package version
compatible with that predecessor. Study source identity is separate from the
pre-build receipt; never label uncommitted source as the base commit's engine.

Use a separate successor runner/reporter/controller. The historical sources and
their archived copies remain unchanged. At predecessor verification, only current
engine files and `tests/test_model_contracts.rs` may differ; verify every historical
archived source hash, and reject changes to old execution, analysis, protocol and
numerical helpers. Freeze the successor's current source/test/protocol/engine
inventory after independent AI source review and before generating responses.
The new study records every prior namespace/manifest hash and accepts only a
disjoint `covariance-precision-v1-...` namespace. Verify source copies, controls,
receipts and native identity throughout execution; reject output overlap/reuse.

Explicitly use `lmm_resource_caps_stable.py`. Its disk scan tolerates children that
vanish after enumeration, but rejects denied subtrees, missing roots and other I/O
failures. Do not patch the historical sampler. Exercise resource/store controls
and retain any errors. Pin the local import closure used by the cap controls,
reconstruct command identities from the original interpreter, fixed program and
disk target, and bind plan/start/end/log bytes. Revalidate raw expected aborts
and nonzero exits before accepting those controls. Memory enforcement requires
an allocation-denial marker with its expected exit or a native limit event;
OS peak charge alone is insufficient. Copy and hash the control evidence into
the prospective study. Keep append-only WAL/FULL journals with saved inputs and
durable starts, failed/interrupted attempts and resumption of never-started keys.

Run candidate, R bank, Python analysis and independent R check sequentially in
separate owned Windows jobs, four fit workers and one numerical thread each.
Unchanged limits: 900 seconds per phase, 4 GiB aggregate job-committed memory,
512 MiB sampled study disk, 1 GiB free reserve; poll 0.1 seconds plus sampling
cost. Retain actual gaps, OS job charges, PIDs, exits, limits/events and command/
input/output bindings. Build, environment, archive and parent process are outside
measured jobs. Fail closed without a supported hard-limit backend. Do not run
builds/tests/benchmarks concurrently with the measured phases or delete evidence
to meet a cap.

Keep all prior numerical/method/reference gates and reference selection order.
Check 1,840 balanced constrained optima and covariance ranks; the 40 imbalanced
inputs have no balanced oracle. Preserve known-variance GLS and inference/p/CI/
count checks on all inputs, all-attempt denominators, 564 binary-count rows and
72 simultaneous exact-bound rules. Secondary moments stay descriptive. The
small n cannot establish nominal calibration even when numerical checks pass.

Any candidate or selected-reference numerical/method failure, paired numerical
or decision discrepancy blocks numerical readiness. Unselected R failures are
retained, counted and prospectively nongating. Engineering acceptance remains
separate. Preserve failed gates; diagnose them before proposing another fresh
study. Require independent AI source, result and publication review, disclose
raw-evidence availability and exact identities. AI review is not human peer
review. Only a passing reviewed fresh pilot can reopen conditional production
preparation; full campaign authorization and release remain separate.
