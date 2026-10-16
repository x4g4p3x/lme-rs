# Internal implementation review for random-slope pilot v2

Prepared before observations, 16 October 2026. Status: **reviewed for freeze**.
This is internal AI review, not human statistical peer review or calibration proof.

V2 is an intentionally separate version of the frozen v1 runner and R adapter.
The original source files stay byte-identical so v1 remains verifiable. The main
changes are protocol/engine identity, tolerance 1e-12, prospectively included
balanced global-optimum/precision/rank diagnostics for both engines, and removal
of the inaccurate Windows launcher memory measurement. Design, gates, failure
handling, R controls, distributions and uncertainty rules remain unchanged.

Before freeze, review disjoint deterministic seeds, native/source/wheel identity,
unchanged gates, no retries/substitutions, missing/invalid optimum diagnostics,
candidate payload integrity, all-attempt denominators, and the limits of the
balanced oracle and selected-input control evidence. Run targeted v2 tests plus
the existing scientific controls, script lint, required R/Julia formatting and
documentation checks.

## Review findings resolved before freeze

The independent AI reviewer verified the selected-input controls against original
hashes, derived dense analytic REML scores/Hessians and checked transformation
curvature. This supports prospective 1e-12/5,000 controls while retaining the
1e-14 nonconvergence. It is not a new fitting oracle or calibration result.

Review of the v2 fork confirmed unchanged designs/gates, disjoint v2 seed prefixes,
native/source identity checks, both-engine balanced diagnostics and no outcome
substitution. Two reporter gaps were corrected before observations: independently
recompute paired disagreements from aligned raw outcomes (marginal totals alone
do not determine them), and reject missing/inconsistent balanced metric or rank
inventories. Dedicated negative controls cover both. Final independent review
identified no remaining pre-freeze blocker.

Six v2 controls and six dense-derivative/integrity controls passed. The full
89-test scientific/script suite passed with the installed R available; script
lint and required R/Julia formatting passed. Documentation checks passed before
the final reporting additions; a fresh link check covers those additions, reusing
unchanged passing examples, doctests and API documentation. No engine code changed;
the preceding Rust/Basin/golden/fair validation remains applicable. The isolated
wheel rebuilt at the exact engine commit matches the recorded boundary-fix native.

Freeze this source/review/protocol set before smoke or pilot generation. Any
later correction requires a separately identified freeze and fresh namespace;
completed observations must be retained. Outcomes belong in a separate report.
