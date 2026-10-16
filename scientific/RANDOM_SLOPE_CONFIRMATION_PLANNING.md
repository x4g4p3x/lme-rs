# Random-slope confirmation planning results

16 October 2026. **Planning and selected post-study diagnostics only. No new
confirmation campaign, package release or public raw-archive upload.**
The [proposal](../docs/LMM_RANDOM_SLOPE_CONFIRMATION_PROPOSAL.md) defines the
scientific decision and the next implementation milestone. The completed
[v2 pilot](RANDOM_SLOPE_PILOT_V2.md) and earlier studies remain unchanged.

## Design recommendation

Propose **30,000 replicates per primary case**, with candidate-only Satterthwaite
and Kenward–Roger claims across 36 balanced interior null cases (M=72).
Use 2,000 replicates in the other 58 cases for exploration and a predetermined
first-1,000-per-case R subset. This is **1,196,000 candidate datasets** and
**94,000 reference datasets**; two methods share each primary dataset.

The [inputs](RANDOM_SLOPE_CONFIRMATION_PLANNING_INPUTS.json) declared a .05
familywise error rate, [.04,.06] rejection band, >=.90 global decision-power
target at ideal p=.05/no failures, and the tested replication/sensitivity grid
before calculation. The 30,000 recommendation was made **after** inspecting
that grid to allow more failure margin; it is not a prospectively frozen campaign.

## Exact count and decision-power calculations

The [machine-readable calculation](RANDOM_SLOPE_CONFIRMATION_PLANNING.json)
uses 144 Bonferroni one-sided Clopper–Pearson bounds. With n attempts, m failures
and K observed evaluable rejections, bound the observable lower event
`evaluable AND reject` using K and upper event `unevaluable OR reject` using K+m.
This simultaneously covers the failure-completion envelope without a
missing-at-random assumption. All 72 envelopes must lie within [.04,.06].

Planning conditions on fixed m and hypothetical evaluable rejection probability
p_valid. Compute single-claim failure q by exact binomial tails and global power
lower bound `max(0,1-72*q)`. Independence between claims is not assumed.

| n per primary case | Complete-data acceptance counts | Joint power lower bound, p=.05 and m=0 |
|:---|:---|---:|
| 20,000 | 897–1,087 | .795976278 |
| 21,000 | 939–1,144 | .875708887 |
| 22,000 | 981–1,201 | .924261523 |
| 22,500 | 1,002–1,230 | .943606326 |
| 25,000 | 1,108–1,373 | .984365364 |
| 30,000 | 1,318–1,661 | .999061075 |

**22,000 is the smallest passing tested budget**, not a global integer minimum.
Discreteness can make power nonmonotone in n. The old 48-claim, 20,000-replicate
calculation cannot be reused for this 72-claim family.

| n | No failures | Fixed .01% failures | Fixed .1% failures |
|:---|---:|---:|---:|
| 22,000 | .924261523 | .901156408 (m=3) | .508076586 (m=22) |
| 25,000 | .984365364 | .979199816 (m=3) | .848818317 (m=25) |
| 30,000 | .999061075 | .998717189 (m=3) | .982000016 (m=30) |

Missing counts use decimal ceiling rounding. At 30,000 and .5% or 1% failures,
the joint union-bound lower bound is zero. At no failures and p=.045/.055,
single-claim powers are only .8171/.6159 and the joint lower bound is zero.
Zero here is an uninformative conservative bound, not measured zero joint power.
Near .04/.06 boundaries, certification power is intrinsically small. The chosen
budget cannot promise 90% power uniformly over the acceptance band. The pilot's
20 observations per case do not estimate the true probabilities precisely enough
to replace these explicit assumptions.

Independent [R verification](RANDOM_SLOPE_CONFIRMATION_PLANNING_R_CHECK.json)
checks **432 rows** (12 ideal plus 420 sensitivity) using beta-quantile count
searches versus the planner's binomial-tail searches. Every count interval agrees;
maximum absolute probability discrepancy **2.21e-13**. Small rational binomial
tests also validate counts, missing-completion power, boundary intervals,
empty acceptance ranges and the non-independent union bound.

## Separately retained R optimizer diagnostic

Selected all nine v2 R fits whose objectives exceeded the balanced oracle gate.
Before refitting, wrote the [selection/source receipt](RANDOM_SLOPE_REFERENCE_CONTROLS_RECEIPT.json),
exact inputs and a fixed four-variant bank. The default, tight-default and two
signed fixed starts use the same REML model, data and locked R environment.
Every one of **36 attempts** is retained, including suboptimal converged fits.
The [diagnostic ledger](RANDOM_SLOPE_REFERENCE_CONTROLS.json) contains full fits
and independent numerical checks. No v2 record is replaced.

| Variant | Objective/covariance/rank checks passed |
|:---|---:|
| Original baseline | 0/9 |
| Tighter default start | 3/9 |
| Tighter positive off-diagonal start | 9/9 |
| Tighter negative off-diagonal start | 8/9 |
| Smallest-criterion converged valid fit | 9/9 |

Baseline beta, sigma2 and criterion reproduce the original nine records exactly.
Selected maximum absolute objective difference from the global oracle is
**1.71e-13**. All 27 selected R method results are evaluable; independent review
also verified all **153 candidate/selected-reference numerical comparisons**
against the unchanged v2 gates. This selected recovery does not estimate future
reliability or prove inference calibration. The future general reference runner
needs covariance/finite/convergence eligibility independent of the balanced-only
oracle, including imbalanced cases.

An initial diagnostic retained the same 36 numerical results. Formatting changed
its Python source layout after receipt capture; the exact executed source was
recovered and archived with the receipt's matching hash. The final diagnostic was
run separately after adding fixed-order tie regression coverage. Both original
and final receipts/attempts are preserved; no hidden replacement or deletion.

## Resource provision

Use actual v2 four-worker, one-thread phase wall times, apportioned by measured
case-specific fit costs, and each case's compressed input/candidate and paired
outcome sizes. Primary cases averaged 14.5% more candidate time than the
pilot-wide mixture; a simple global-average proxy was therefore replaced.
The final calculation verifies retained input/outcome hashes before using them.

The proposed R bank has four attempts per reference dataset. The table assumes
**four times the baseline R cost/record size**, then doubles phase-wall and raw
storage estimates as a provisional provision. This multiplier is an assumption,
not a measured bound on tighter/multistart fits. Sizes use decimal GB.

| 30,000-replicate scenario | Candidate datasets | Reference datasets | Four-variant phase-wall proxy | Twofold time provision | Twofold raw storage proxy |
|:---|---:|---:|---:|---:|---:|
| Primary only | 1,080,000 | 36,000 | 8.95 h | 17.90 h | 6.71 GB |
| Primary + 2,000 secondary, proposed | 1,196,000 | 94,000 | 11.73 h | 23.46 h | 8.42 GB |
| Every case at 30,000, R subset | 2,820,000 | 94,000 | 20.81 h | 41.62 h | 16.48 GB |
| Every case at 30,000, all paired | 2,820,000 | 2,820,000 | 121.69 h | 243.38 h | 69.98 GB |

This excludes numerical audit, analysis/bootstrap, fixed environment artifacts,
archive copies and their storage overhead. Paired records contain a duplicate
candidate payload. Compression, startup, scheduler balance and long-run costs may
change. Native peak memory was not measured in v2. A small streaming rehearsal
must measure actual worker memory and total wall/disk cost before fixing the
campaign resource cap. These estimates are not a fair engine-speed benchmark.

## Evidence, validation and next action

Reproduce the planner with optional SciPy (used here: Python 3.11.15/SciPy 1.16.3),
passing the checked-in inputs and a **new** output path to
`scripts/plan_lmm_random_slope_confirmation.py`. The R checker is
`comparisons/scientific/lmm_random_slope_planning.R`; diagnostic entry points are
`scripts/diagnose_lmm_random_slope_reference.py` and its separate R adapter.
These new files do not modify frozen pilot sources or production dependencies.

Local raw archive: `benchmark-results/scientific/random-slope-pilot/lmm-random-slope-confirmation-planning-20261016.zip`.
The [bundle descriptor](RANDOM_SLOPE_CONFIRMATION_PLANNING_BUNDLE.json) records
**140,185 bytes, 25 entries**, SHA/CRC verification and source-layout provenance.
It is prepared locally, **not uploaded**. Exact source/input/environment hashes
match final receipts; historical draft calculations and executed sources remain
retained separately. Raw diagnostic reproduction also requires the retained v2
inputs; neither archive is currently a public download.

Validation: all **98 script tests passed**, including both R-dependent cases;
the nine new focused planner/reference-policy tests passed. Script lint/format
and required R/Julia comparison formatting passed. Documentation validation and
the [internal AI review](../docs/LMM_RANDOM_SLOPE_CONFIRMATION_PLANNING_REVIEW.md)
are recorded in the [living work record](../docs/work/2026-10-16-random-slope-confirmation-planning.md).
No Rust/binding source changed; no full CI, full campaign or release ran here.

Next: implement and validate the streaming execution/analysis/resume contract,
run a separate fresh resource rehearsal, review and freeze actual sources and
protocol, then obtain authorization for the full campaign. The proposal's
failure, numerical and resource gates must be concrete and tested first.
