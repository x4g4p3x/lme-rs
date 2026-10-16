# Internal AI review: random-slope confirmation planning

16 October 2026. A separate AI reviewer examined calculations, estimands,
selected reference diagnostics and resource arithmetic without changing the
candidate or running fits. This is internal AI scrutiny, not a human statistician's
endorsement or a certification of the library.

## Findings

- Observable Bernoulli endpoints `E AND reject` and `not E OR reject` support
  the 144 one-sided CP bounds and simultaneous all-completion certification.
  Missing-at-random is unnecessary for the actual primary decision.
- Conditional `Bin(n-m,p_valid)` is an explicit planning scenario. The union-bound
  joint power calculation avoids an independence assumption between claims.
- Standalone R beta/binomial calculations agree with every sensitivity/ideal
  count interval and probability; maximum probability difference 2.21e-13.
  22,000 is the smallest passing tested ideal budget. The 30,000/.1%-failure
  scenario has joint power lower bound .982000016, conditional on p_valid=.05.
- Initial resource arithmetic used the whole pilot's mixture. The review found
  primary cases cost more; final estimates now use each case's measured costs.
  Decimal ceiling rounding and explicit fixed-order tie regression coverage were
  also added before final validation.
- All 36 selected R attempts are retained, and nine baseline fits reproduce
  retained beta, sigma2 and criterion exactly. Objective-only selection passes
  all nine independent global objective/covariance/rank controls, with maximum
  objective discrepancy 1.71e-13. All 153 candidate/selected-reference numerical
  comparisons meet unchanged gates. This is selected post-study evidence only.
- Current and historical executed sources, final input/environment/result
  identities and all eight weighted resource scenarios were independently checked.
  The local archive's SHA, 25 entries and CRC agree with its descriptor.

## Remaining gates

No planning-mathematics blocker remains. The proposal is not executable confirmation:
streaming persistence, attempted-outcome accounting, crash/resume integrity,
bounded analysis, generic reference eligibility (including imbalanced cases),
fresh seeds, measured resource rehearsal and the final freeze still need
implementation and independent review. Numerical checks must not select reference
results by p-value agreement or silently discard candidate failures.

No confirmation data or library release was generated. Preserve original study
records and report unsupported claims; selected recovery and ideal power are not
proof of future numerical reliability or scientific calibration.
