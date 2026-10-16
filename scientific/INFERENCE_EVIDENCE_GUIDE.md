# What the current inference evidence supports

The completed study applies to **published lme-python/lme-rs 0.2.7**, Gaussian
random-intercept models `y ~ x + (1 | group)`, REML, the specified fixed designs,
independent Gaussian errors, no weights and no offsets. It does not validate
the current development binary, every dataset with the same formula, or other
random-effects structures.

Read the [confirmatory results](CALIBRATION_CONFIRMATORY_0.2.7.md) together with
the [audit and corrections](BOUNDARY_AUDIT_0.2.7.md). The
[case/method ledger](LMM_EVIDENCE_0.2.7.json) records the precise conditions and
statuses; `descriptive_only` rows were not primary validation claims.

| Question | Evidence and limit |
|:---|:---|
| Do implementations agree? | Candidate and R rejection/coverage decisions matched on 79,994 common-evaluable records from the 80,000-record audit subsample. This is implementation agreement, not proof of nominal calibration. |
| Does Satterthwaite pass the entire declared scope? | No: 23 supported cases, one inconclusive case; the original all-24-case gate is false. |
| Does KR pass the entire declared scope? | No: 21 supported cases, two inconclusive cases and one conservative contradicted case; its gate is false. |
| What performed well in the finite tested set? | Both methods supported all 12 primary within-group cases, all six between-group cases at generating rho=0.7, and three additional moderate/large-group low-rho cases. This does not establish an untested continuous region. |
| Where is caution needed? | Small-group, low-rho, between-group designs; see the exact casewise rates and uncertainty. Generating rho is known to the simulator, not to an ordinary data analyst. |
| Is normal-Wald inference interchangeable with adjusted inference? | No: it exceeded the nominal rejection rate substantially in several small-group cases, reaching about 12%. It remains a descriptive comparator in this study. |
| Are singular fits automatically invalid? | No: finite evaluable singular fits were retained; the numerical threshold was abs(theta)<1e-4. A mathematical zero variance and this threshold classification are distinct. |
| Are boundary/imbalanced/alternative results confirmatory passes? | Those strata were descriptive under this protocol. Their findings remain visible, but do not enlarge its primary scope. |

The 4%–6% rejection and 94%–96% coverage bands are declared engineering criteria.
They are not an external certification standard. The original 48-claim family
and simultaneous exact binomial failure envelopes remain fixed. Selecting the
supported rows after seeing results does not make the original global gates pass.

There is no validated automatic method-switching rule based on group count,
estimated ICC or singularity. Choosing a method after looking at data creates
a new procedure whose full behavior needs its own study. This milestone does
not change inference defaults or present observed simulation rho as a usable
runtime assurance rule.

The new exact boundary diagnostics also exposed a custom-control scalar search
defect in the published package, fixed in development. Default-control study
records are preserved. A new release needs evidence for its own binary before
inheriting any statistical claims. The next prospective extension is the
[random-slope pilot protocol](../docs/LMM_RANDOM_SLOPE_PILOT_PROTOCOL.md), which
has not been executed by this audit.
