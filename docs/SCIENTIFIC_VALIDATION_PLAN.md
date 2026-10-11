# Scientific validation plan

Draft concept, assessed 11 October 2026 against `b347bf9d69f2713cf75ae271c98c8311b86734e9`
(`0.2.7-dev.0`). This is a proposed research programme, not completed validation
or a certification.

The initial deterministic contracts and reproduction runner are now described
in the [frozen baseline](../scientific/README.md). The
[calibration protocol](LMM_CALIBRATION_PROTOCOL.md) remains a review-ready draft.
The inventory below records the starting assessment; completed evidence is
reported separately by version and scope.

## Objective and sequence

Establish which mathematical computations and statistical procedures are reliable,
under which model conditions, with reproducible evidence and explicit limitations.
Finite testing cannot prove correctness for every possible input.

Release 0.2.7 first under the existing release gates. Then align the Python and
MCP/SCP interfaces to that released engine and freeze the initial validation
baseline. Later fixes require a new version and a rerun of affected evidence;
do not silently change the baseline during a study. A release does not itself
establish the scientific claims proposed here.

Separate three questions throughout the programme:

1. Does the implementation calculate the specified mathematical quantity accurately?
2. Does the statistical method have useful operating properties under stated assumptions?
3. Can the library reproduce a published analysis with the same data and choices?

A correct implementation can reproduce an approximation that has poor small-sample
coverage. Reproducing an analysis also does not establish its causal interpretation
or validate the original scientific conclusion independently.

## Evidence already present

| Evidence inspected | Current scope | Limit |
|:---|:---|:---|
| [Golden manifest](../tests/data/golden_parity_manifest.json) | 24 cases: 6 LMM, 8 GLMM, 10 NLMM; R version, package versions, reference calls and per-output tolerances | Selected examples; some tolerances are broad. GLMM likelihood levels are deliberately excluded |
| [Mathematical identities](../tests/test_statistical_identities.rs) | OLS orthogonality, prediction identities, response decomposition, variance/domain checks | Mainly deterministic checks on a few datasets |
| [Modeling reference](../tests/data/modeling_reference.json) and [generator](../scripts/validation/generate_modeling_reference.R) | Project-owned synthetic data; independent black-box R ANOVA, intervals, contrasts and marginal means | References depend on correct model alignment and documented provenance |
| [Cross-language verifier](../scripts/verify_cross_language_parity.py) | Four named cases, comparing fixed-effect coefficients | Not a complete covariance, likelihood or inference audit; Rust and its Python binding share an engine |
| [Workflow comparison](MODEL_WORKFLOW_BENCHMARK.md) | Matched statsmodels random-intercept workflows, common simulated responses, inference/refit failure gates | Small bootstrap counts; explicitly not long-run false-positive calibration |
| [Fuzzing](../fuzz/README.md) | Formula parsing and design construction | Crash/error behaviour rather than statistical accuracy |
| [Provenance](../PROVENANCE.md) | Independent implementation policy, test-data origins and distribution boundaries | Each additional study needs its own data/code permissions |
| Recent correctness regressions | Units, interactions, splines/polynomials, inference, ANOVA, reference means and nesting | Useful counterexamples, not exhaustive evidence |

The recent local CI results were 137 unit tests, 399 integration tests and 6
doctests; each native/wheel/ABI3 Python pass had 89 passes and one optional skip.
These are execution counts, not a statistical-validity score. The repository
[completion map](../REPO_COMPLETION_BY_AREA.md) measures locked implementation
commitments, not scientific confidence.

The [completion manifest](../completion_manifest.json) explicitly leaves broad
variance-component interval references and the full R formula-edge matrix open.
NLMM multi-group and dedicated post-fit inference limitations remain declared.
No systematic ADEMP calibration study or NIST campaign was located in the
inspected tests, comparison scripts and documentation. This assessment did not
rerun every historical comparison or certify the freshness of every fixture.

Local sibling manifests inspected during this assessment pin `lme-rs-agent`
and `lme-rs-mcp` to engine 0.2.6. Update and test the dependency chain after 0.2.7
is published. Record the actual engine resolved by each executable and wheel;
the wrapper's own version is insufficient.

## Scientific frameworks to follow

These are complementary frameworks and reference resources, not interchangeable
certifications of a numerical library.

- [NIST Statistical Reference Datasets](https://www.nist.gov/itl/sed/products-services/statistical-reference-data-sets-strd)
  provide reference values for linear regression, ANOVA, nonlinear regression
  and univariate statistics. Use the supported portions as external accuracy
  challenges. Their nonlinear solutions are best-available references, rather
  than exact algebraic truths.
- [Morris, White and Crowther, 2019](https://doi.org/10.1002/sim.8086)
  ([open manuscript](https://arxiv.org/html/1712.03198v3)) provide ADEMP:
  Aims, Data-generating mechanisms, Estimands, Methods and Performance measures.
  Use it to specify and report simulation studies, including Monte Carlo uncertainty.
- [FAIR principles](https://doi.org/10.1038/sdata.2016.18) guide identifiers,
  metadata, provenance and reusable artefacts. They are guiding principles,
  not a numerical-accuracy standard.
- [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
  are a useful eventual external software-review target for testing, documentation
  and reproducible installation. Such review complements the mathematical study.

## Work package 1: contracts and evidence registry

Create a separate scientific validation registry without changing the locked
completion-score scopes. For every public statistic, specify the estimand,
formula, assumptions, parameterization and supported inputs. Map it to an
independent reference, identity, calibration experiment and known limitation.

Each experiment record should contain:

- Case identifier, research question, paper DOI/table and source-data identifier.
- Data checksum, licence, row-selection/preprocessing script, missing-value policy.
- Formula/design matrices, factor order/contrasts, response family/link, weights
  and their meaning, offsets, ML/REML and quadrature choices.
- Engine revision/version, wrapper and binary hashes, compiler/OS/BLAS,
  reference package versions, optimizer settings, starts and seeds/random states.
- Outputs and their units, predefined tolerances/acceptance rules, warnings,
  convergence status, failures, skips and raw logs.
- Independent reviewer, result status and the exact scope of any resulting claim.

Keep development cases separate from a held-out validation set. Once a held-out
failure drives a fix, retain it as a regression and replenish independent cases.

## Work package 2: mathematical and numerical verification

Start with OLS, weighted least squares and small Gaussian mixed models, where
the expected quantities can be derived independently. Use exact/simple cases,
high-precision arithmetic and a small dense implementation based on the
marginal covariance `V = Z G Z' + R`, separate from the production solver.

Check coefficients, covariance, residual variance, BLUPs, ML/REML objectives and
stationarity or boundary optimality conditions. Compare design column spaces,
estimable hypotheses and fitted values when valid factor or spline bases differ.
Do not require raw coefficients or Cholesky parameters to agree across different
parameterizations; align them or compare the implied covariance and predictions.

Add NIST Longley/Wampler linear challenges and suitable ANOVA datasets. Map
nonlinear NIST cases only to supported mathematical subproblems: a nonlinear
least-squares optimum is not automatically a mixed-model optimum.

Extend existing unit-change checks to row permutations, factor relabeling,
equivalent parameterizations, zero-variance reductions, affine response changes
where applicable and weight transformations under the declared weight contract.
Retain negative controls for singular/non-estimable designs and invalid domains.

For GLMMs, compare tiny random-intercept cases with independent numerical
integration; assess Laplace/AGQ approximation separately from optimizer error.
For nonlinear means and inference derivatives, compare analytic expressions or
independent automatic/high-precision differentiation with the implementation.

Set accuracy criteria before running candidate results. Use scaled residuals,
relative/absolute tolerances and conditioning; assess objective accuracy as well
as parameter accuracy. An optimizer difference alone does not justify a broad
tolerance. Near-singular problems can have unstable parameters with stable
predictions; record that distinction explicitly.

## Work package 3: matched software comparisons

Expand R references through `lme4`, `lmerTest`, `pbkrtest`, `car`, `emmeans` and
the appropriate robust-covariance packages. Add Julia `MixedModels.jl` and
statsmodels only for models and methods that genuinely match. Independent
implementations reduce reliance on one oracle; agreement is still not proof.

Compare more than fixed effects: covariance matrices, variance components,
population/conditional predictions, likelihood conventions, degrees of freedom,
test statistics, confidence limits, contrasts and marginal means. Use the same
observations, settings and random response vectors in every engine.

Document GLMM likelihood constants and approximation differences. Align only
identified, parameter-independent constants when justified; do not dismiss every
likelihood discrepancy as a convention. Extend random-slope and GLMM
variance-component interval references, sparse/unbalanced cases, unusual links,
weights/offset combinations and nonlinear starts.

## Work package 4: statistical calibration

Predefine an ADEMP protocol and obtain independent statistical review before
the confirmatory run. Begin with Gaussian random-intercept/slope models. Vary
group counts, observations per group, balance, variance components, correlations,
fixed-effect sizes and predictor scales. Include crossed/nested structures and
near-boundary cases in staged extensions.

Use independently generated datasets with known parameters, rather than relying
solely on the library's own simulator. Feed exactly the same datasets to all
engines. Validate the generator separately.

Measure bias, RMSE, empirical versus reported standard errors, 95% interval
coverage, false-positive rates under the null, power under alternatives and
failure/nonconvergence frequency. Evaluate scalar and joint hypotheses. Separate
correct-model scenarios from heteroscedastic/non-normal stress scenarios; the
latter assess robustness under misspecification, not universal exactness.

Run a small pilot to debug the harness and estimate cost. Choose final repetition
counts from precision requirements. For an estimated binary proportion `p`,
`MCSE = sqrt(p * (1-p) / R)`: at `p = 0.05` and `R = 10,000`, this is about
0.00218, or 0.218 percentage points. This is a planning example, not a universal
minimum. Bootstrap inference has additional within-dataset Monte Carlo error;
choose and report that simulation budget separately.

Declare practical calibration bands before seeing results. Report confidence
intervals for performance estimates and paired differences between methods.
Inspect every scenario rather than averaging away a bad small-sample stratum.
Do not force approximate methods to have exact nominal coverage in all settings.

Keep all attempted datasets in the failure denominator. Report accuracy/coverage
among successful fits with the explicit denominator, and analyse whether failures
select difficult datasets. An unevaluable fit is not automatically a non-rejection
or a covered interval. Predefine retry policies and report original failures too.

## Work package 5: published-result reproduction

| Candidate | What to reproduce | Access checked |
|:---|:---|:---|
| [Bates et al., 2015](https://www.jstatsoft.org/article/view/v067i01) | LMM fitting, profiled criterion and worked analyses | Open paper; [R replication code](https://www.jstatsoft.org/article/download/v067i01/3163) retrieved |
| [Kuznetsova et al., 2017](https://www.jstatsoft.org/article/view/v082i13) | Satterthwaite, ANOVA I-III and supported inference examples | Open paper; [R replication code](https://www.jstatsoft.org/article/download/v082i13/3598) retrieved |
| [Halekoh and Højsgaard, 2014](https://www.jstatsoft.org/article/view/v059i09) | KR and null parametric-bootstrap comparisons | Open paper; [R example code](https://www.jstatsoft.org/article/download/v059i09/2831) retrieved; the linked simulation archive still needs retrieval verification |
| [Luke, 2017](https://doi.org/10.3758/s13428-016-0809-y) | The reported small-sample significance-calibration patterns under the original design | Public abstract and appendix inspected; full methods and simulation artefacts still need verification |

These are candidates, not completed reproductions. Run the authors' original
code first, retain its outputs, then independently translate the specified
analysis to the supported library API. Freeze preprocessing and hypotheses.
For simulation papers, compare results within Monte Carlo uncertainty rather
than expecting identical plotted percentages from different random draws.

Add two or three applied studies with accessible raw data, analysis scripts,
clear model specifications and compatible model structures. Select them before
seeing lme-rs results and record excluded candidates/reasons. Data availability
and inspectable methods matter more than citation count. Do not substitute an
easier model for an unsupported published one and call it reproduction.

## Interfaces, MCP and literature access

Rust is the mathematical engine. Python and MCP/SCP need additional tests of
conversion, factor levels, options, defaults, errors and provenance against the
same native calls. Comparing these interfaces is useful integration evidence,
not an independent mathematical oracle.

[Codex supports local STDIO and remote HTTP MCP servers](https://developers.openai.com/codex/mcp).
Use MCP for literature discovery, DOI/citation management, authorised data
retrieval and calling versioned experiment runners. A plugin-catalogue search
found Consensus and Scite available but not installed during this assessment;
their research output still needs checking against the original papers.

Public metadata APIs such as [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
and [OpenAlex](https://help.openalex.org/api/) can support a research adapter;
API availability does not imply an already installed MCP server or full-text access.
An optional local experiment MCP could expose validate-case, run-comparison,
run-calibration and export-evidence operations with bounded arguments and immutable
case records. The underlying scripts must also run without an AI assistant.

Institutional subscriptions can help obtain original methods and supplementary
material. Verify each provider's access, redistribution, automated-retrieval and
AI-processing conditions with the library before connecting licensed full text.
For example, [Elsevier distinguishes API credentials from content entitlement](https://dev.elsevier.com/tecdoc_api_authentication.html).
Prefer openly licensed data/code for publicly reproducible tests. Access to a
paper does not imply access to its raw data or permission to publish its contents.

## Initial delivery and evidence gates

1. Release 0.2.7 under [RELEASING.md](../RELEASING.md), align wrappers, and capture
   exact binary/engine provenance. Existing release validation remains required.
2. Inventory current references, justify tolerances and define the independent
   numerical contracts. Deliver a small analytic/NIST suite and an initial
   three-paper reproduction set.
3. Review and freeze the LMM calibration protocol; run the pilot, then the
   precision-planned confirmatory study on a held-out seed/case set.
4. Extend to supported GLMM/NLMM shapes and difficult inference, then obtain an
   independent statistician/numerical analyst review and a clean-environment rerun.
5. Archive versioned protocol, source identifiers, permitted data, generators,
   raw results, failures and report. Publish per-method evidence and limitations.

Fast identities and compact fixtures belong in ordinary CI. Larger simulations,
reference regeneration and reproduction campaigns belong in scheduled/manual
jobs and version-specific release assessment. A failed case remains visible;
it either blocks the applicable claim or leads to a declared limitation.

The defensible claim is that a named version passed specified numerical,
reference and calibration studies under defined conditions. Avoid a blanket
claim that all mathematics is proven correct or that a coverage percentage is
scientific certification.
