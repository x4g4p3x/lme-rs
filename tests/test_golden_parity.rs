use lme_rs::anova::DdfMethod;
use lme_rs::family::{Family, Link};
use lme_rs::{glmer_weighted_with_link, lmer, LmeFit};
use ndarray::Array1;
use polars::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct GoldenManifest {
    schema_version: u32,
    reference_environment: ReferenceEnvironment,
    cases: Vec<GoldenCase>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ReferenceEnvironment {
    engine: String,
    generator: String,
    notes: Vec<String>,
    packages: HashMap<String, Option<String>>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct GoldenCase {
    id: String,
    description: String,
    kind: String,
    data_path: String,
    formula: String,
    #[serde(default)]
    reml: Option<bool>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    link: Option<String>,
    #[serde(default)]
    weights_column: Option<String>,
    #[serde(default)]
    nlmm_start: Option<HashMap<String, f64>>,
    #[serde(default)]
    nlmm_reml: Option<bool>,
    #[serde(default)]
    n_agq: Option<usize>,
    reference: ReferenceSpec,
    #[serde(default)]
    post_fit: Option<PostFitSpec>,
    expected: ExpectedSpec,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ReferenceSpec {
    engine: String,
    call: String,
    source_fixture: String,
}

#[derive(Debug, Deserialize, Default)]
struct PostFitSpec {
    #[serde(default)]
    robust_cluster: Option<String>,
    #[serde(default)]
    satterthwaite: bool,
    #[serde(default)]
    kenward_roger: bool,
}

#[derive(Debug, Deserialize, Default)]
struct ExpectedSpec {
    #[serde(default)]
    coefficients: Vec<ScalarCheck>,
    #[serde(default)]
    theta: Vec<ScalarCheck>,
    #[serde(default)]
    sigma2: Option<ScalarCheck>,
    #[serde(default)]
    deviance: Option<ScalarCheck>,
    #[serde(default)]
    beta_se: Vec<ScalarCheck>,
    #[serde(default)]
    population_predictions: Vec<PredictionCheck>,
    #[serde(default)]
    conditional_predictions: Vec<PredictionCheck>,
    #[serde(default)]
    satterthwaite_anova: Vec<AnovaCheck>,
    #[serde(default)]
    kenward_roger_anova: Vec<AnovaCheck>,
    #[serde(default)]
    robust_se: Vec<ScalarCheck>,
    #[serde(default)]
    robust_t: Vec<ScalarCheck>,
    #[serde(default)]
    finite_deviance: bool,
}

#[derive(Debug, Deserialize)]
struct ScalarCheck {
    name: String,
    value: f64,
    tolerance: f64,
}

#[derive(Debug, Deserialize)]
struct PredictionCheck {
    name: String,
    newdata: HashMap<String, Value>,
    values: Vec<f64>,
    tolerance: f64,
    #[serde(default)]
    allow_new_levels: bool,
}

#[derive(Debug, Deserialize)]
struct AnovaCheck {
    term: String,
    f_value: ScalarCheck,
    p_value: ScalarCheck,
    num_df: ScalarCheck,
    den_df: ScalarCheck,
}

fn load_manifest() -> GoldenManifest {
    let file = File::open("tests/data/golden_parity_manifest.json")
        .expect("tests/data/golden_parity_manifest.json not found");
    serde_json::from_reader(file).expect("failed to parse golden parity manifest")
}

fn read_csv_data(path: &str) -> DataFrame {
    let mut file = File::open(path).unwrap_or_else(|_| panic!("could not open {}", path));
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(&mut file)
        .finish()
        .unwrap_or_else(|_| panic!("failed to read {}", path))
}

fn assert_close(case_id: &str, check: &ScalarCheck, actual: f64) {
    assert!(
        actual.is_finite(),
        "{}: {} actual value is not finite: {}",
        case_id,
        check.name,
        actual
    );
    assert!(
        (actual - check.value).abs() <= check.tolerance,
        "{}: {} mismatch: actual={} expected={} tolerance={} diff={}",
        case_id,
        check.name,
        actual,
        check.value,
        check.tolerance,
        (actual - check.value).abs()
    );
}

fn assert_coefficients_close(case_id: &str, fit: &LmeFit, expected: &[ScalarCheck]) {
    let names = fit
        .fixed_names
        .as_ref()
        .unwrap_or_else(|| panic!("{}: fit.fixed_names is None", case_id));
    let coef = fit.coefficients.as_slice().unwrap();
    for check in expected {
        let idx = names
            .iter()
            .position(|n| n == &check.name)
            .unwrap_or_else(|| {
                panic!(
                    "{}: coefficient '{}' not in fixed_names {:?}",
                    case_id, check.name, names
                )
            });
        assert_close(case_id, check, coef[idx]);
    }
}

fn assert_series_close(case_id: &str, label: &str, expected: &[ScalarCheck], actual: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: {} length mismatch: actual={} expected={}",
        case_id,
        label,
        actual.len(),
        expected.len()
    );
    for (idx, check) in expected.iter().enumerate() {
        assert_close(case_id, check, actual[idx]);
    }
}

fn is_named_random_intercept_theta(check: &ScalarCheck) -> bool {
    check.name.ends_with(".(Intercept)") && !check.name.contains("Days")
}

fn theta_group_name(check: &ScalarCheck) -> &str {
    check
        .name
        .strip_suffix(".(Intercept)")
        .unwrap_or_else(|| panic!("{} is not a named random-intercept theta", check.name))
}

fn polars_value_string(value: AnyValue<'_>) -> String {
    value.to_string().trim_matches('"').to_string()
}

fn assert_named_random_intercept_theta(case_id: &str, fit: &LmeFit, expected: &[ScalarCheck]) {
    let sigma2 = fit
        .sigma2
        .unwrap_or_else(|| panic!("{}: named theta checks require sigma2", case_id));
    let var_corr = fit
        .var_corr
        .as_ref()
        .unwrap_or_else(|| panic!("{}: named theta checks require var_corr", case_id));
    let groups = var_corr
        .column("Group")
        .unwrap_or_else(|_| panic!("{}: var_corr missing Group column", case_id));
    let effect1 = var_corr
        .column("Effect1")
        .unwrap_or_else(|_| panic!("{}: var_corr missing Effect1 column", case_id));
    let effect2 = var_corr
        .column("Effect2")
        .unwrap_or_else(|_| panic!("{}: var_corr missing Effect2 column", case_id));
    let variances = var_corr
        .column("Variance")
        .unwrap_or_else(|_| panic!("{}: var_corr missing Variance column", case_id))
        .f64()
        .unwrap_or_else(|_| panic!("{}: var_corr Variance column is not f64", case_id));

    for check in expected {
        let group_name = theta_group_name(check);
        let mut actual = None;
        for row in 0..var_corr.height() {
            let group = polars_value_string(groups.get(row).unwrap());
            let e1 = polars_value_string(effect1.get(row).unwrap());
            let e2 = polars_value_string(effect2.get(row).unwrap());
            if group == group_name && e1 == "(Intercept)" && e2 == "(Intercept)" {
                let variance = variances
                    .get(row)
                    .unwrap_or_else(|| panic!("{}: missing variance for {}", case_id, check.name));
                actual = Some((variance / sigma2).sqrt());
                break;
            }
        }

        assert_close(
            case_id,
            check,
            actual.unwrap_or_else(|| {
                panic!(
                    "{}: could not find random-intercept variance for group {}",
                    case_id, group_name
                )
            }),
        );
    }
}

fn assert_theta_close(case_id: &str, fit: &LmeFit, expected: &[ScalarCheck]) {
    if fit.sigma2.is_some()
        && expected.len() > 1
        && expected.iter().all(is_named_random_intercept_theta)
    {
        assert_named_random_intercept_theta(case_id, fit, expected);
    } else {
        let theta = fit
            .theta
            .as_ref()
            .unwrap_or_else(|| panic!("{}: expected theta but fit.theta is None", case_id));
        assert_series_close(case_id, "theta", expected, theta.as_slice().unwrap());
    }
}

fn family_from_name(name: &str) -> Family {
    match name.to_ascii_lowercase().as_str() {
        "binomial" => Family::Binomial,
        "poisson" => Family::Poisson,
        "gaussian" => Family::Gaussian,
        "gamma" => Family::Gamma,
        other => panic!("unsupported golden parity GLMM family: {}", other),
    }
}

fn link_from_name(name: &str) -> Link {
    Link::parse(name).unwrap_or_else(|err| panic!("invalid golden parity link: {err}"))
}

fn weights_from_column(data: &DataFrame, col: &str) -> Array1<f64> {
    let s = data
        .column(col)
        .unwrap_or_else(|_| panic!("weights column '{col}' missing from data"));
    if let Ok(ca) = s.f64() {
        return Array1::from_iter(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)));
    }
    if let Ok(ca) = s.i64() {
        return Array1::from_iter(ca.into_iter().map(|v| v.unwrap_or(0) as f64));
    }
    panic!("weights column '{col}' must be numeric");
}

fn fit_case(case: &GoldenCase, data: &DataFrame) -> LmeFit {
    match case.kind.as_str() {
        "lmm" => lmer(&case.formula, data, case.reml.unwrap_or(true))
            .unwrap_or_else(|err| panic!("{}: lmer failed: {}", case.id, err)),
        "glmm" => {
            let family = family_from_name(
                case.family
                    .as_deref()
                    .expect("GLMM golden case must specify family"),
            );
            let n_agq = case.n_agq.unwrap_or(1);
            let link = case
                .link
                .as_deref()
                .map(link_from_name)
                .unwrap_or_else(|| Link::default_for(family));
            let weights = case
                .weights_column
                .as_deref()
                .map(|col| weights_from_column(data, col));
            glmer_weighted_with_link(&case.formula, data, family, link, n_agq, weights)
                .unwrap_or_else(|err| panic!("{}: glmer failed: {}", case.id, err))
        }
        "nlmm" => {
            let mut start = lme_rs::NlmmStart::new();
            if let Some(map) = &case.nlmm_start {
                for (k, v) in map {
                    start.insert(k.clone(), *v);
                }
            }
            let opts = lme_rs::NlmerOptions {
                reml: case.nlmm_reml.unwrap_or(false),
                start,
                n_agq: case.n_agq.unwrap_or(1),
                ..lme_rs::NlmerOptions::default()
            };
            lme_rs::nlmer_with_options(&case.formula, data, &opts)
                .unwrap_or_else(|err| panic!("{}: nlmer failed: {}", case.id, err))
        }
        other => panic!("{}: unsupported golden case kind: {}", case.id, other),
    }
}

fn prediction_frame(spec: &PredictionCheck) -> DataFrame {
    let mut columns = Vec::with_capacity(spec.newdata.len());
    let mut expected_len = None;

    for (name, values) in &spec.newdata {
        let array = values
            .as_array()
            .unwrap_or_else(|| panic!("{} newdata column is not an array", name));
        match expected_len {
            Some(len) => assert_eq!(
                len,
                array.len(),
                "prediction newdata columns must all have the same length"
            ),
            None => expected_len = Some(array.len()),
        }

        if array.iter().all(Value::is_number) {
            let vals: Vec<f64> = array
                .iter()
                .map(|v| {
                    v.as_f64()
                        .unwrap_or_else(|| panic!("{} contains a non-f64 number", name))
                })
                .collect();
            columns.push(Series::new(name.as_str().into(), vals).into());
        } else if array.iter().all(Value::is_string) {
            let vals: Vec<String> = array
                .iter()
                .map(|v| {
                    v.as_str()
                        .unwrap_or_else(|| panic!("{} contains a non-string value", name))
                        .to_string()
                })
                .collect();
            columns.push(Series::new(name.as_str().into(), vals).into());
        } else {
            panic!(
                "{} newdata column must contain only numbers or only strings",
                name
            );
        }
    }

    DataFrame::new(columns).expect("failed to build prediction DataFrame")
}

fn assert_predictions(case_id: &str, fit: &LmeFit, checks: &[PredictionCheck], conditional: bool) {
    for check in checks {
        let newdata = prediction_frame(check);
        let actual = if conditional {
            fit.predict_conditional(&newdata, check.allow_new_levels)
                .unwrap_or_else(|err| panic!("{}: conditional prediction failed: {}", case_id, err))
        } else {
            fit.predict(&newdata)
                .unwrap_or_else(|err| panic!("{}: population prediction failed: {}", case_id, err))
        };
        assert_eq!(
            actual.len(),
            check.values.len(),
            "{}: {} prediction length mismatch",
            case_id,
            check.name
        );
        for (idx, expected) in check.values.iter().enumerate() {
            let diff = (actual[idx] - expected).abs();
            assert!(
                diff <= check.tolerance,
                "{}: {} prediction[{}] mismatch: actual={} expected={} tolerance={} diff={}",
                case_id,
                check.name,
                idx,
                actual[idx],
                expected,
                check.tolerance,
                diff
            );
        }
    }
}

fn assert_anova(case_id: &str, fit: &LmeFit, method: DdfMethod, checks: &[AnovaCheck]) {
    let table = fit
        .anova(method)
        .unwrap_or_else(|err| panic!("{}: ANOVA failed: {}", case_id, err));
    assert_eq!(
        table.terms.len(),
        checks.len(),
        "{}: ANOVA row count mismatch",
        case_id
    );

    for (idx, check) in checks.iter().enumerate() {
        assert_eq!(
            table.terms[idx], check.term,
            "{}: ANOVA term mismatch at row {}",
            case_id, idx
        );
        assert_close(case_id, &check.f_value, table.f_value[idx]);
        assert_close(case_id, &check.p_value, table.p_value[idx]);
        assert_close(case_id, &check.num_df, table.num_df[idx]);
        assert_close(case_id, &check.den_df, table.den_df[idx]);
    }
}

fn assert_golden_case(case: &GoldenCase) {
    assert!(
        std::path::Path::new(&case.reference.source_fixture).exists(),
        "{}: referenced source fixture does not exist: {}",
        case.id,
        case.reference.source_fixture
    );

    let data = read_csv_data(&case.data_path);
    let mut fit = fit_case(case, &data);

    if let Some(post_fit) = &case.post_fit {
        if let Some(cluster) = &post_fit.robust_cluster {
            fit.with_robust_se(&data, Some(cluster.as_str()))
                .unwrap_or_else(|err| panic!("{}: robust SE failed: {}", case.id, err));
        }
        if post_fit.satterthwaite {
            fit.with_satterthwaite(&data)
                .unwrap_or_else(|err| panic!("{}: Satterthwaite failed: {}", case.id, err));
        }
        if post_fit.kenward_roger {
            fit.with_kenward_roger(&data)
                .unwrap_or_else(|err| panic!("{}: Kenward-Roger failed: {}", case.id, err));
        }
    }

    let expected = &case.expected;
    assert_coefficients_close(&case.id, &fit, &expected.coefficients);

    if !expected.theta.is_empty() {
        assert_theta_close(&case.id, &fit, &expected.theta);
    }

    if let Some(check) = &expected.sigma2 {
        assert_close(
            &case.id,
            check,
            fit.sigma2
                .unwrap_or_else(|| panic!("{}: expected sigma2 but fit.sigma2 is None", case.id)),
        );
    }

    if let Some(check) = &expected.deviance {
        assert_close(
            &case.id,
            check,
            fit.deviance.unwrap_or_else(|| {
                panic!("{}: expected deviance but fit.deviance is None", case.id)
            }),
        );
    }

    if expected.finite_deviance {
        let deviance = fit
            .deviance
            .unwrap_or_else(|| panic!("{}: expected finite deviance", case.id));
        assert!(
            deviance.is_finite(),
            "{}: expected finite deviance, got {}",
            case.id,
            deviance
        );
    }

    if !expected.beta_se.is_empty() {
        let beta_se = fit
            .beta_se
            .as_ref()
            .unwrap_or_else(|| panic!("{}: expected beta_se but fit.beta_se is None", case.id));
        assert_series_close(
            &case.id,
            "beta_se",
            &expected.beta_se,
            beta_se.as_slice().unwrap(),
        );
    }

    assert_predictions(&case.id, &fit, &expected.population_predictions, false);
    assert_predictions(&case.id, &fit, &expected.conditional_predictions, true);

    if !expected.satterthwaite_anova.is_empty() {
        assert_anova(
            &case.id,
            &fit,
            DdfMethod::Satterthwaite,
            &expected.satterthwaite_anova,
        );
    }

    if !expected.kenward_roger_anova.is_empty() {
        assert_anova(
            &case.id,
            &fit,
            DdfMethod::KenwardRoger,
            &expected.kenward_roger_anova,
        );
    }

    if !expected.robust_se.is_empty() || !expected.robust_t.is_empty() {
        let robust = fit
            .robust
            .as_ref()
            .unwrap_or_else(|| panic!("{}: expected robust results", case.id));
        assert_series_close(
            &case.id,
            "robust_se",
            &expected.robust_se,
            robust.robust_se.as_slice().unwrap(),
        );
        assert_series_close(
            &case.id,
            "robust_t",
            &expected.robust_t,
            robust.robust_t.as_slice().unwrap(),
        );
    }
}

#[test]
fn golden_parity_manifest_is_well_formed() {
    let manifest = load_manifest();
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.reference_environment.engine, "R");
    assert!(
        !manifest.cases.is_empty(),
        "golden parity manifest should contain at least one case"
    );

    for case in &manifest.cases {
        assert!(
            std::path::Path::new(&case.data_path).exists(),
            "{}: data fixture does not exist: {}",
            case.id,
            case.data_path
        );
        assert!(
            std::path::Path::new(&case.reference.source_fixture).exists(),
            "{}: source fixture does not exist: {}",
            case.id,
            case.reference.source_fixture
        );
        assert!(
            !case.expected.coefficients.is_empty(),
            "{}: each golden case should assert coefficients",
            case.id
        );
    }
}

#[test]
fn golden_parity_against_reference_fixtures() {
    let manifest = load_manifest();
    manifest
        .cases
        .par_iter()
        .for_each(|case| assert_golden_case(case));
}
