//! Schema lock for the committed nlmer / inference / Python FFI timing artifact.

use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;

const ARTIFACT: &str = "benchmarks/external-nlmm-inference-python-timings-2026-08-14.json";

#[derive(Debug, Deserialize)]
struct Artifact {
    schema_version: u32,
    families_present: Vec<String>,
    reports: Vec<Report>,
}

#[derive(Debug, Deserialize)]
struct Report {
    implementation: String,
    case: String,
    family: String,
    summary: Summary,
}

#[derive(Debug, Deserialize)]
struct Summary {
    median_seconds: f64,
}

#[test]
fn external_timings_artifact_covers_required_families() {
    let file = File::open(ARTIFACT).unwrap_or_else(|err| panic!("open {ARTIFACT}: {err}"));
    let artifact: Artifact =
        serde_json::from_reader(file).unwrap_or_else(|err| panic!("parse {ARTIFACT}: {err}"));
    assert_eq!(artifact.schema_version, 1);

    let families: HashSet<_> = artifact.families_present.iter().cloned().collect();
    for required in ["nlmm_fit", "post_fit_inference", "python_ffi"] {
        assert!(
            families.contains(required),
            "{ARTIFACT} missing family {required}; present={families:?}"
        );
    }

    let mut saw_rust_nlmer = false;
    let mut saw_r_nlmer = false;
    let mut saw_rust_kr = false;
    let mut saw_python = false;
    for report in &artifact.reports {
        assert!(
            report.summary.median_seconds.is_finite() && report.summary.median_seconds > 0.0,
            "{} {} median must be a positive finite duration",
            report.implementation,
            report.case
        );
        if report.implementation == "rust" && report.case == "orange_nlmer" {
            saw_rust_nlmer = true;
        }
        if report.family == "nlmm_fit" && report.implementation.starts_with('r') {
            saw_r_nlmer = true;
        }
        if report.implementation == "rust" && report.case == "sleepstudy_kenward_roger" {
            saw_rust_kr = true;
        }
        if report.family == "python_ffi" {
            saw_python = true;
        }
    }
    assert!(saw_rust_nlmer, "artifact must time rust orange_nlmer");
    assert!(saw_r_nlmer, "artifact must time R nlmer");
    assert!(saw_rust_kr, "artifact must time rust Kenward-Roger ANOVA");
    assert!(saw_python, "artifact must time python lme_python FFI");
}
