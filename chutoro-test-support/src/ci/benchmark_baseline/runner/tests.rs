//! Unit tests for helper orchestration.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::{PREFLIGHT_FILE, run_classify, run_stage};
use crate::ci::benchmark_baseline::args::ClassifyArgs;
use crate::ci::benchmark_baseline::manifest::{
    BaselineCase, BaselineManifest, RequiredFiles, ToolVersions, write_manifest,
};
use crate::ci::benchmark_baseline::preflight::PreflightReport;
use crate::ci::benchmark_baseline::{BaselineVerdict, CaseClass, Outcome};

/// Write a complete baseline tree for one identifier.
fn write_baseline(root: &Path, directory: &str, baseline: &str) {
    let path = root.join(directory).join(baseline);
    fs::create_dir_all(&path).expect("baseline directory must be created");
    for file in ["benchmark.json", "estimates.json", "sample.json"] {
        fs::write(path.join(file), "{}").expect("file must be written");
    }
}

/// Write change estimates with a supplied mean interval.
fn write_change(root: &Path, directory: &str, lower: f64, upper: f64) {
    let change = root.join(directory).join("change");
    fs::create_dir_all(&change).expect("change directory must be created");
    fs::write(
        change.join("estimates.json"),
        format!(
            "{{\"mean\":{{\"confidence_interval\":{{\"confidence_level\":0.95,\
             \"lower_bound\":{lower},\"upper_bound\":{upper}}},\"point_estimate\":{lower},\
             \"standard_error\":0.01}}}}"
        ),
    )
    .expect("change estimates must be written");
}

/// Build a manifest for one side of a comparison.
fn manifest(
    root: &Path,
    baseline: &str,
    revision: &str,
    cases: &[(&str, &str)],
) -> BaselineManifest {
    BaselineManifest {
        schema_version: 1,
        revision: revision.to_owned(),
        bench: "hnsw".to_owned(),
        baseline: baseline.to_owned(),
        output_root: root.to_path_buf(),
        tool_versions: ToolVersions {
            criterion: "0.5.1".to_owned(),
            cargo: "cargo 1.99.1".to_owned(),
            rustc: "rustc 1.99.1".to_owned(),
        },
        bench_arguments: vec!["--noplot".to_owned()],
        environment: BTreeMap::new(),
        cases: cases
            .iter()
            .map(|(full_id, directory)| BaselineCase {
                full_id: (*full_id).to_owned(),
                directory_name: (*directory).to_owned(),
                relative_directory: PathBuf::from(directory),
                required_files: RequiredFiles {
                    benchmark: true,
                    estimates: true,
                    sample: true,
                },
            })
            .collect(),
    }
}

/// A staged comparison fixture with one shared identifier.
fn staged_fixture() -> (TempDir, TempDir, TempDir) {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    let evidence = TempDir::new().expect("evidence directory");
    write_baseline(reference.path(), "fixture/alpha", "ci-reference");
    write_baseline(current.path(), "fixture/alpha", "ci-current");
    (reference, current, evidence)
}

/// The two case lists a staged comparison fixture needs.
#[derive(Clone, Copy)]
struct CaseLists<'a> {
    /// Cases the reference tree recorded.
    reference: &'a [(&'a str, &'a str)],
    /// Cases the current tree recorded.
    current: &'a [(&'a str, &'a str)],
}

/// Write both manifests and run staging.
fn stage_fixture(
    reference: &TempDir,
    current: &TempDir,
    evidence: &TempDir,
    cases: CaseLists<'_>,
) -> PreflightReport {
    let reference_cases = cases.reference;
    let current_cases = cases.current;
    let reference_path = evidence.path().join("reference-manifest.json");
    let current_path = evidence.path().join("current-manifest.json");
    write_manifest(
        &reference_path,
        &manifest(reference.path(), "ci-reference", "aaa", reference_cases),
    )
    .expect("reference manifest must be written");
    write_manifest(
        &current_path,
        &manifest(current.path(), "ci-current", "bbb", current_cases),
    )
    .expect("current manifest must be written");

    run_stage(&crate::ci::benchmark_baseline::args::StageArgs {
        bench: "hnsw".to_owned(),
        reference_manifest: reference_path,
        current_manifest: current_path,
        evidence: evidence.path().to_path_buf(),
        filter_output: Some(evidence.path().join("filter.txt")),
    })
    .expect("staging must succeed")
}

/// Classify a staged fixture with a default threshold.
fn classify(evidence: &TempDir) -> BaselineVerdict {
    run_classify(&ClassifyArgs {
        preflight: evidence.path().join(PREFLIGHT_FILE),
        bench: "hnsw".to_owned(),
        verdict: evidence.path().join("verdict.json"),
        threshold: 0.05,
        no_reference: None,
        failed_stage: None,
        failure_reason: None,
    })
    .expect("classification must succeed")
}

#[test]
fn unchanged_measurements_pass() {
    let (reference, current, evidence) = staged_fixture();
    let report = stage_fixture(
        &reference,
        &current,
        &evidence,
        CaseLists {
            reference: &[("fixture/alpha", "fixture/alpha")],
            current: &[("fixture/alpha", "fixture/alpha")],
        },
    );
    write_change(current.path(), "fixture/alpha", -0.01, 0.01);

    let verdict = classify(&evidence);

    assert_eq!(report.outcome, Outcome::Pass);
    assert_eq!(verdict.outcome, Outcome::Pass);
    assert_eq!(verdict.comparable_count, 1);
    assert_eq!(
        verdict.cases.first().map(|case| case.class),
        Some(CaseClass::WithinNoise)
    );
    assert_eq!(verdict.reference_revision.as_deref(), Some("aaa"));
    assert_eq!(verdict.current_revision.as_deref(), Some("bbb"));
}

#[test]
fn regressed_measurements_fail_with_the_regressed_id() {
    let (reference, current, evidence) = staged_fixture();
    stage_fixture(
        &reference,
        &current,
        &evidence,
        CaseLists {
            reference: &[("fixture/alpha", "fixture/alpha")],
            current: &[("fixture/alpha", "fixture/alpha")],
        },
    );
    write_change(current.path(), "fixture/alpha", 0.30, 0.45);

    let verdict = classify(&evidence);

    assert_eq!(verdict.outcome, Outcome::Regression);
    assert_eq!(verdict.regressed_ids(), vec!["fixture/alpha"]);
    let reason = verdict.reason.clone().unwrap_or_default();
    assert!(reason.contains("fixture/alpha"), "{reason}");
}

#[test]
fn current_only_identifier_is_reported_but_does_not_fail() {
    let (reference, current, evidence) = staged_fixture();
    write_baseline(current.path(), "fixture/beta", "ci-current");
    stage_fixture(
        &reference,
        &current,
        &evidence,
        CaseLists {
            reference: &[("fixture/alpha", "fixture/alpha")],
            current: &[
                ("fixture/alpha", "fixture/alpha"),
                ("fixture/beta", "fixture/beta"),
            ],
        },
    );
    write_change(current.path(), "fixture/alpha", -0.01, 0.01);

    let verdict = classify(&evidence);

    assert_eq!(verdict.outcome, Outcome::Pass);
    assert_eq!(verdict.unverified_ids(), vec!["fixture/beta"]);
}

#[test]
fn reference_only_identifier_is_reported_as_removed() {
    let (reference, current, evidence) = staged_fixture();
    write_baseline(reference.path(), "fixture/beta", "ci-reference");
    stage_fixture(
        &reference,
        &current,
        &evidence,
        CaseLists {
            reference: &[
                ("fixture/alpha", "fixture/alpha"),
                ("fixture/beta", "fixture/beta"),
            ],
            current: &[("fixture/alpha", "fixture/alpha")],
        },
    );
    write_change(current.path(), "fixture/alpha", -0.01, 0.01);

    let verdict = classify(&evidence);

    assert_eq!(verdict.outcome, Outcome::Pass);
    let removed: Vec<&str> = verdict
        .cases
        .iter()
        .filter(|case| case.class == CaseClass::Removed)
        .map(|case| case.id.as_str())
        .collect();
    assert_eq!(removed, vec!["fixture/beta"]);
}

#[test]
fn missing_change_estimates_are_incomplete_not_a_pass() {
    let (reference, current, evidence) = staged_fixture();
    stage_fixture(
        &reference,
        &current,
        &evidence,
        CaseLists {
            reference: &[("fixture/alpha", "fixture/alpha")],
            current: &[("fixture/alpha", "fixture/alpha")],
        },
    );

    let verdict = classify(&evidence);

    assert_eq!(verdict.outcome, Outcome::Incomplete);
    assert_eq!(
        verdict.cases.first().map(|case| case.class),
        Some(CaseClass::Unreadable)
    );
    assert!(verdict.reason.is_some());
}

#[test]
fn incompatible_manifests_fail_staging_with_a_preflight_report() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    let evidence = TempDir::new().expect("evidence directory");
    write_baseline(reference.path(), "fixture/alpha", "ci-reference");
    write_baseline(current.path(), "fixture/alpha", "ci-current");
    let reference_path = evidence.path().join("reference-manifest.json");
    let current_path = evidence.path().join("current-manifest.json");
    let mut reference_manifest = manifest(
        reference.path(),
        "ci-reference",
        "aaa",
        &[("fixture/alpha", "fixture/alpha")],
    );
    reference_manifest.tool_versions.criterion = "0.4.0".to_owned();
    write_manifest(&reference_path, &reference_manifest).expect("manifest must be written");
    write_manifest(
        &current_path,
        &manifest(
            current.path(),
            "ci-current",
            "bbb",
            &[("fixture/alpha", "fixture/alpha")],
        ),
    )
    .expect("manifest must be written");

    let error = run_stage(&crate::ci::benchmark_baseline::args::StageArgs {
        bench: "hnsw".to_owned(),
        reference_manifest: reference_path,
        current_manifest: current_path,
        evidence: evidence.path().to_path_buf(),
        filter_output: None,
    })
    .expect_err("incompatible manifests must fail staging");

    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
    let report = PreflightReport::read(&evidence.path().join(PREFLIGHT_FILE))
        .expect("a preflight report must be written on failure");
    assert_eq!(report.outcome, Outcome::Incompatible);
    assert_eq!(report.diagnostics.len(), 1);
}

#[test]
fn no_reference_produces_a_warning_outcome_without_a_pass() {
    let evidence = TempDir::new().expect("evidence directory");
    let verdict = run_classify(&ClassifyArgs {
        preflight: evidence.path().join("unused-preflight.json"),
        bench: "hnsw".to_owned(),
        verdict: evidence.path().join("verdict.json"),
        threshold: 0.05,
        no_reference: Some(crate::ci::benchmark_baseline::args::NoReferenceReason::Shallow),
        failed_stage: None,
        failure_reason: None,
    })
    .expect("a missing reference is an outcome, not a failure");

    assert_eq!(verdict.outcome, Outcome::NoReference);
    assert_eq!(verdict.comparable_count, 0);
    assert_eq!(verdict.cases, Vec::new());
    let remedy = verdict
        .diagnostics
        .first()
        .map(|diagnostic| diagnostic.remedy.clone())
        .unwrap_or_default();
    assert!(remedy.contains("fetch-depth"), "{remedy}");
}

#[test]
fn verdict_is_written_with_the_threshold_and_stage() {
    let (reference, current, evidence) = staged_fixture();
    stage_fixture(
        &reference,
        &current,
        &evidence,
        CaseLists {
            reference: &[("fixture/alpha", "fixture/alpha")],
            current: &[("fixture/alpha", "fixture/alpha")],
        },
    );
    write_change(current.path(), "fixture/alpha", -0.01, 0.01);

    classify(&evidence);

    let verdict = BaselineVerdict::read(&evidence.path().join("verdict.json"))
        .expect("verdict must be readable");
    assert_eq!(verdict.threshold, 0.05);
    assert_eq!(verdict.stage, "classify");
    assert!(verdict.to_markdown().contains("| Outcome | pass |"));
}
