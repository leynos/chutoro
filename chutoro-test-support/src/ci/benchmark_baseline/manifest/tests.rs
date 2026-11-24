//! Unit tests for baseline discovery and manifest recording.

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use tempfile::TempDir;

use super::{BaselineManifest, ToolVersions, discover_cases, write_manifest};
use crate::ci::benchmark_baseline::BaselineError;

/// One case to stage for a discovery test.
struct CaseSpec<'a> {
    /// Recorded full identifier.
    full_id: &'a str,
    /// Directory name Criterion assigned.
    directory_name: &'a str,
    /// Whether the baseline carries every required file.
    required: bool,
}

/// Create an identifier directory with a `benchmark.json` and optional files.
fn write_case(root: &Path, baseline: &str, spec: &CaseSpec<'_>) -> PathBuf {
    let full_id = spec.full_id;
    let directory_name = spec.directory_name;
    let required = spec.required;
    let identifier_directory = root.join(directory_name);
    let baseline_directory = identifier_directory.join(baseline);
    fs::create_dir_all(&baseline_directory).expect("baseline directory must be created");
    let benchmark =
        format!("{{\"full_id\":\"{full_id}\",\"directory_name\":\"{directory_name}\"}}");
    fs::write(baseline_directory.join("benchmark.json"), benchmark)
        .expect("benchmark.json must be written");
    if required {
        fs::write(baseline_directory.join("estimates.json"), "{}")
            .expect("estimates.json must be written");
        fs::write(baseline_directory.join("sample.json"), "{}")
            .expect("sample.json must be written");
    }
    baseline_directory
}

#[test]
fn discovery_finds_nested_identifiers_and_records_required_files() {
    let directory = TempDir::new().expect("temporary directory");
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/beta/n=1",
            directory_name: "fixture/beta/n=1",
            required: false,
        },
    );

    let cases =
        discover_cases(directory.path(), "ci-reference", None).expect("discovery must succeed");

    assert_eq!(cases.len(), 2);
    assert_eq!(
        cases.first().map(|case| case.full_id.as_str()),
        Some("fixture/alpha/plain")
    );
    assert!(
        cases
            .first()
            .is_some_and(|case| case.required_files.is_complete())
    );
    assert!(
        cases
            .get(1)
            .is_some_and(|case| !case.required_files.estimates)
    );
    assert_eq!(
        cases.get(1).map(|case| case.relative_directory.clone()),
        Some(PathBuf::from("fixture/beta/n=1"))
    );
}

#[test]
fn discovery_ignores_other_baseline_names() {
    let directory = TempDir::new().expect("temporary directory");
    write_case(
        directory.path(),
        "ci-current",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );

    let cases =
        discover_cases(directory.path(), "ci-reference", None).expect("discovery must succeed");

    assert_eq!(cases, []);
}

#[test]
fn duplicate_identifiers_fail_discovery() {
    let directory = TempDir::new().expect("temporary directory");
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain_2",
            required: true,
        },
    );

    let error = discover_cases(directory.path(), "ci-reference", None)
        .expect_err("duplicate identifiers must fail");

    assert!(matches!(error, BaselineError::DuplicateId { .. }));
    let message = error.to_string();
    assert!(message.contains("fixture/alpha/plain"), "{message}");
}

#[test]
fn malformed_benchmark_json_names_the_path() {
    let directory = TempDir::new().expect("temporary directory");
    let baseline_directory = write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );
    fs::write(baseline_directory.join("benchmark.json"), "{not json")
        .expect("malformed benchmark.json must be written");

    let error = discover_cases(directory.path(), "ci-reference", None)
        .expect_err("malformed metadata must fail");

    let message = error.to_string();
    assert!(message.contains("benchmark.json"), "{message}");
}

#[test]
fn directory_name_mismatch_is_reported() {
    let directory = TempDir::new().expect("temporary directory");
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );
    let other = directory.path().join("fixture/wrong");
    let other_baseline = other.join("ci-reference");
    fs::create_dir_all(&other_baseline).expect("directory must be created");
    fs::write(
        other_baseline.join("benchmark.json"),
        "{\"full_id\":\"fixture/alpha/plain\",\"directory_name\":\"fixture/alpha/plain\"}",
    )
    .expect("benchmark.json must be written");

    let error = discover_cases(directory.path(), "ci-reference", None)
        .expect_err("mismatched directory must fail");

    let message = error.to_string();
    assert!(message.contains("records directory"), "{message}");
}

#[test]
fn stale_baselines_are_rejected_against_the_marker() {
    let directory = TempDir::new().expect("temporary directory");
    let baseline_directory = write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );
    // Give the baseline file a timestamp strictly before the marker that is
    // created afterwards. Filesystem timestamp granularity is coarse on some
    // hosts, so make the gap unambiguous.
    thread::sleep(Duration::from_millis(20));
    let marker = directory.path().join("verdict.json");
    fs::write(&marker, "{}").expect("marker must be written");

    let error = discover_cases(directory.path(), "ci-reference", Some(&marker))
        .expect_err("stale baseline must fail");

    assert!(matches!(error, BaselineError::StaleBaseline { .. }));
    let message = error.to_string();
    assert!(message.contains("older than"), "{message}");
    assert!(baseline_directory.join("benchmark.json").is_file());
}

#[test]
fn fresh_baselines_pass_the_marker_check() {
    let directory = TempDir::new().expect("temporary directory");
    let marker = directory.path().join("verdict.json");
    fs::write(&marker, "{}").expect("marker must be written");
    thread::sleep(Duration::from_millis(20));
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );

    let cases = discover_cases(directory.path(), "ci-reference", Some(&marker))
        .expect("fresh baseline must be accepted");

    assert_eq!(cases.len(), 1);
}

#[test]
fn missing_marker_names_the_marker_path() {
    let directory = TempDir::new().expect("temporary directory");
    write_case(
        directory.path(),
        "ci-reference",
        &CaseSpec {
            full_id: "fixture/alpha/plain",
            directory_name: "fixture/alpha/plain",
            required: true,
        },
    );
    let marker = directory.path().join("missing-marker.json");

    let error = discover_cases(directory.path(), "ci-reference", Some(&marker))
        .expect_err("missing marker must fail");

    let message = error.to_string();
    assert!(message.contains("missing-marker.json"), "{message}");
}

#[test]
fn manifest_is_written_as_readable_json() {
    let directory = TempDir::new().expect("temporary directory");
    let manifest = BaselineManifest {
        schema_version: 1,
        revision: "abc".to_owned(),
        bench: "hnsw".to_owned(),
        baseline: "ci-reference".to_owned(),
        output_root: PathBuf::from("/work/target/criterion"),
        tool_versions: ToolVersions {
            criterion: "0.5.1".to_owned(),
            cargo: "cargo 1.99.1".to_owned(),
            rustc: "rustc 1.99.1".to_owned(),
        },
        bench_arguments: vec!["--noplot".to_owned()],
        environment: std::collections::BTreeMap::new(),
        cases: Vec::new(),
    };
    let path = directory.path().join("evidence/manifest.json");

    write_manifest(&path, &manifest).expect("manifest must be written");

    let text = fs::read_to_string(&path).expect("manifest must be readable");
    assert!(text.contains("\"schema_version\": 1"));
    assert!(text.contains("\"baseline\": \"ci-reference\""));
}
