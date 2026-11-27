//! Unit tests for transactional baseline staging.

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::{StageFault, StageRequest, build_filter, stage, stage_with_fault};
use crate::ci::benchmark_baseline::BaselineError;
use crate::ci::benchmark_baseline::preflight::ComparableCase;

/// Write one baseline directory with distinguishable file contents.
fn write_baseline(root: &Path, directory: &str, baseline: &str, marker: &str) -> PathBuf {
    let path = root.join(directory).join(baseline);
    fs::create_dir_all(&path).expect("baseline directory must be created");
    for file in ["benchmark.json", "estimates.json", "sample.json"] {
        fs::write(path.join(file), format!("{marker}:{file}")).expect("file must be written");
    }
    path
}

/// Build a comparable case for one shared identifier.
fn comparable(full_id: &str, reference_directory: &str, current_directory: &str) -> ComparableCase {
    ComparableCase {
        full_id: full_id.to_owned(),
        reference_directory: reference_directory.into(),
        current_directory: current_directory.into(),
    }
}

#[test]
fn staging_preserves_nested_hierarchy_and_copies_required_files() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    write_baseline(
        reference.path(),
        "fixture/alpha/with_slash",
        "ci-reference",
        "reference",
    );
    let cases = [comparable(
        "fixture/alpha/with/slash",
        "fixture/alpha/with_slash",
        "fixture/alpha/with_slash_2",
    )];

    let outcome = stage(&StageRequest {
        reference_root: reference.path(),
        current_root: current.path(),
        reference_baseline: "ci-reference",
        comparable: &cases,
    })
    .expect("staging must succeed");

    let destination = current
        .path()
        .join("fixture/alpha/with_slash_2/ci-reference");
    assert!(destination.join("benchmark.json").is_file());
    assert_eq!(
        fs::read_to_string(destination.join("sample.json")).expect("sample must be readable"),
        "reference:sample.json"
    );
    assert_eq!(outcome.staged.len(), 1);
    assert_eq!(
        outcome.staged.first().map(|case| case.full_id.as_str()),
        Some("fixture/alpha/with/slash")
    );
}

#[test]
fn staging_replaces_an_existing_destination() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    write_baseline(reference.path(), "fixture/alpha", "ci-reference", "new");
    write_baseline(current.path(), "fixture/alpha", "ci-reference", "stale");
    let cases = [comparable(
        "fixture/alpha",
        "fixture/alpha",
        "fixture/alpha",
    )];

    stage(&StageRequest {
        reference_root: reference.path(),
        current_root: current.path(),
        reference_baseline: "ci-reference",
        comparable: &cases,
    })
    .expect("staging must succeed");

    let destination = current.path().join("fixture/alpha/ci-reference");
    assert_eq!(
        fs::read_to_string(destination.join("estimates.json")).expect("estimates must be readable"),
        "new:estimates.json"
    );
}

#[test]
fn injected_copy_failure_preserves_the_destination_byte_for_byte() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    write_baseline(reference.path(), "fixture/alpha", "ci-reference", "new");
    let destination = write_baseline(current.path(), "fixture/alpha", "ci-reference", "previous");
    let before: Vec<Vec<u8>> = ["benchmark.json", "estimates.json", "sample.json"]
        .iter()
        .map(|file| fs::read(destination.join(file)).expect("previous file must be readable"))
        .collect();
    let cases = [comparable(
        "fixture/alpha",
        "fixture/alpha",
        "fixture/alpha",
    )];

    let error = stage_with_fault(
        &StageRequest {
            reference_root: reference.path(),
            current_root: current.path(),
            reference_baseline: "ci-reference",
            comparable: &cases,
        },
        StageFault::AfterFirstCopy,
    )
    .expect_err("injected copy failure must fail staging");

    assert!(matches!(error, BaselineError::StageCopy { .. }));
    let after: Vec<Vec<u8>> = ["benchmark.json", "estimates.json", "sample.json"]
        .iter()
        .map(|file| fs::read(destination.join(file)).expect("destination file must remain"))
        .collect();
    assert_eq!(before, after);
}

#[test]
fn missing_source_file_preserves_the_destination() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    let source = write_baseline(reference.path(), "fixture/alpha", "ci-reference", "new");
    fs::remove_file(source.join("sample.json")).expect("sample must be removed");
    let destination = write_baseline(current.path(), "fixture/alpha", "ci-reference", "previous");
    let before = fs::read(destination.join("sample.json")).expect("destination must be readable");
    let cases = [comparable(
        "fixture/alpha",
        "fixture/alpha",
        "fixture/alpha",
    )];

    let error = stage(&StageRequest {
        reference_root: reference.path(),
        current_root: current.path(),
        reference_baseline: "ci-reference",
        comparable: &cases,
    })
    .expect_err("a missing source file must fail staging");

    assert!(matches!(error, BaselineError::MissingRequiredFile { .. }));
    assert!(error.to_string().contains("sample.json"), "{error}");
    let after = fs::read(destination.join("sample.json")).expect("destination must remain");
    assert_eq!(before, after);
}

#[test]
fn empty_source_file_is_rejected() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    let source = write_baseline(reference.path(), "fixture/alpha", "ci-reference", "new");
    fs::write(source.join("estimates.json"), "").expect("empty file must be written");
    let cases = [comparable(
        "fixture/alpha",
        "fixture/alpha",
        "fixture/alpha",
    )];

    let error = stage(&StageRequest {
        reference_root: reference.path(),
        current_root: current.path(),
        reference_baseline: "ci-reference",
        comparable: &cases,
    })
    .expect_err("an empty source file must fail staging");

    assert!(matches!(error, BaselineError::MissingRequiredFile { .. }));
}

#[test]
fn stale_change_directory_is_removed_before_comparison() {
    let reference = TempDir::new().expect("reference directory");
    let current = TempDir::new().expect("current directory");
    write_baseline(reference.path(), "fixture/alpha", "ci-reference", "new");
    let case = comparable("fixture/alpha", "fixture/alpha", "fixture/alpha");
    let change = current.path().join("fixture/alpha/change");
    fs::create_dir_all(&change).expect("stale change directory must be created");
    fs::write(change.join("estimates.json"), "{}").expect("stale estimates must be written");

    stage(&StageRequest {
        reference_root: reference.path(),
        current_root: current.path(),
        reference_baseline: "ci-reference",
        comparable: &[case],
    })
    .expect("staging must succeed");

    assert!(!change.exists());
}

#[test]
fn filter_is_anchored_and_escapes_regex_metacharacters() {
    let filter = build_filter(&[
        "group/a.b".to_owned(),
        "group/(x)+[y]|z?".to_owned(),
        r"group/back\slash".to_owned(),
    ]);

    assert_eq!(
        filter,
        r"^(group/a\.b|group/\(x\)\+\[y\]\|z\?|group/back\\slash)$"
    );
}

#[test]
fn filter_selects_each_identifier_exactly_once() {
    let filter = build_filter(&["fixture/alpha".to_owned(), "fixture/beta".to_owned()]);

    assert!(filter.starts_with("^("));
    assert!(filter.ends_with(")$"));
    assert_eq!(filter.matches('|').count(), 1);
    assert!(filter.contains("fixture/alpha"));
    assert!(filter.contains("fixture/beta"));
}
