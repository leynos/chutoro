//! Unit tests for benchmark baseline compatibility rules.

use std::collections::BTreeMap;
use std::path::Path;

use rstest::rstest;

use super::check;
use crate::ci::benchmark_baseline::{
    BaselineCase, BaselineError, BaselineManifest, Outcome, RequiredFiles, ToolVersions,
};

fn case(full_id: &str, complete: bool) -> BaselineCase {
    case_with_files(full_id, complete, complete)
}

fn case_with_files(full_id: &str, estimates: bool, sample: bool) -> BaselineCase {
    BaselineCase {
        full_id: full_id.to_owned(),
        directory_name: full_id.to_owned(),
        relative_directory: full_id.into(),
        required_files: RequiredFiles {
            benchmark: true,
            estimates,
            sample,
        },
    }
}

fn manifest(baseline: &str, cases: Vec<BaselineCase>) -> BaselineManifest {
    BaselineManifest {
        schema_version: 1,
        revision: "abc".to_owned(),
        bench: "hnsw".to_owned(),
        baseline: baseline.to_owned(),
        output_root: Path::new("/work/target/criterion").to_path_buf(),
        tool_versions: ToolVersions {
            criterion: "0.5.1".to_owned(),
            cargo: "cargo 1.99.1".to_owned(),
            rustc: "rustc 1.99.1".to_owned(),
        },
        bench_arguments: vec!["--noplot".to_owned()],
        environment: BTreeMap::new(),
        cases,
    }
}

/// Run `check` on two prepared manifests and return the error.
fn check_pair_error(
    reference_manifest: &BaselineManifest,
    current_manifest: &BaselineManifest,
    expectation: &str,
) -> BaselineError {
    check(reference_manifest, current_manifest).expect_err(expectation)
}

/// Run `check` on two case lists and return the error.
fn check_cases_error(
    reference_cases: Vec<BaselineCase>,
    current_cases: Vec<BaselineCase>,
    expectation: &str,
) -> BaselineError {
    check_pair_error(
        &reference(reference_cases),
        &current(current_cases),
        expectation,
    )
}

/// Assert that a message names every expected fragment.
fn assert_mentions(error: &BaselineError, fragments: &[&str]) {
    let message = error.to_string();
    for fragment in fragments {
        assert!(message.contains(fragment), "{message}");
    }
}

/// Assert that a failure is reported as incompatible.
fn assert_incompatible(error: &BaselineError) {
    assert_eq!(error.outcome(), Some(Outcome::Incompatible));
}

/// Assert that a failure is reported as incomplete.
fn assert_incomplete(error: &BaselineError) {
    assert_eq!(error.outcome(), Some(Outcome::Incomplete));
}

fn reference(cases: Vec<BaselineCase>) -> BaselineManifest {
    manifest("ci-reference", cases)
}

fn current(cases: Vec<BaselineCase>) -> BaselineManifest {
    manifest("ci-current", cases)
}

#[test]
fn shared_identifiers_are_comparable() {
    let compatibility = check(
        &reference(vec![case("fixture/alpha", true)]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect("shared identifiers are compatible");

    assert_eq!(compatibility.comparable.len(), 1);
    assert_eq!(compatibility.new_unverified, Vec::<String>::new());
    assert_eq!(compatibility.removed, Vec::<String>::new());
}

#[test]
fn current_only_identifiers_are_new_unverified() {
    let compatibility = check(
        &reference(vec![case("fixture/alpha", true)]),
        &current(vec![
            case("fixture/alpha", true),
            case("fixture/beta", true),
        ]),
    )
    .expect("a current-only identifier is reported, not fatal");

    assert_eq!(compatibility.new_unverified, vec!["fixture/beta"]);
}

#[test]
fn reference_only_identifiers_are_removed() {
    let compatibility = check(
        &reference(vec![
            case("fixture/alpha", true),
            case("fixture/beta", true),
        ]),
        &current(vec![case("fixture/alpha", true)]),
    )
    .expect("a reference-only identifier is reported, not fatal");

    assert_eq!(compatibility.removed, vec!["fixture/beta"]);
}

/// Mutate the two manifest sides to create one incompatibility.
type Mutator = fn(&mut BaselineManifest, &mut BaselineManifest);

fn criterion_versions_differ(reference: &mut BaselineManifest, current: &mut BaselineManifest) {
    reference.tool_versions.criterion = "0.5.0".to_owned();
    current.tool_versions.criterion = "0.5.1".to_owned();
}

fn criterion_version_is_unsupported(
    reference: &mut BaselineManifest,
    current: &mut BaselineManifest,
) {
    reference.tool_versions.criterion = "0.6.0".to_owned();
    current.tool_versions.criterion = "0.6.0".to_owned();
}

fn rustc_versions_differ(reference: &mut BaselineManifest, current: &mut BaselineManifest) {
    reference.tool_versions.rustc = "rustc 1.99.1".to_owned();
    current.tool_versions.rustc = "rustc 1.100.0".to_owned();
}

fn benchmark_targets_differ(_: &mut BaselineManifest, current: &mut BaselineManifest) {
    current.bench = "mst".to_owned();
}

fn measurement_arguments_differ(_: &mut BaselineManifest, current: &mut BaselineManifest) {
    current.bench_arguments = vec!["--quick".to_owned()];
}

#[rstest]
#[case::criterion_versions(criterion_versions_differ, "criterion version support")]
#[case::unsupported_version(criterion_version_is_unsupported, "0.6.0")]
#[case::rustc_versions(rustc_versions_differ, "rustc -Vv")]
#[case::benchmark_targets(benchmark_targets_differ, "benchmark target")]
#[case::measurement_arguments(measurement_arguments_differ, "benchmark arguments")]
fn incompatible_fields_are_incompatible(#[case] mutate: Mutator, #[case] fragment: &str) {
    let mut reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    mutate(&mut reference_manifest, &mut current_manifest);

    let error = check_pair_error(
        &reference_manifest,
        &current_manifest,
        "an incompatible field must be refused",
    );

    assert_incompatible(&error);
    assert_mentions(&error, &[fragment]);
}

#[test]
fn differing_environment_values_name_the_key() {
    let reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let mut current_manifest = current(vec![case("fixture/alpha", true)]);
    current_manifest.environment.insert(
        "CHUTORO_BENCH_HNSW_MEMORY_PROFILE".to_owned(),
        "1".to_owned(),
    );

    let error = check_pair_error(
        &reference_manifest,
        &current_manifest,
        "different environment values cannot be compared",
    );

    assert_incompatible(&error);
    assert!(
        error
            .to_string()
            .contains("CHUTORO_BENCH_HNSW_MEMORY_PROFILE"),
        "{error}"
    );
}

#[test]
fn identical_baseline_names_are_incompatible() {
    let reference_manifest = reference(vec![case("fixture/alpha", true)]);
    let current_manifest = reference(vec![case("fixture/alpha", true)]);

    let error = check_pair_error(
        &reference_manifest,
        &current_manifest,
        "staging into the current baseline must be refused",
    );

    assert_incompatible(&error);
    assert_mentions(&error, &["baseline names"]);
}

#[test]
fn empty_comparable_set_is_incomplete() {
    let error = check_cases_error(
        vec![case("fixture/alpha", true)],
        vec![case("fixture/beta", true)],
        "no shared identifier cannot produce a comparison",
    );

    assert_incomplete(&error);
    assert!(matches!(error, BaselineError::EmptyComparable { .. }));
    // The empty-comparable message names both baseline names, because the
    // identifiers differ and there is no single one to point at.
    assert_mentions(&error, &["ci-reference", "ci-current"]);
}

#[test]
fn missing_reference_sample_is_incomplete_and_names_the_path() {
    let error = check_cases_error(
        vec![case_with_files("fixture/alpha", true, false)],
        vec![case("fixture/alpha", true)],
        "missing reference sample cannot be compared",
    );

    assert_incomplete(&error);
    assert_mentions(&error, &["sample.json", "ci-reference"]);
}

#[test]
fn missing_reference_estimates_are_incomplete_and_name_the_path() {
    let error = check_cases_error(
        vec![case_with_files("fixture/alpha", false, true)],
        vec![case("fixture/alpha", true)],
        "missing reference estimates cannot be compared",
    );

    assert_incomplete(&error);
    assert_mentions(&error, &["estimates.json"]);
}

#[test]
fn duplicate_identifiers_are_incomplete_and_name_both_paths() {
    let error = check_cases_error(
        vec![case("fixture/alpha", true), case("fixture/alpha", true)],
        vec![case("fixture/alpha", true)],
        "duplicate identifiers cannot be compared",
    );

    assert_incomplete(&error);
    assert!(matches!(error, BaselineError::DuplicateId { .. }));
    assert_mentions(&error, &["benchmark.json"]);
}

#[test]
fn unsupported_schema_is_incompatible() {
    let mut reference_manifest = reference(vec![case("fixture/alpha", true)]);
    reference_manifest.schema_version = 99;
    let current_manifest = current(vec![case("fixture/alpha", true)]);

    let error = check_pair_error(
        &reference_manifest,
        &current_manifest,
        "an unknown schema cannot be compared",
    );

    assert_incompatible(&error);
    assert_mentions(&error, &["schema"]);
}
